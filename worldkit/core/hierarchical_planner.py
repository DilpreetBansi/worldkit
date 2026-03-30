"""Hierarchical planner for long-horizon tasks.

Breaks long-horizon planning into subgoal-based segments by interpolating
in latent space, then uses CEM-style planning to connect consecutive
subgoals with action sequences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .planner import PlanResult


@dataclass
class HierarchicalPlanResult:
    """Result from hierarchical planning.

    Attributes:
        actions: Full concatenated action sequence across all segments.
        subgoals: Latent vectors for each subgoal (including start and goal).
        segment_plans: Individual PlanResult for each segment.
        total_planning_time_ms: Wall-clock planning time in milliseconds.
    """

    actions: list[np.ndarray]
    subgoals: list[torch.Tensor]
    segment_plans: list[PlanResult]
    total_planning_time_ms: float


class HierarchicalPlanner:
    """Hierarchical planner that decomposes long-horizon goals into subgoals.

    Strategy:
    1. Encode current and goal observations into latent space.
    2. Linearly interpolate to create intermediate subgoal latents.
    3. For each consecutive pair of subgoals, run CEM to find an action sequence.
    4. Concatenate all action sequences into a full plan.
    """

    def __init__(
        self,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        n_candidates: int = 200,
        n_elite: int = 20,
        n_iterations: int = 5,
    ):
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iterations = n_iterations

    @torch.no_grad()
    def plan(
        self,
        model,
        context_pixels: torch.Tensor,
        context_actions: torch.Tensor,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        max_subgoals: int = 5,
        steps_per_subgoal: int = 50,
        context_length: int = 1,
        device: str = "cpu",
    ) -> HierarchicalPlanResult:
        """Plan a hierarchical action sequence from current state to goal.

        Args:
            model: The JEPA nn.Module.
            context_pixels: Initial observation frames (B, T_ctx, C, H, W).
            context_actions: Context actions (B, T_ctx, action_dim).
            current_latent: Latent encoding of the current state (D,).
            goal_latent: Latent encoding of the goal state (D,).
            max_subgoals: Number of intermediate subgoals.
            steps_per_subgoal: Planning horizon per segment.
            context_length: Number of context frames.
            device: Device for computation.

        Returns:
            HierarchicalPlanResult with full plan and subgoal info.
        """
        start_time = time.time()

        # Generate subgoal latents via linear interpolation
        # Returns max_subgoals + 2 latents (start + intermediates + goal)
        subgoals = self._interpolate_subgoals(
            current_latent, goal_latent, max_subgoals
        )

        # Plan between consecutive subgoals
        segment_plans: list[PlanResult] = []
        all_actions: list[np.ndarray] = []

        for i in range(len(subgoals) - 1):
            target_latent = subgoals[i + 1]  # (D,)

            segment_result = self._plan_segment(
                model=model,
                context_pixels=context_pixels,
                context_actions=context_actions,
                goal_latent=target_latent,
                planning_horizon=steps_per_subgoal,
                context_length=context_length,
                device=device,
            )

            segment_plans.append(segment_result)
            all_actions.extend(segment_result.actions)

        elapsed_ms = (time.time() - start_time) * 1000

        return HierarchicalPlanResult(
            actions=all_actions,
            subgoals=subgoals,
            segment_plans=segment_plans,
            total_planning_time_ms=elapsed_ms,
        )

    def _interpolate_subgoals(
        self,
        start_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        n_subgoals: int,
    ) -> list[torch.Tensor]:
        """Generate subgoal latents via linear interpolation.

        Returns n_subgoals + 2 latents: [start, sub_1, ..., sub_n, goal].
        """
        subgoals = []
        total_points = n_subgoals + 2
        for i in range(total_points):
            alpha = i / (total_points - 1)
            subgoal = (1 - alpha) * start_latent + alpha * goal_latent
            subgoals.append(subgoal)
        return subgoals

    def _plan_segment(
        self,
        model,
        context_pixels: torch.Tensor,
        context_actions: torch.Tensor,
        goal_latent: torch.Tensor,
        planning_horizon: int,
        context_length: int,
        device: str,
    ) -> PlanResult:
        """Plan actions for one segment using CEM with a latent goal target.

        Similar to CEMPlanner.plan() but scores trajectories against a latent
        goal vector instead of encoding goal pixels.
        """
        segment_start = time.time()

        mean = torch.zeros(planning_horizon, self.action_dim, device=device)
        init_std = (self.action_high - self.action_low) / 3.0
        std = torch.ones(planning_horizon, self.action_dim, device=device) * init_std

        best_actions = None
        best_cost = float("inf")

        for _iteration in range(self.n_iterations):
            # Sample candidate action sequences  # (S, T, action_dim)
            noise = torch.randn(
                self.n_candidates,
                planning_horizon,
                self.action_dim,
                device=device,
            )
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidates = candidates.clamp(self.action_low, self.action_high)
            candidates_4d = candidates.unsqueeze(0)  # (1, S, T, action_dim)

            # Rollout through the model
            trajectory = model.rollout(
                context_pixels,
                context_actions,
                candidates_4d,
                context_length,
            )  # (1, S, T, D)

            # Score: MSE between final predicted state and goal latent
            final_states = trajectory[:, :, -1, :]  # (1, S, D)
            goal_expanded = goal_latent.unsqueeze(0).unsqueeze(0).expand_as(
                final_states
            )
            costs = (
                F.mse_loss(final_states, goal_expanded, reduction="none")
                .mean(dim=-1)
                .squeeze(0)
            )  # (S,)

            # Select elite candidates
            elite_indices = costs.argsort()[: self.n_elite]
            elite_actions = candidates[elite_indices]
            elite_costs = costs[elite_indices]

            # Update sampling distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.01)

            if elite_costs[0].item() < best_cost:
                best_cost = elite_costs[0].item()
                best_actions = elite_actions[0]

        # Get trajectory for the best action sequence
        best_candidates = best_actions.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D_act)
        best_trajectory = model.rollout(
            context_pixels, context_actions, best_candidates, context_length
        )

        elapsed_ms = (time.time() - segment_start) * 1000

        return PlanResult(
            actions=[a.cpu().numpy() for a in best_actions],
            expected_cost=best_cost,
            latent_trajectory=(
                best_trajectory.cpu() if best_trajectory is not None else None
            ),
            success_probability=max(0.0, 1.0 - best_cost),
            planning_time_ms=elapsed_ms,
        )
