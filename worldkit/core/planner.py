"""CEM (Cross-Entropy Method) planner for world models.

Optimizes action sequences by sampling candidates, evaluating them
via latent rollouts, and iteratively refining around the best ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PlanResult:
    """Result from the CEM planner."""

    actions: list
    expected_cost: float
    latent_trajectory: torch.Tensor | None
    success_probability: float
    planning_time_ms: float


class CEMPlanner:
    """Cross-Entropy Method planner for goal-conditioned action optimization.

    Searches for optimal action sequences by:
    1. Sampling N random action sequences
    2. Rolling out each through the world model in latent space
    3. Scoring by latent distance to goal
    4. Keeping top K (elite) sequences
    5. Re-sampling around the elite mean/std
    6. Repeating for M iterations
    """

    def __init__(
        self,
        action_dim: int = 2,
        action_low: float = -1.0,
        action_high: float = 1.0,
        n_candidates: int = 200,
        n_elite: int = 20,
        n_iterations: int = 5,
        planning_horizon: int = 50,
    ):
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.planning_horizon = planning_horizon

    @torch.no_grad()
    def plan(
        self,
        model,
        context_pixels: torch.Tensor,
        context_actions: torch.Tensor,
        goal_pixels: torch.Tensor,
        context_length: int = 3,
        device: str = "cpu",
    ) -> PlanResult:
        """Plan an action sequence to reach the goal."""
        import time

        start_time = time.time()

        B = context_pixels.shape[0]
        assert B == 1, "CEM planner currently supports batch size 1"

        mean = torch.zeros(self.planning_horizon, self.action_dim, device=device)
        std = torch.ones(self.planning_horizon, self.action_dim, device=device) * 0.5

        best_actions = None
        best_cost = float("inf")

        for iteration in range(self.n_iterations):
            noise = torch.randn(
                self.n_candidates,
                self.planning_horizon,
                self.action_dim,
                device=device,
            )
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidates = candidates.clamp(self.action_low, self.action_high)
            candidates = candidates.unsqueeze(0)

            costs = model.get_cost(
                pixels=context_pixels,
                actions=context_actions,
                goal_pixels=goal_pixels,
                action_candidates=candidates,
                context_length=context_length,
            )

            costs = costs.squeeze(0)

            elite_indices = costs.argsort()[: self.n_elite]
            elite_actions = candidates[0, elite_indices]
            elite_costs = costs[elite_indices]

            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.01)

            if elite_costs[0].item() < best_cost:
                best_cost = elite_costs[0].item()
                best_actions = elite_actions[0]

        best_candidates = best_actions.unsqueeze(0).unsqueeze(0)
        best_trajectory = model.rollout(
            context_pixels, context_actions, best_candidates, context_length
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return PlanResult(
            actions=[a.cpu().numpy() for a in best_actions],
            expected_cost=best_cost,
            latent_trajectory=best_trajectory.cpu() if best_trajectory is not None else None,
            success_probability=max(0.0, 1.0 - best_cost),
            planning_time_ms=elapsed_ms,
        )
