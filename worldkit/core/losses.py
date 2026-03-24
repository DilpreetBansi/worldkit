from __future__ import annotations

"""Loss functions for WorldKit world models.

Implements:
- SIGReg: Sketch Isotropic Gaussian Regularizer (prevents collapse with 1 hyperparameter)
- Prediction loss: MSE between predicted and actual next-state embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Enforces a Gaussian distribution on the latent space to prevent
    representation collapse. Uses random projections and a trigonometric basis
    to compute a smooth approximation of the KL divergence between the
    empirical distribution of latent representations and a unit Gaussian.

    Args:
        knots: Number of knot points for the trigonometric basis. Default 17.
        num_proj: Number of random projections for the sketch. Default 1024.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.knots = knots
        self.num_proj = num_proj

        t = torch.linspace(-5, 5, knots)
        self.register_buffer("t", t)

        phi = torch.zeros(knots)
        for i in range(knots):
            phi[i] = torch.exp(-0.5 * t[i] ** 2)
        self.register_buffer("phi", phi)

        weights = torch.ones(knots) * (t[1] - t[0])
        weights[0] = weights[0] / 2
        weights[-1] = weights[-1] / 2
        self.register_buffer("weights", weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            z: Latent representations, shape (T, B, D) or (B, D)

        Returns:
            Scalar loss value.
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)

        T, B, D = z.shape
        z_flat = z.reshape(-1, D)

        if not hasattr(self, "_proj") or self._proj.shape != (self.num_proj, D):
            proj = torch.randn(self.num_proj, D, device=z.device)
            proj = proj / proj.norm(dim=-1, keepdim=True)
            self._proj = proj

        projected = z_flat @ self._proj.T

        mean = projected.mean(dim=0, keepdim=True)
        std = projected.std(dim=0, keepdim=True).clamp(min=1e-6)
        projected = (projected - mean) / std

        loss = 0.0
        for i in range(self.knots):
            empirical_cf = torch.cos(self.t[i] * projected).mean(dim=0)
            target_cf = self.phi[i]
            loss = loss + self.weights[i] * ((empirical_cf - target_cf) ** 2).mean()

        var_loss = ((std.squeeze() - 1.0) ** 2).mean()
        mean_loss = (mean.squeeze() ** 2).mean()

        return loss + var_loss + mean_loss


def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute prediction loss between predicted and actual next-state embeddings."""
    return F.mse_loss(predicted, target.detach())


def worldkit_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    latent_z: torch.Tensor,
    lambda_reg: float = 1.0,
    sigreg: SIGReg | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the full WorldKit/LeWM loss: L = L_pred + lambda * SIGReg(Z)."""
    pred_l = prediction_loss(predicted, target)

    if sigreg is None:
        sigreg = SIGReg().to(predicted.device)

    if latent_z.dim() == 3 and latent_z.shape[0] != latent_z.shape[1]:
        z_for_sigreg = latent_z.permute(1, 0, 2)
    else:
        z_for_sigreg = latent_z

    sig_l = sigreg(z_for_sigreg)
    total = pred_l + lambda_reg * sig_l

    loss_dict = {
        "loss/total": total.item(),
        "loss/prediction": pred_l.item(),
        "loss/sigreg": sig_l.item(),
    }

    return total, loss_dict
