"""Tests for loss functions."""

import torch

from worldkit.core.losses import SIGReg, prediction_loss, worldkit_loss


def test_sigreg():
    sigreg = SIGReg(knots=17, num_proj=256)
    z = torch.randn(4, 8, 192)  # (T, B, D)
    loss = sigreg(z)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0


def test_prediction_loss():
    pred = torch.randn(8, 10, 192)
    target = torch.randn(8, 10, 192)
    loss = prediction_loss(pred, target)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_worldkit_loss():
    pred = torch.randn(8, 10, 192)
    target = torch.randn(8, 10, 192)
    z = torch.randn(8, 3, 192)
    total, loss_dict = worldkit_loss(pred, target, z, lambda_reg=1.0)
    assert total.dim() == 0
    assert "loss/total" in loss_dict
    assert "loss/prediction" in loss_dict
    assert "loss/sigreg" in loss_dict
