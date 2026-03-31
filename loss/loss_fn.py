"""Loss functions for training a 2D Physics-Informed Neural Network (PINN)."""

import math
import os
import sys

import torch
import torch.nn as nn

# Make imports work when running this file directly: python loss/loss_fn.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics.pde import compute_laplacian

# Define MSE once and reuse it in all loss functions.
mse_loss = nn.MSELoss()


def compute_pde_loss(model, interior_points):
    """PDE loss: enforces the governing physics inside the domain."""
    # We need gradients with respect to x and y for autograd derivatives.
    interior_points = interior_points.requires_grad_(True)

    lap = compute_laplacian(model, interior_points)

    x = interior_points[:, 0:1]
    y = interior_points[:, 1:2]

    # RHS for: Δu = -2*pi^2*sin(pi*x)*sin(pi*y)
    rhs = -2 * (math.pi ** 2) * torch.sin(math.pi * x) * torch.sin(math.pi * y)

    loss = mse_loss(lap, rhs)
    return loss


def compute_boundary_loss(model, boundary_points):
    """Boundary loss: enforces boundary condition u=0 on the domain edges."""
    u = model(boundary_points)
    zeros = torch.zeros_like(u)
    loss = mse_loss(u, zeros)
    return loss


def total_loss(model, interior_points, boundary_points):
    """Total PINN loss = PDE loss + boundary loss."""
    pde = compute_pde_loss(model, interior_points)
    bc = compute_boundary_loss(model, boundary_points)
    total = pde + bc
    return total, pde, bc


if __name__ == "__main__":
    from data.sampler import sample_boundary, sample_interior
    from models.network import PINNNet

    model = PINNNet()

    interior = sample_interior(100)
    boundary = sample_boundary(100)

    total, pde, bc = total_loss(model, interior, boundary)

    print("Total loss:", float(total.item()))
    print("PDE loss:", float(pde.item()))
    print("Boundary loss:", float(bc.item()))

