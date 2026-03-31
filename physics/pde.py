"""PDE utilities for 2D PINN."""

import torch


def compute_laplacian(model, x):
    """Compute the Laplacian of u(x, y) from a neural network model.

    Args:
        model: Neural network that maps (x, y) -> u.
        x: Input tensor of shape (N, 2), with requires_grad=True.

    Returns:
        Tensor of shape (N, 1): d2u/dx2 + d2u/dy2.
    """
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    # Forward pass: u(x, y)
    u = model(x)  # shape: (N, 1)

    # First derivatives: grad_u[:, 0] = du/dx, grad_u[:, 1] = du/dy
    # Autograd lets us compute derivatives directly from the computation graph.
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    # Second derivative with respect to x: d2u/dx2
    grad_u_x = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_xx = grad_u_x[:, 0:1]

    # Second derivative with respect to y: d2u/dy2
    grad_u_y = torch.autograd.grad(
        outputs=u_y,
        inputs=x,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_yy = grad_u_y[:, 1:2]

    # Laplacian for 2D: d2u/dx2 + d2u/dy2
    laplacian = u_xx + u_yy
    return laplacian


if __name__ == "__main__":
    import os
    import sys

    # Allow running this file directly: python physics/pde.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from models.network import PINNNet

    model = PINNNet()

    x = torch.rand((10, 2), requires_grad=True)
    lap = compute_laplacian(model, x)

    print("Input shape:", tuple(x.shape))
    print("Laplacian shape:", tuple(lap.shape))

