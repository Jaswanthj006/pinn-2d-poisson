"""Training script for a 2D Physics-Informed Neural Network (PINN)."""

import os
import sys
import math

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# Make imports work when running this file directly: python training/train.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.sampler import sample_boundary, sample_interior
from loss.loss_fn import total_loss
from models.network import PINNNet


def train_pinn():
    """Train the PINN model and return training history."""
    # Initialize model and optimizer.
    model = PINNNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training setup.
    epochs = 2000
    n_interior = 1000
    n_boundary = 400

    # Store total loss for plotting.
    loss_history = []

    model.train()
    for epoch in range(epochs):
        # Sample fresh interior and boundary points each epoch.
        interior = sample_interior(n_interior)
        boundary = sample_boundary(n_boundary)

        # Compute PINN losses.
        loss, pde, bc = total_loss(model, interior, boundary)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Total: {loss.item():.6f} | "
                f"PDE: {pde.item():.6f} | "
                f"BC: {bc.item():.6f}"
            )

    return model, loss_history


def predict_on_grid(model, grid_size=50):
    """Evaluate the trained model on a regular [0, 1] x [0, 1] grid."""
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # Flatten grid points into shape (N, 2).
    grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    model.eval()
    with torch.no_grad():
        u_pred = model(grid_points)

    # Reshape predicted values to 2D for surface plotting.
    u_grid = u_pred.reshape(grid_size, grid_size)

    return xx, yy, u_grid


def plot_results(loss_history, xx, yy, u_pred, u_exact, u_error):
    """Plot loss curve plus predicted, exact, and error surfaces."""
    # Plot 1: Loss curve.
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 2: Predicted solution surface.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx.numpy(),
        yy.numpy(),
        u_pred.numpy(),
        cmap="viridis",
        edgecolor="none",
    )
    ax.set_title("Predicted Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    plt.tight_layout()

    # Plot 3: Exact solution surface.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx.numpy(),
        yy.numpy(),
        u_exact.numpy(),
        cmap="viridis",
        edgecolor="none",
    )
    ax.set_title("Exact Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    plt.tight_layout()

    # Plot 4: Absolute error surface.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx.numpy(),
        yy.numpy(),
        u_error.numpy(),
        cmap="magma",
        edgecolor="none",
    )
    ax.set_title("Error Surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("|error|")
    plt.tight_layout()

    plt.show()


def train():
    """Run full training and visualization pipeline."""
    model, loss_history = train_pinn()
    xx, yy, u_pred = predict_on_grid(model, grid_size=50)

    # Exact solution for comparison: u(x, y) = sin(pi*x) * sin(pi*y)
    u_exact = torch.sin(math.pi * xx) * torch.sin(math.pi * yy)

    # Absolute error and MAE across all grid points.
    u_error = torch.abs(u_pred - u_exact)
    mae = torch.mean(u_error)
    print("Mean Absolute Error:", mae.item())

    plot_results(loss_history, xx, yy, u_pred, u_exact, u_error)

