"""Sampling utilities for 2D PINN training points."""

import torch


def sample_interior(n_points):
    """Sample random interior points in [0, 1] x [0, 1].

    Interior points are used to enforce the PDE residual.
    """
    return torch.rand((n_points, 2))


def sample_boundary(n_points):
    """Sample boundary points on the square edges.

    Boundary points are used to enforce boundary conditions.
    """
    # Split points as evenly as possible across 4 edges.
    counts = [n_points // 4] * 4
    for i in range(n_points % 4):
        counts[i] += 1

    # Edge y = 0
    x_bottom = torch.rand((counts[0], 1))
    y_bottom = torch.zeros((counts[0], 1))
    bottom = torch.cat([x_bottom, y_bottom], dim=1)

    # Edge y = 1
    x_top = torch.rand((counts[1], 1))
    y_top = torch.ones((counts[1], 1))
    top = torch.cat([x_top, y_top], dim=1)

    # Edge x = 0
    x_left = torch.zeros((counts[2], 1))
    y_left = torch.rand((counts[2], 1))
    left = torch.cat([x_left, y_left], dim=1)

    # Edge x = 1
    x_right = torch.ones((counts[3], 1))
    y_right = torch.rand((counts[3], 1))
    right = torch.cat([x_right, y_right], dim=1)

    boundary = torch.cat([bottom, top, left, right], dim=0)

    # Optional shuffle to avoid grouped edges in one block.
    boundary = boundary[torch.randperm(boundary.shape[0])]

    return boundary


if __name__ == "__main__":
    interior = sample_interior(100)
    boundary = sample_boundary(100)

    print("Interior shape:", tuple(interior.shape))
    print("Boundary shape:", tuple(boundary.shape))

