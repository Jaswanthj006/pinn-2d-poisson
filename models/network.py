"""Neural network model for a 2D Physics-Informed Neural Network (PINN)."""

import torch
import torch.nn as nn


class PINNNet(nn.Module):
    """Fully connected network for mapping (x, y) -> u(x, y)."""

    def __init__(self) -> None:
        super().__init__()

        # Define a simple feed-forward network:
        # Input(2) -> 3 hidden layers (50 neurons each, Tanh) -> Output(1)
        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        # Initialize trainable parameters.
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize linear layers with Xavier uniform weights and zero bias."""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape (N, 2).

        Returns:
            Output tensor of shape (N, 1).
        """
        return self.model(x)


if __name__ == "__main__":
    # Quick local test to verify I/O shapes.
    model = PINNNet()

    x = torch.rand((10, 2))
    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
