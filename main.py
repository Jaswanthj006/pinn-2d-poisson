"""Single entry point for running the 2D PINN pipeline."""

from training.train import train


def main():
    """Run training from the training module."""
    print("Starting 2D PINN training...")
    train()
    print("Training completed!")


if __name__ == "__main__":
    main()

