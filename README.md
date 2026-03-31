# 2D PINN for Poisson Equation

This is my 2D Physics-Informed Neural Network project.  
I built it to solve a Poisson equation on a unit square using a neural network instead of a standard numerical solver.

## What problem I solved

I solve this PDE on the domain `[0, 1] x [0, 1]`:

`∇²u(x, y) = -2π² sin(πx) sin(πy)`

Boundary condition on all edges:

`u(x, y) = 0`

Exact solution (used for comparison):

`u(x, y) = sin(πx) sin(πy)`

## My approach

I used a fully connected neural network that takes `(x, y)` and predicts `u(x, y)`.

During training:
- I sampled interior points for the PDE loss
- I sampled boundary points for the boundary condition loss
- I used PyTorch autograd to compute first and second derivatives
- I manually built the Laplacian term `u_xx + u_yy`
- I trained with total loss = PDE loss + boundary loss

After training:
- I compared prediction with the exact solution
- I computed MAE
- I plotted predicted surface, exact surface, error surface, and loss curve

## Project structure

- `models/` -> neural network
- `data/` -> point sampling
- `physics/` -> Laplacian computation
- `loss/` -> PDE + boundary losses
- `training/` -> training loop and plots
- `main.py` -> single entry point

## How to run

From project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

To leave the environment later:

```bash
deactivate
```

## Results

The model learns the overall shape of the solution pretty well.  
My Mean Absolute Error is around **0.04** (can change a bit between runs).

## Observations

Training is stable, but it is not super fast on CPU.  
Loss generally goes down, but the curve can be noisy.  
Sampling quality matters more than I first expected.

## What I learned

I got a much better understanding of how PINNs work in practice.  
I learned how to use autograd for higher-order derivatives, not just normal backprop.  
I also learned that balancing PDE and boundary losses is important if I want better accuracy.

