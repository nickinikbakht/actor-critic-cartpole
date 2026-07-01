# Actor-Critic Reinforcement Learning on CartPole-v1

Implementation of the **Advantage Actor-Critic (A2C)** algorithm from scratch in PyTorch, trained on OpenAI Gym's `CartPole-v1` environment. The project compares two update strategies for the same actor-critic architecture: a **one-step** update and an **n-step** update.

## Overview

Actor-Critic methods combine two neural networks:

- **Actor** — a policy network that outputs a probability distribution over actions given a state.
- **Critic** — a value network that estimates how good a given state is, and is used to compute the advantage signal that trains the actor.

This repository implements both networks and two training loops:

1. **One-step Actor-Critic** — the networks are updated after every single environment step, using the immediate reward and the critic's estimate of the next state.
2. **N-step Actor-Critic** — transitions are stored in a memory buffer and the networks are updated every `max_steps` steps (or at the end of an episode), using n-step returns computed from the stored rewards.

## Project Structure

```
.
└── RL.ipynb   # Full implementation: networks, memory buffer, training loops, and result plots
```

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [OpenAI Gym](https://www.gymlibrary.dev/)
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install torch gym numpy matplotlib
```

## Usage

Open and run `RL.ipynb` in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook RL.ipynb
```

The notebook is organized into the following sections:

1. Import libraries
2. Build the actor and critic networks
3. Initialize the environment and hyperparameters
4. Train and evaluate the one-step algorithm
5. Train and evaluate the n-step algorithm

Each training section runs for 400 episodes on `CartPole-v1` and plots the total reward per episode at the end.

## Method Summary

- **Environment:** `CartPole-v1` (OpenAI Gym)
- **Actor network:** 2 hidden layers (64 → 32 units, Tanh activations), Softmax output over actions
- **Critic network:** 2 hidden layers (64 → 32 units, ReLU activations), single scalar output (state value)
- **Optimizer:** Adam (`lr=1e-3`) for both networks
- **Discount factor (γ):** 0.99
- **Advantage estimation:**
  - One-step: `advantage = reward + γ * (1 - done) * V(next_state) - V(state)`
  - N-step: bootstrapped n-step returns computed over a buffer of stored transitions

## Results

Training curves (total reward per episode) for both the one-step and n-step algorithms are included as saved outputs in the notebook.

## License

This project is open source and available under the [MIT License](LICENSE).
