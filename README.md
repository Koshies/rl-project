# Autonomous Warehouse Robot Navigation with Reinforcement Learning

This project implements a warehouse navigation environment and trains a reinforcement learning policy that learns to reach a goal while avoiding obstacles and inefficiencies.

## Contents

- `warehouse_rl.py` — environment, PPO-style policy training, and plot generation
- `results/` — generated reward plot, path visualization, and demo assets
- `warehouse_ppo_model.pt` — saved trained model state

## Requirements

Python 3.10+ and the packages below:

- torch
- numpy
- gymnasium
- matplotlib

Install with:

```bash
pip install torch numpy gymnasium matplotlib 
```

## Run the project

From the project root:

```bash
python warehouse_rl.py --episodes 500
```

This will:

1. Train the warehouse RL agent
2. Save training metrics to `results/training_history.csv`
3. Save the reward curve to `results/training_reward_curve.png`
4. Save a demo path image and GIF

## Result summary

The project demonstrates a lightweight RL navigation workflow for a warehouse robot using a grid-based state representation, reward shaping, and a policy network trained to improve over time.
