"""Warehouse robot navigation with PPO reinforcement learning.

This project implements a lightweight warehouse navigation environment and trains a
policy to route a robot to a goal while avoiding walls and obstacles.
"""

from __future__ import annotations

# Standard Python utilities used for CLI handling, CSV export, and filesystem paths.
import argparse
import csv
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Visualization and plotting support for training curves and demo maps.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Core RL / neural network dependencies.
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Env, spaces
from torch.distributions import Categorical

# Optional utilities for rendering the demo gif and exporting the slide deck.
from PIL import Image, ImageDraw

# Project output paths: results folder, saved model, and final presentation file.
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODEL_FILE = ROOT / "warehouse_ppo_model.pt"


class WarehouseEnv(Env):
    """A compact 2D warehouse environment for a robot navigation task."""

    # Gymnasium environment metadata used to describe supported rendering modes.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size: int = 8, seed: int | None = None):
        super().__init__()
        self.grid_size = grid_size

        # Action space: move north, east, south, west, or stay in place.
        self.action_space = spaces.Discrete(5)

        # Observation: relative goal direction, agent position, time progress,
        # and local obstacle sensor values around the robot.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )

        # Warehouse map layout: a goal location, start position, and blocked cells.
        self.goal = np.array([grid_size - 2, grid_size - 2], dtype=int)
        self.start = np.array([1, 1], dtype=int)
        self.walls = {
            (2, 1), (2, 2), (2, 3), (2, 5), (3, 5), (4, 2), (4, 3), (4, 5),
            (5, 2), (5, 3), (5, 4), (1, 5), (6, 1), (6, 5), (4, 6)
        }
        self.agent_pos = self.start.copy()
        self.steps_taken = 0
        self.rng = random.Random(seed)
        self.seed_value = seed

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _wall_at(self, pos: np.ndarray) -> bool:
        return tuple(pos.tolist()) in self.walls

    def _distance_to_goal(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.goal))

    def _build_observation(self) -> np.ndarray:
        # Relative goal vector helps the robot understand where the target is.
        rel = self.goal - self.agent_pos

        # Local sensor checks for nearby wall/blockage in four directions.
        sensor_vectors = {
            0: np.array([0, -1]),  # north
            1: np.array([1, 0]),   # east
            2: np.array([0, 1]),   # south
            3: np.array([-1, 0]),  # west
        }

        sensor_values = []
        for direction in range(4):
            vec = sensor_vectors[direction]
            probe = self.agent_pos.copy()
            blocked = 1.0
            for _ in range(2):
                probe = probe + vec
                if not self._in_bounds(probe):
                    blocked = 1.0
                    break
                if self._wall_at(probe):
                    blocked = 1.0
                    break
                blocked = 0.0
            sensor_values.append(blocked)

        # Final observation vector contains the goal direction, position, time, and nearby obstacle signals.
        obs = np.array([
            rel[0] / self.grid_size,
            rel[1] / self.grid_size,
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.steps_taken / 40.0,
            sensor_values[0],
            sensor_values[1],
            sensor_values[2],
        ], dtype=np.float32)

        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Reset the environment to a new start state and random goal location.
        if seed is None:
            seed = self.rng.randrange(0, 10_000)
        self.seed_value = seed
        self.rng.seed(seed)

        self.agent_pos = self.start.copy()
        self.steps_taken = 0
        self.goal = np.array([
            self.rng.randint(5, self.grid_size - 2),
            self.rng.randint(5, self.grid_size - 2),
        ], dtype=int)
        return self._build_observation(), {}

    def step(self, action: int):
        # Keep the simulation deterministic and safe by validating the action before movement.
        assert 0 <= action < self.action_space.n, f"Invalid action {action}"
        self.steps_taken += 1

        # Map each action to a movement vector.
        move_map = {
            0: np.array([0, -1]),  # north
            1: np.array([1, 0]),   # east
            2: np.array([0, 1]),   # south
            3: np.array([-1, 0]),  # west
            4: np.array([0, 0]),   # stay
        }

        prev_dist = self._distance_to_goal(self.agent_pos)
        delta = move_map[action]
        next_pos = self.agent_pos + delta

        # Reward shaping: small step penalty, strong collision penalty, and progress bonus.
        reward = -0.08
        collision = False

        if not self._in_bounds(next_pos):
            collision = True
        elif self._wall_at(next_pos):
            collision = True
        else:
            self.agent_pos = next_pos

        if collision:
            reward -= 2.0

        new_dist = self._distance_to_goal(self.agent_pos)
        reward += 0.25 * (prev_dist - new_dist)

        goal_reached = np.array_equal(self.agent_pos, self.goal)
        if goal_reached:
            reward += 6.0

        # Episode ends when the robot reaches the goal or exceeds the step limit.
        terminated = goal_reached or self.steps_taken >= 40
        truncated = self.steps_taken >= 40 and not goal_reached
        if truncated:
            reward -= 1.0

        observation = self._build_observation()
        info = {"goal": self.goal.copy(), "position": self.agent_pos.copy()}
        return observation, float(reward), terminated, truncated, info

    def render(self, mode: str = "human"):
        grid = np.full((self.grid_size, self.grid_size), 0, dtype=np.uint8)
        for x, y in self.walls:
            grid[y, x] = 1
        grid[self.goal[1], self.goal[0]] = 2
        grid[self.agent_pos[1], self.agent_pos[0]] = 3

        if mode == "human":
            chars = {0: ".", 1: "#", 2: "G", 3: "R"}
            print("\n" + "\n".join(" ".join(chars.get(cell, ".") for cell in row) for row in grid))
        return grid


@dataclass
class RolloutMemory:
    # Temporary replay buffer for PPO-style training.
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self):
        # Clear all collected rollout data after one update step.
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.values.clear()
        self.dones.clear()


class ActorCritic(nn.Module):
    # A simple policy-value network: the actor produces action probabilities,
    # while the critic estimates the expected return for each state.
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        hidden = self.shared(x)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value.squeeze(-1)


def compute_gae(rewards: List[float], values: List[float], dones: List[bool], gamma: float = 0.99, lam: float = 0.95):
    # Generalized Advantage Estimation (GAE) is a standard PPO-style approach
    # to reduce variance while keeping the training signal stable.
    advantages = []
    returns = []
    gae = 0.0
    value_arr = np.asarray(values, dtype=np.float32)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    done_arr = np.asarray(dones, dtype=np.float32)

    for step in reversed(range(len(rewards))):
        delta = reward_arr[step] + gamma * (1.0 - done_arr[step]) * (value_arr[step + 1] if step + 1 < len(values) else 0.0) - value_arr[step]
        gae = delta + gamma * lam * (1.0 - done_arr[step]) * gae
        advantages.append(gae)

    advantages.reverse()
    adv_array = np.asarray(advantages, dtype=np.float32)
    ret_array = adv_array + value_arr[: len(advantages)]
    return adv_array, ret_array


def train_agent(env: WarehouseEnv, episodes: int = 700, epochs_per_update: int = 10):
    # Train the PPO-like policy using rollout data and policy-gradient updates.
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    memory = RolloutMemory()
    history: List[float] = []

    for episode in range(1, episodes + 1):
        # Start a fresh training episode with a random goal layout.
        observation, _ = env.reset(seed=episode)
        episode_reward = 0.0
        done = False

        while not done:
            # Use the current policy to sample an action from the probability distribution.
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits, value = model(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

            # Apply the chosen action and collect the environment feedback.
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Save the experience in the rollout buffer for policy updates.
            memory.obs.append(observation.copy())
            memory.actions.append(action.item())
            memory.rewards.append(reward)
            memory.logprobs.append(logprob.item())
            memory.values.append(value.item())
            memory.dones.append(done)

            observation = next_obs
            episode_reward += reward

        history.append(episode_reward)

        # Perform a PPO-style optimization update when enough rollout data is available.
        if len(memory.obs) >= 32:
            obs_array = np.asarray(memory.obs, dtype=np.float32)
            actions = torch.tensor(memory.actions, dtype=torch.int64)
            old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
            values = np.asarray(memory.values, dtype=np.float32)
            rewards = np.asarray(memory.rewards, dtype=np.float32)
            dones = np.asarray(memory.dones, dtype=np.float32)

            returns = np.zeros_like(rewards)
            adv = np.zeros_like(rewards)
            running_return = 0.0
            running_adv = 0.0

            for i in reversed(range(len(rewards))):
                running_return = rewards[i] + 0.99 * (1.0 - dones[i]) * running_return
                returns[i] = running_return
                next_value = values[i + 1] if i + 1 < len(values) else 0.0
                delta = rewards[i] + 0.99 * (1.0 - dones[i]) * next_value - values[i]
                running_adv = delta + 0.95 * 0.99 * (1.0 - dones[i]) * running_adv
                adv[i] = running_adv

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            obs_t = torch.as_tensor(obs_array, dtype=torch.float32)
            act_t = actions
            old_logp = old_logprobs
            adv_t = torch.tensor(adv, dtype=torch.float32)
            ret_t = torch.tensor(returns, dtype=torch.float32)

            for _ in range(epochs_per_update):
                # Actor-critic update: policy loss + critic loss.
                logits, pred_values = model(obs_t)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t)
                ratio = torch.exp(new_logp - old_logp)
                clipped_adv = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_t
                actor_loss = -(torch.min(ratio * adv_t, clipped_adv)).mean()
                critic_loss = F.mse_loss(pred_values, ret_t)
                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            memory.clear()

        if episode % 100 == 0:
            avg_reward = float(np.mean(history[-100:]))
            print(f"Episode {episode}: avg reward over last 100 = {avg_reward:.2f}")

    return model, history


def run_policy(env: WarehouseEnv, model: ActorCritic, seed: int = 42, max_steps: int = 40):
    # Evaluate the trained model on a single deterministic run and record the path taken.
    observation, _ = env.reset(seed=seed)
    path = [tuple(env.agent_pos.tolist())]
    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(obs_tensor)
        action = torch.argmax(logits, dim=-1).item()
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        path.append(tuple(env.agent_pos.tolist()))
        steps += 1
        done = terminated or truncated

    return total_reward, path


def save_reward_plot(history: List[float], filename: Path):
    # Plot cumulative reward per episode to visualize learning progress.
    plt.figure(figsize=(10, 5))
    plt.plot(history, color="#1f77b4", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Warehouse Navigation Reward Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_demo_grid(path: List[Tuple[int, int]], grid_size: int, filename: Path):
    # Visualize the agent's route through the warehouse grid, marking walls, start, goal, and the path taken.
    grid = np.full((grid_size, grid_size), 0, dtype=np.uint8)
    walls = {
        (2, 1), (2, 2), (2, 3), (2, 5), (3, 5), (4, 2), (4, 3), (4, 5),
        (5, 2), (5, 3), (5, 4), (1, 5), (6, 1), (6, 5), (4, 6)
    }
    for x, y in walls:
        grid[y, x] = 1

    for idx, (x, y) in enumerate(path):
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[y, x] = 2 if idx == 0 else 3

    goal = (grid_size - 2, grid_size - 2)
    grid[goal[1], goal[0]] = 4

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(grid_size):
        for j in range(grid_size):
            color = "white"
            if grid[i, j] == 1:
                color = "#4b5563"
            elif grid[i, j] == 2:
                color = "#22c55e"
            elif grid[i, j] == 3:
                color = "#f59e0b"
            elif grid[i, j] == 4:
                color = "#ef4444"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="#d1d5db", linewidth=1))

    xs = [x for x, _ in path]
    ys = [y for _, y in path]
    ax.plot(xs, ys, color="#2563eb", linewidth=2.5, alpha=0.9)
    ax.scatter([xs[0]], [ys[0]], color="#f59e0b", s=80, zorder=5)
    ax.scatter([goal[0]], [goal[1]], color="#ef4444", s=80, zorder=5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_demo_gif(path: List[Tuple[int, int]], grid_size: int, filename: Path):
    # Convert the path into a short GIF animation showing the robot's movement through the warehouse grid.
    frames: List[Image.Image] = []
    walls = {
        (2, 1), (2, 2), (2, 3), (2, 5), (3, 5), (4, 2), (4, 3), (4, 5),
        (5, 2), (5, 3), (5, 4), (1, 5), (6, 1), (6, 5), (4, 6)
    }
    goal = (grid_size - 2, grid_size - 2)

    for idx, (x, y) in enumerate(path):
        img = Image.new("RGB", (grid_size * 80, grid_size * 80), "white")
        draw = ImageDraw.Draw(img)
        for j in range(grid_size):
            for i in range(grid_size):
                px = j * 80
                py = i * 80
                color = (255, 255, 255)
                if (j, i) in walls:
                    color = (75, 85, 99)
                elif (j, i) == goal:
                    color = (239, 68, 68)
                draw.rectangle([px, py, px + 80, py + 80], fill=color, outline=(200, 200, 200), width=2)

        draw.rectangle([x * 80 + 10, y * 80 + 10, x * 80 + 70, y * 80 + 70], fill=(245, 158, 11), outline=(120, 113, 108))
        if idx == len(path) - 1:
            draw.rectangle([goal[0] * 80 + 10, goal[1] * 80 + 10, goal[0] * 80 + 70, goal[1] * 80 + 70], fill=(239, 68, 68), outline=(127, 29, 29))
        frames.append(img)

    if frames:
        frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=250)


def save_training_csv(history: List[float], filename: Path):
    # Export episode-by-episode training data for further analysis or reporting.
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "reward"])
        for idx, reward in enumerate(history, start=1):
            writer.writerow([idx, reward])


def run_pipeline(train_episodes: int = 500, render_demo: bool = True):
    # Full training pipeline: train model, save artifacts, and produce visual outputs.
    RESULTS_DIR.mkdir(exist_ok=True)
    env = WarehouseEnv(grid_size=8, seed=7)
    model, history = train_agent(env, episodes=train_episodes)

    model_path = ROOT / MODEL_FILE.name
    torch.save(model.state_dict(), model_path)

    reward_plot = RESULTS_DIR / "training_reward_curve.png"
    save_reward_plot(history, reward_plot)
    save_training_csv(history, RESULTS_DIR / "training_history.csv")

    total_reward, path = run_policy(WarehouseEnv(grid_size=8, seed=42), model)
    demo_plot = RESULTS_DIR / "demo_policy_path.png"
    save_demo_grid(path, 8, demo_plot)
    demo_gif = RESULTS_DIR / "demo_policy_path.gif"
    create_demo_gif(path, 8, demo_gif)

    print(f"Training complete. Model saved to {model_path}")
    print(f"Reward plot saved to {reward_plot}")
    print(f"Demo path saved to {demo_plot}")
    print(f"Demo GIF saved to {demo_gif}")
    print(f"Final reward: {total_reward:.2f}")
    return {"model": model, "history": history, "path": path}


def parse_args():
    # Parse command-line arguments for training configuration.
    parser = argparse.ArgumentParser(description="Train a warehouse RL project.")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes for PPO.")
    parser.add_argument("--demo-seed", type=int, default=42, help="Random seed used for policy demonstration.")
    return parser.parse_args()


if __name__ == "__main__":
    # Entry point for running the project directly from the command line.
    args = parse_args()
    run_pipeline(train_episodes=args.episodes)
