from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from src.metrics import moving_average


def plot_training_curve(df, output_path: str, title: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["reward"], alpha=0.4, label="reward")
    plt.plot(df["episode"], moving_average(df["reward"], 10), label="moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_gridworld_policy(env, q, output_path: str):
    arrows = ["↑", "↓", "←", "→"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks(range(env.size + 1))
    ax.set_yticks(range(env.size + 1))
    ax.grid(True)

    for r in range(env.size):
        for c in range(env.size):
            pos = (r, c)
            x = c + 0.5
            y = env.size - r - 0.5
            if pos in env.walls:
                ax.text(x, y, "■", ha="center", va="center", fontsize=18)
            elif pos == env.goal:
                ax.text(x, y, "G", ha="center", va="center", fontsize=18)
            else:
                state = env.state_id(pos)
                ax.text(x, y, arrows[int(np.argmax(q[state]))], ha="center", va="center", fontsize=18)

    ax.set_title("Learned GridWorld Policy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_mini_atari_rollout_gif(env, q_model, output_path: str, max_steps: int = 30):
    import torch

    frames = []
    obs = env.reset()
    for _ in range(max_steps):
        frames.append(env.render_rgb())
        with torch.no_grad():
            q_values = q_model(torch.tensor(obs).float().unsqueeze(0))
            action = int(torch.argmax(q_values).item())
        obs, _, done, _ = env.step(action)
        if done:
            frames.append(env.render_rgb())
            break

    imageio.mimsave(output_path, frames, duration=0.15)
