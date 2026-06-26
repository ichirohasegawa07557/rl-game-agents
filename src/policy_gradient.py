from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

from src.networks import PolicyNetwork


@dataclass
class PGConfig:
    episodes: int = 80
    gamma: float = 0.99
    lr: float = 1e-2
    max_steps: int = 300
    seed: int = 0


def discounted_returns(rewards, gamma: float):
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out = list(reversed(out))
    arr = torch.tensor(out).float()
    if len(arr) > 1:
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr


def train_reinforce_cartpole(env, config: PGConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    obs, _ = env.reset(seed=config.seed)
    obs_dim = int(np.asarray(obs).shape[0])
    action_dim = int(env.action_space.n)

    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    rows = []

    for episode in range(config.episodes):
        obs, _ = env.reset(seed=config.seed + episode)
        log_probs = []
        rewards = []
        total_reward = 0.0

        for step in range(config.max_steps):
            x = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            probs = policy(x).squeeze(0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rewards.append(float(reward))
            total_reward += reward

            if done:
                break

        returns = discounted_returns(rewards, config.gamma)
        loss = -torch.stack(log_probs).mul(returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rows.append({
            "episode": episode,
            "reward": total_reward,
            "loss": float(loss.item()),
        })

    return policy, pd.DataFrame(rows)
