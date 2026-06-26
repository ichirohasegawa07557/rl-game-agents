from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.replay_buffer import ReplayBuffer
from src.networks import MLPQNetwork, ConvQNetwork


@dataclass
class DQNConfig:
    episodes: int = 80
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 10000
    min_buffer_size: int = 300
    target_update: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    max_steps: int = 300
    seed: int = 0


def _obs_to_numpy(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32)


def train_cartpole_dqn(env, config: DQNConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    obs, _ = env.reset(seed=config.seed)
    obs_dim = int(np.asarray(obs).shape[0])
    action_dim = int(env.action_space.n)

    q = MLPQNetwork(obs_dim, action_dim)
    target = MLPQNetwork(obs_dim, action_dim)
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size)

    epsilon = config.epsilon_start
    rows = []

    for episode in range(config.episodes):
        obs, _ = env.reset(seed=config.seed + episode)
        episode_reward = 0.0
        losses = []

        for step in range(config.max_steps):
            state = _obs_to_numpy(obs)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q(torch.tensor(state).float().unsqueeze(0))
                    action = int(torch.argmax(q_values).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, _obs_to_numpy(next_obs), done)
            episode_reward += reward
            obs = next_obs

            if len(buffer) >= config.min_buffer_size:
                s, a, r, ns, d = buffer.sample(config.batch_size)
                s = torch.tensor(s).float()
                a = torch.tensor(a).long()
                r = torch.tensor(r).float()
                ns = torch.tensor(ns).float()
                d = torch.tensor(d).float()

                q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = r + config.gamma * (1.0 - d) * target(ns).max(dim=1).values

                loss = nn.functional.mse_loss(q_sa, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        if episode % config.target_update == 0:
            target.load_state_dict(q.state_dict())

        rows.append({
            "episode": episode,
            "reward": episode_reward,
            "epsilon": epsilon,
            "loss": float(np.mean(losses)) if losses else np.nan,
        })

    return q, pd.DataFrame(rows)


def train_pixel_dqn(env, config: DQNConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    action_dim = env.action_space_n
    q = ConvQNetwork(env.observation_shape, action_dim)
    target = ConvQNetwork(env.observation_shape, action_dim)
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size)

    epsilon = config.epsilon_start
    rows = []

    for episode in range(config.episodes):
        obs = env.reset()
        episode_reward = 0.0
        losses = []

        for step in range(config.max_steps):
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_values = q(torch.tensor(obs).float().unsqueeze(0))
                    action = int(torch.argmax(q_values).item())

            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs = next_obs

            if len(buffer) >= config.min_buffer_size:
                s, a, r, ns, d = buffer.sample(config.batch_size)
                s = torch.tensor(s).float()
                a = torch.tensor(a).long()
                r = torch.tensor(r).float()
                ns = torch.tensor(ns).float()
                d = torch.tensor(d).float()

                q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = r + config.gamma * (1.0 - d) * target(ns).max(dim=1).values

                loss = nn.functional.mse_loss(q_sa, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        if episode % config.target_update == 0:
            target.load_state_dict(q.state_dict())

        rows.append({
            "episode": episode,
            "reward": episode_reward,
            "epsilon": epsilon,
            "loss": float(np.mean(losses)) if losses else np.nan,
        })

    return q, pd.DataFrame(rows)
