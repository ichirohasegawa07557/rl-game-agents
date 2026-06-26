from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class QLearningConfig:
    episodes: int = 300
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    seed: int = 0


def train_q_learning(env, config: QLearningConfig):
    rng = np.random.default_rng(config.seed)
    q = np.zeros((env.n_states, env.action_space_n), dtype=np.float32)
    epsilon = config.epsilon_start
    rows = []

    for episode in range(config.episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.action_space_n))
            else:
                action = int(np.argmax(q[state]))

            next_state, reward, done, _ = env.step(action)
            td_target = reward + config.gamma * np.max(q[next_state]) * (not done)
            q[state, action] += config.alpha * (td_target - q[state, action])
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        rows.append({
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "epsilon": epsilon,
        })

    return q, pd.DataFrame(rows)
