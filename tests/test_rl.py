import numpy as np
import torch

from src.environments import GridWorld, MiniAtariCatchEnv
from src.replay_buffer import ReplayBuffer
from src.networks import MLPQNetwork
from src.q_learning import QLearningConfig, train_q_learning


def test_gridworld_step():
    env = GridWorld()
    env.reset()
    ns, r, done, _ = env.step(3)
    assert isinstance(ns, int)
    assert isinstance(r, float)
    assert isinstance(done, bool)


def test_mini_atari_observation_shape():
    env = MiniAtariCatchEnv()
    obs = env.reset()
    assert obs.shape == env.observation_shape


def test_replay_buffer_sample():
    b = ReplayBuffer(capacity=10)
    for i in range(5):
        b.push(np.array([i]), 0, 1.0, np.array([i + 1]), False)
    s, a, r, ns, d = b.sample(3)
    assert s.shape[0] == 3


def test_network_forward():
    net = MLPQNetwork(obs_dim=4, action_dim=2)
    out = net(torch.zeros(1, 4))
    assert out.shape == (1, 2)


def test_q_learning_runs():
    env = GridWorld()
    q, df = train_q_learning(env, QLearningConfig(episodes=5))
    assert q.shape == (env.n_states, env.action_space_n)
    assert len(df) == 5
