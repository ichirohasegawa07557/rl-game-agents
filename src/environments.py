from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class GridWorld:
    """Small deterministic GridWorld for tabular Q-learning."""

    def __init__(self, size: int = 5, max_steps: int = 60, seed: int = 0):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.walls = {(1, 1), (2, 1), (3, 3)}
        self.action_space_n = 4
        self.reset()

    @property
    def n_states(self) -> int:
        return self.size * self.size

    def state_id(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def reset(self) -> int:
        self.pos = self.start
        self.steps = 0
        return self.state_id(self.pos)

    def step(self, action: int):
        self.steps += 1
        r, c = self.pos
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        elif action == 3:
            c += 1

        new_pos = (int(np.clip(r, 0, self.size - 1)), int(np.clip(c, 0, self.size - 1)))
        if new_pos in self.walls:
            new_pos = self.pos

        self.pos = new_pos
        done = self.pos == self.goal or self.steps >= self.max_steps
        reward = 1.0 if self.pos == self.goal else -0.02
        return self.state_id(self.pos), reward, done, {}


@dataclass
class MiniAtariCatchConfig:
    width: int = 10
    height: int = 10
    max_steps: int = 80
    seed: int = 0


class MiniAtariCatchEnv:
    """
    Lightweight Atari-like Catch environment.

    Observation is a 1 x H x W image:
    - falling object = 1.0
    - paddle = 0.5
    """

    def __init__(self, config: MiniAtariCatchConfig | None = None):
        self.config = config or MiniAtariCatchConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.action_space_n = 3
        self.reset()

    @property
    def observation_shape(self):
        return (1, self.config.height, self.config.width)

    def reset(self):
        self.steps = 0
        self.paddle_x = self.config.width // 2
        self.object_x = int(self.rng.integers(0, self.config.width))
        self.object_y = 0
        return self._obs()

    def _obs(self):
        grid = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        grid[self.object_y, self.object_x] = 1.0
        grid[self.config.height - 1, self.paddle_x] = 0.5
        return grid[None, :, :]

    def step(self, action: int):
        self.steps += 1
        if action == 0:
            self.paddle_x -= 1
        elif action == 2:
            self.paddle_x += 1
        self.paddle_x = int(np.clip(self.paddle_x, 0, self.config.width - 1))

        self.object_y += 1
        reward = 0.0
        done = False

        if self.object_y >= self.config.height - 1:
            caught = self.object_x == self.paddle_x
            reward = 1.0 if caught else -1.0
            done = True

        if self.steps >= self.config.max_steps:
            done = True

        return self._obs(), reward, done, {}

    def render_rgb(self):
        obs = self._obs()[0]
        rgb = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        rgb[obs == 1.0] = [255, 255, 255]
        rgb[obs == 0.5] = [80, 180, 255]
        return np.kron(rgb, np.ones((20, 20, 1), dtype=np.uint8))


def make_cartpole_env(seed: int = 0):
    """Create Gymnasium CartPole if available."""
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        return env
    except Exception as exc:
        raise RuntimeError(
            "CartPole requires gymnasium[classic-control]. "
            "Install with: pip install 'gymnasium[classic-control]'"
        ) from exc
