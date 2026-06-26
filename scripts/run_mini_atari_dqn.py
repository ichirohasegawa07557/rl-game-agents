import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch

from src.environments import MiniAtariCatchEnv, MiniAtariCatchConfig
from src.dqn import DQNConfig, train_pixel_dqn
from src.visualize import plot_training_curve, make_mini_atari_rollout_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150)
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    env = MiniAtariCatchEnv(MiniAtariCatchConfig())
    model, df = train_pixel_dqn(
        env,
        DQNConfig(
            episodes=args.episodes,
            batch_size=32,
            min_buffer_size=80,
            max_steps=20,
            target_update=8,
        ),
    )
    df.to_csv(results / "mini_atari_dqn_history.csv", index=False)
    torch.save(model.state_dict(), results / "mini_atari_dqn_model.pt")
    plot_training_curve(df, results / "mini_atari_dqn_curve.png", "Mini-Atari Catch DQN")
    make_mini_atari_rollout_gif(env, model, results / "mini_atari_rollout.gif")
    print(df.tail())


if __name__ == "__main__":
    main()
