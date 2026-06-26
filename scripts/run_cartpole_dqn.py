import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  

import argparse
from pathlib import Path
import torch

from src.environments import make_cartpole_env
from src.dqn import DQNConfig, train_cartpole_dqn
from src.visualize import plot_training_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    env = make_cartpole_env()
    model, df = train_cartpole_dqn(env, DQNConfig(episodes=args.episodes))
    df.to_csv(results / "cartpole_dqn_history.csv", index=False)
    torch.save(model.state_dict(), results / "cartpole_dqn_model.pt")
    plot_training_curve(df, results / "cartpole_dqn_curve.png", "CartPole DQN")
    print(df.tail())


if __name__ == "__main__":
    main()
