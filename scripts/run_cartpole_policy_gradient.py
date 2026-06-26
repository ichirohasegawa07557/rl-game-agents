import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  

import argparse
from pathlib import Path
import torch

from src.environments import make_cartpole_env
from src.policy_gradient import PGConfig, train_reinforce_cartpole
from src.visualize import plot_training_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    env = make_cartpole_env()
    model, df = train_reinforce_cartpole(env, PGConfig(episodes=args.episodes))
    df.to_csv(results / "cartpole_reinforce_history.csv", index=False)
    torch.save(model.state_dict(), results / "cartpole_reinforce_model.pt")
    plot_training_curve(df, results / "cartpole_reinforce_curve.png", "CartPole REINFORCE")
    print(df.tail())


if __name__ == "__main__":
    main()
