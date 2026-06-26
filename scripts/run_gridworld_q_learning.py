import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) 


import argparse
from pathlib import Path

from src.environments import GridWorld
from src.q_learning import QLearningConfig, train_q_learning
from src.visualize import plot_training_curve, plot_gridworld_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    env = GridWorld()
    q, df = train_q_learning(env, QLearningConfig(episodes=args.episodes))
    df.to_csv(results / "gridworld_q_learning_history.csv", index=False)
    plot_training_curve(df, results / "gridworld_q_learning_curve.png", "GridWorld Q-learning")
    plot_gridworld_policy(env, q, results / "gridworld_policy.png")
    print(df.tail())


if __name__ == "__main__":
    main()
