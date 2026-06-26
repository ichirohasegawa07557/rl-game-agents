import subprocess
import sys


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    py = sys.executable
    run([py, "scripts/run_gridworld_q_learning.py", "--episodes", "120"])
    run([py, "scripts/run_cartpole_dqn.py", "--episodes", "20"])
    run([py, "scripts/run_cartpole_policy_gradient.py", "--episodes", "20"])
    run([py, "scripts/run_mini_atari_dqn.py", "--episodes", "40"])
    print("Done. Results saved in results/.")


if __name__ == "__main__":
    main()
