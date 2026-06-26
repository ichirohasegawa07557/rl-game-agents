from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RL Game Agents Lab", layout="wide")
st.title("RL Game Agents Lab")

results = Path("results")

st.write("Small reinforcement learning experiments: Q-learning, DQN, REINFORCE, and Mini-Atari Catch.")

for name in [
    "gridworld_q_learning_curve.png",
    "cartpole_dqn_curve.png",
    "cartpole_reinforce_curve.png",
    "mini_atari_dqn_curve.png",
    "gridworld_policy.png",
]:
    path = results / name
    if path.exists():
        st.subheader(name)
        st.image(str(path))

gif = results / "mini_atari_rollout.gif"
if gif.exists():
    st.subheader("Mini-Atari Rollout")
    st.image(str(gif))

for csv_name in [
    "gridworld_q_learning_history.csv",
    "cartpole_dqn_history.csv",
    "cartpole_reinforce_history.csv",
    "mini_atari_dqn_history.csv",
]:
    path = results / csv_name
    if path.exists():
        st.subheader(csv_name)
        st.dataframe(pd.read_csv(path).tail(20))
