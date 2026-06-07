# DQN Agent — LunarLander-v3

A Deep Q-Network (DQN) agent trained to solve OpenAI Gymnasium's LunarLander-v3 environment.

## Project Structure
- `random_agent.py` — baseline random agent
- `dqn_agent.py` — full DQN implementation with replay buffer and target network

## How It Works
The agent uses a neural network to approximate Q-values and learns via:
- **Experience replay** — samples random past transitions to break correlation
- **Target network** — a frozen copy of the online network for stable training
- **Epsilon-greedy exploration** — decays from 1.0 → 0.01 over 10,000 steps

## Setup
pip install gymnasium torch numpy
python dqn_agent.py
