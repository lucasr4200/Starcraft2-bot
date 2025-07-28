# Self-Learning StarCraft II Bot

A simple Terran vs. Terran bot for the game StarCraft II that learns basic build-and-attack strategies using Q-learning with a Q-table. The **SmartAgent** improves over time against a random-action baseline.

## Features

- **QLearningTable** implementation with ε-greedy action selection  
- **SmartAgent** that observes game state (unit counts, resources, build queues, enemy units)  
- **RandomAgent** opponent for baseline comparison  
- Periodic saving/loading of the Q-table (`q_table.pkl`)  

## Prerequisites

- Python 3.7+  
- StarCraft II installed with map **Simple64**  
- SC2 API & [pysc2](https://github.com/deepmind/pysc2)  
- `numpy`, `pandas`, `absl-py`

## Installation

```bash
git clone <(https://github.com/lucasr4200/Starcraft2-bot>
cd <Starcraft2-bot>
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy pandas pysc2 absl-py
```
