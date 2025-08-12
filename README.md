![Environment Overview](assets/rendered_environment.png)


# Deadlock Reference Models for Multi-Agent Pathfinding with RLlib

This repository collects reference implementations for training and evaluating reinforcement learning agents on multi-agent pathfinding problems. The environments explicitly support deadlocks so that agents must cooperate to resolve them. All algorithms are built with RLlib to encourage reproducibility and extensibility.
The code base is part of ongoing doctoral research; version `v1.0.0` corresponds to the snapshot that will be referenced in the author's PhD thesis.

## Features

- **Multi-Agent Environments**: Grid worlds with obstacles, deterministic or random start and goal positions. Implemented in [`src/environments`](src/environments).
- **RLlib Agents**: Configuration helpers for PPO, DQN and IMPALA located in [`src/agents`](src/agents).
- **Custom Models**: Action masking models in [`models`](models) integrate with RLlib.
- **Training Script**: [`main.py`](main.py) can train or test agents depending on the selected mode.
- **Classical Planners**: Baseline implementations such as A* and CBS in [`scripts`](scripts).
- **Experiment Results**: Logs and heatmaps are stored under [`experiments`](experiments).

## Overview of available Reference Models

![Reference Model Overview](assets/overview_reference_models.png)


## Getting Started

Install the project dependencies (Python 3.11 is recommended) and run the training script:

```bash
python -m pip install -r requirements.txt
python main.py
```

Run tests with:

```bash
pytest
```

## Project Structure

```
├── src/               # Environment and trainer code
├── models/            # Custom RLlib models
├── scripts/           # Baseline planning algorithms and utilities
├── experiments/       # Generated results
└── tests/             # Unit tests
```

## Versioning

The `v1.0.0` tag represents the reference version that will accompany the author's PhD thesis. A URL for citation will be added once the thesis is published.

## License

This project is released under the MIT License.
