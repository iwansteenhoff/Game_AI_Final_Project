# Adaptive Pacman-Like Game AI Prototype

This repository contains a lightweight Pacman-inspired environment for the Game AI final assignment. The focus is on procedural content generation, ghost agents, and adaptive difficulty rather than recreating the full original Pacman game.

## Features

- Grid-based Pacman-like game implemented in Python
- Procedural maze generation with BFS validation
- Random and heuristic ghost agents
- Adaptive difficulty between playthroughs
- Pygame visualization

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Controls:

- Arrow keys or WASD: move Pacman
- R: restart with a newly generated maze
- Escape or close window: quit

After each win or loss, the program computes a performance score and adjusts the difficulty for the next generated maze.

## Experiments

### Maze Feasibility

Generate mazes for every difficulty level and report structural playability metrics:

```bash
python3 experiments/maze_feasibility.py --samples 100 --seed 42
```

To save the same results as CSV:

```bash
python3 experiments/maze_feasibility.py --samples 100 --seed 42 --output results/maze_feasibility.csv
```

To also generate a Matplotlib visualization:

```bash
python3 experiments/maze_feasibility.py --samples 100 --seed 42 --output results/maze_feasibility.csv --visualization results/maze_feasibility.png
```

The experiment reports generation/validity rates, connectivity, pellet reachability, dead-end ratio, junction count, branching factor, corridor length, nearest ghost distance, estimated collection distance, and estimated maze difficulty score.

### Difficulty Calibration and Ghost Comparison

Run automated Pacman baselines across all difficulty levels, then compare ghost agents on a fixed difficulty:

```bash
python3 experiments/difficulty_and_ghosts.py --episodes 10 --seed 42 --difficulty-output results/difficulty_calibration.csv --ghost-output results/ghost_comparison.csv --visualization results/difficulty_and_ghosts.png
```

For a stronger final-report run, increase `--episodes`, for example:

```bash
python3 experiments/difficulty_and_ghosts.py --episodes 30 --seed 42 --difficulty-output results/difficulty_calibration.csv --ghost-output results/ghost_comparison.csv --visualization results/difficulty_and_ghosts.png
```

The difficulty calibration runs random, greedy, and cautious Pacman baselines and reports win rate, pellet completion, survival time, performance score, and ghosts eaten. The ghost comparison keeps maze difficulty fixed and compares random, heuristic, and MCTS ghosts using win rate, pellet completion, steps until the episode ends, and average ghost distance.

### Adaptive Difficulty

Run weak, medium, and strong scripted players through consecutive games with adaptive difficulty enabled:

```bash
python3 experiments/adaptive_difficulty.py --episodes 20 --seed 42 --output results/adaptive_difficulty.csv --visualization results/adaptive_difficulty.png
```

The experiment reports and visualizes difficulty over time, performance score over time, win/loss sequence, and pellet completion over time. The weak player uses random movement, the medium player greedily collects pellets, and the strong player greedily collects pellets while avoiding nearby ghosts.

## Project Structure

```text
main.py
experiments/
  maze_feasibility.py
  difficulty_and_ghosts.py
  adaptive_difficulty.py
game/
  pacman_env.py
  maze_generator.py
  difficulty.py
agents/
  ghost_random.py
  ghost_heuristic.py
```

The "game maker" is mainly implemented in `game/maze_generator.py` and `game/difficulty.py`.
