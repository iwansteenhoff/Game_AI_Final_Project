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

## Project Structure

```text
main.py
game/
  pacman_env.py
  maze_generator.py
  difficulty.py
agents/
  ghost_random.py
  ghost_heuristic.py
```

The "game maker" is mainly implemented in `game/maze_generator.py` and `game/difficulty.py`.
