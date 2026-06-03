"""Adaptive difficulty configuration for generated Pacman-like games."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyConfig:
    ghost_agent: str
    ghost_count: int
    power_pellets: int
    maze_openness: float
    min_start_distance: int
    ghost_speed: int = 1
    spike_count: int = 0
    frighten_duration: int = 20  # steps ghosts stay frightened


DIFFICULTY_LEVELS: dict[int, DifficultyConfig] = {
    0: DifficultyConfig("random",    1, 6, 0.58, 16, spike_count=0,  frighten_duration=30),
    1: DifficultyConfig("random",    2, 5, 0.54, 14, spike_count=2,  frighten_duration=25),
    2: DifficultyConfig("heuristic", 2, 4, 0.50, 12, spike_count=4,  frighten_duration=20),
    3: DifficultyConfig("heuristic", 3, 3, 0.46, 10, spike_count=6,  frighten_duration=15),
    4: DifficultyConfig("heuristic", 4, 2, 0.42,  8, spike_count=8,  frighten_duration=10),
    5: DifficultyConfig("mcts",      4, 1, 0.38,  6, ghost_speed=2, spike_count=10, frighten_duration=7),
}


def clamp_difficulty(level: int) -> int:
    return max(min(DIFFICULTY_LEVELS), min(max(DIFFICULTY_LEVELS), level))


def get_config(level: int) -> DifficultyConfig:
    return DIFFICULTY_LEVELS[clamp_difficulty(level)]


def performance_score(
    pellets_collected: int,
    total_pellets: int,
    steps_survived: int,
    max_steps: int,
    traveled_steps: int,
    generated_greedy_solution: int,
    won: bool,
    win_streak: int,
) -> float:
    pellet_completion = pellets_collected / max(1, total_pellets)
    survival_score = min(1.0, steps_survived / max(1, max_steps))
    solution_score = min(1.0, generated_greedy_solution / max(1, traveled_steps))
    win_factor = 1.0 if won else 0.0
    return (0.5 * pellet_completion) + (0.3 * survival_score) + (win_factor * (0.2 + solution_score)) + (0.1 * win_streak)


def update_difficulty(level: int, performance: float) -> int:
    if performance > 0.75:
        level += 1
    elif performance < 0.35:
        level -= 1
    return clamp_difficulty(level)
