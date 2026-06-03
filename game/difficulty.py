"""Adaptive difficulty configuration for generated Pacman-like games."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class DifficultyConfig:
    ghost_agent: str
    ghost_count: int
    power_pellets: int
    maze_openness: float
    min_start_distance: int
    ghost_speed: int = 1
    ghost_aggression: float = 1.0


@dataclass(frozen=True)
class RunMetrics:
    difficulty_level: int
    won: bool
    steps_survived: int
    max_steps: int
    pellets_collected: int
    total_pellets: int
    power_pellets_collected: int
    ghosts_eaten: int
    deaths_to_ghost: int
    maze_difficulty_score: float
    ghost_agent_type: str
    ghost_count: int
    final_score: float


DIFFICULTY_LEVELS: dict[int, DifficultyConfig] = {
    0: DifficultyConfig("random", 1, 9, 0.66, 24, ghost_aggression=0.0),
    1: DifficultyConfig("random", 1, 8, 0.63, 22, ghost_aggression=0.0),
    2: DifficultyConfig("heuristic", 1, 7, 0.60, 20, ghost_aggression=0.08),
    3: DifficultyConfig("heuristic", 2, 6, 0.57, 18, ghost_aggression=0.18),
    4: DifficultyConfig("heuristic", 2, 5, 0.54, 16, ghost_aggression=0.32),
    5: DifficultyConfig("mcts", 2, 4, 0.50, 14, ghost_speed=1, ghost_aggression=0.45),
}


TARGET_LOW = 0.45
TARGET_HIGH = 0.75


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
    power_pellets_collected: int = 0,
    ghosts_eaten: int = 0,
) -> float:
    pellet_completion = pellets_collected / max(1, total_pellets)
    survival_score = min(1.0, steps_survived / max(1, max_steps))
    win_bonus = 1.0 if won else 0.0
    efficiency = min(1.0, total_pellets / max(1, steps_survived)) if won else 0.0
    ghost_interaction = min(1.0, (power_pellets_collected * 0.25) + (ghosts_eaten * 0.5))
    solution_score = min(1.0, generated_greedy_solution / max(1, traveled_steps))
    return (
        (0.55 * pellet_completion)
        + (0.25 * win_bonus * solution_score)
        + (0.10 * survival_score)
        + (0.10 * max(efficiency, ghost_interaction))
    )


def update_difficulty(level: int, performance: float) -> int:
    if performance > TARGET_HIGH:
        level += 1
    elif performance < TARGET_LOW:
        level -= 1
    return clamp_difficulty(level)


class PlayerProfile:
    def __init__(self, history_size: int = 5) -> None:
        self.runs: Deque[RunMetrics] = deque(maxlen=history_size)

    def add_run(self, metrics: RunMetrics) -> None:
        self.runs.append(metrics)

    @property
    def recent_performance(self) -> float | None:
        if not self.runs:
            return None
        weights = range(1, len(self.runs) + 1)
        weighted_total = sum(run.final_score * weight for run, weight in zip(self.runs, weights))
        return weighted_total / sum(weights)

    @property
    def latest_performance(self) -> float | None:
        if not self.runs:
            return None
        return self.runs[-1].final_score

    @property
    def recent_win_rate(self) -> float | None:
        if not self.runs:
            return None
        return sum(run.won for run in self.runs) / len(self.runs)

    @property
    def recent_pellet_completion(self) -> float | None:
        if not self.runs:
            return None
        return sum(
            run.pellets_collected / max(1, run.total_pellets)
            for run in self.runs
        ) / len(self.runs)

    def recommended_difficulty(self, current_level: int) -> int:
        recent = self.recent_performance
        if recent is None:
            return clamp_difficulty(current_level)
        latest = self.latest_performance
        if latest is not None and latest > TARGET_HIGH and recent > TARGET_LOW:
            return clamp_difficulty(current_level + 1)
        return update_difficulty(current_level, recent)
