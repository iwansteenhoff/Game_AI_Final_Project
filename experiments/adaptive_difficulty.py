"""Adaptive difficulty experiment for scripted Pacman players."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import ghost_mcts
from experiments.difficulty_and_ghosts import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MCTS_DEPTH,
    DEFAULT_MCTS_ROLLOUTS,
    run_episode,
)
from game.difficulty import (
    TARGET_HIGH,
    TARGET_LOW,
    PlayerProfile,
    RunMetrics,
    get_config,
)


DEFAULT_EPISODES = 20
DEFAULT_REPETITIONS = 5
DEFAULT_START_DIFFICULTY = 2
DEFAULT_CANDIDATES = 2
PLAYER_TYPES = {
    "weak": "random",
    "medium": "greedy",
    "strong": "cautious",
}


@dataclass(frozen=True)
class AdaptiveEpisodeResult:
    player_type: str
    pacman_agent: str
    repetition: int
    episode: int
    difficulty_before: int
    difficulty_after: int
    performance_score: float
    recent_performance: float
    won: bool
    pellet_completion: float
    steps_survived: int
    total_pellets: int
    pellets_collected: int


def run_adaptive_experiment(
    episodes: int,
    seed: int | None,
    start_difficulty: int,
    candidates: int,
    max_steps: int,
    repetitions: int,
) -> list[AdaptiveEpisodeResult]:
    if seed is not None:
        random.seed(seed)

    rows = []
    for repetition in range(1, repetitions + 1):
        for player_type, pacman_agent in PLAYER_TYPES.items():
            profile = PlayerProfile(history_size=5)
            difficulty = start_difficulty

            for episode in range(1, episodes + 1):
                config = get_config(difficulty)
                result = run_episode(
                    config=config,
                    target_difficulty=difficulty,
                    pacman_agent=pacman_agent,
                    candidates=candidates,
                    max_steps=max_steps,
                )
                profile.add_run(
                    RunMetrics(
                        difficulty_level=difficulty,
                        won=result.won,
                        steps_survived=result.steps,
                        max_steps=max_steps,
                        pellets_collected=result.pellets_collected,
                        total_pellets=result.total_pellets,
                        power_pellets_collected=result.power_pellets_collected,
                        ghosts_eaten=result.ghosts_eaten,
                        deaths_to_ghost=result.deaths_to_ghost,
                        maze_difficulty_score=result.maze_difficulty_score,
                        ghost_agent_type=config.ghost_agent,
                        ghost_count=config.ghost_count,
                        final_score=result.score,
                    )
                )
                next_difficulty = profile.recommended_difficulty(difficulty)
                rows.append(
                    AdaptiveEpisodeResult(
                        player_type=player_type,
                        pacman_agent=pacman_agent,
                        repetition=repetition,
                        episode=episode,
                        difficulty_before=difficulty,
                        difficulty_after=next_difficulty,
                        performance_score=result.score,
                        recent_performance=profile.recent_performance or result.score,
                        won=result.won,
                        pellet_completion=result.pellet_completion,
                        steps_survived=result.steps,
                        total_pellets=result.total_pellets,
                        pellets_collected=result.pellets_collected,
                    )
                )
                difficulty = next_difficulty

    return rows


def write_csv(path: Path, rows: list[AdaptiveEpisodeResult]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AdaptiveEpisodeResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def print_results(rows: list[AdaptiveEpisodeResult]) -> None:
    print("player rep episode diff_before diff_after score recent won pellet% steps")
    print("---------------------------------------------------------------------")
    for row in rows:
        print(
            f"{row.player_type:>6} {row.repetition:>3} {row.episode:>7} {row.difficulty_before:>11} "
            f"{row.difficulty_after:>10} {row.performance_score:>5.2f} "
            f"{row.recent_performance:>6.2f} {str(row.won):>5} "
            f"{row.pellet_completion:>7.0%} {row.steps_survived:>5}"
        )


def write_visualization(rows: list[AdaptiveEpisodeResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle("Adaptive Difficulty Experiment", fontsize=20, fontweight="bold")
    figure.patch.set_facecolor("#f8fafc")

    colors = {"weak": "#c92a2a", "medium": "#1971c2", "strong": "#2f9e44"}

    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.grid(True, color="#e5e7eb", linewidth=0.9)
        axis.set_xlabel("Episode")
        for spine in axis.spines.values():
            spine.set_color("#d1d5db")

    for player_type in PLAYER_TYPES:
        player_rows = [row for row in rows if row.player_type == player_type]
        episodes = sorted({row.episode for row in player_rows})
        color = colors[player_type]

        plot_metric_with_std(axes[0, 0], episodes, player_rows, lambda row: row.difficulty_before, color, player_type, clamp_high=5.0)
        plot_metric_with_std(axes[0, 1], episodes, player_rows, lambda row: row.performance_score, color, player_type)
        plot_metric_with_std(axes[1, 0], episodes, player_rows, lambda row: 1.0 if row.won else 0.0, color, player_type)
        plot_metric_with_std(axes[1, 1], episodes, player_rows, lambda row: row.pellet_completion, color, player_type)

    axes[0, 0].set_title("Difficulty Over Time", fontweight="bold")
    axes[0, 0].set_ylabel("Difficulty level")
    axes[0, 0].set_ylim(-0.2, 5.2)
    axes[0, 0].legend()

    axes[0, 1].set_title("Performance Score Over Time", fontweight="bold")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].axhspan(TARGET_LOW, TARGET_HIGH, color="#2f9e44", alpha=0.12, label="target zone")
    axes[0, 1].legend()

    axes[1, 0].set_title("Win Rate Over Time", fontweight="bold")
    axes[1, 0].set_ylabel("Win rate")
    axes[1, 0].set_yticks([0, 1], ["0%", "100%"])
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].legend()

    axes[1, 1].set_title("Pellet Completion Over Time", fontweight="bold")
    axes[1, 1].set_ylabel("Pellet completion")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1, 1].legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_metric_with_std(axis, episodes, rows, value_fn, color: str, label: str, clamp_high: float = 1.0) -> None:
    means = []
    stds = []
    for episode in episodes:
        values = [float(value_fn(row)) for row in rows if row.episode == episode]
        mean = average(values)
        means.append(mean)
        stds.append(stddev(values))

    lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
    upper = [min(clamp_high, mean + std) for mean, std in zip(means, stds)]
    axis.plot(episodes, means, marker="o", linewidth=2.3, color=color, label=label)
    axis.fill_between(episodes, lower, upper, color=color, alpha=0.16, linewidth=0)


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values) -> float:
    values = [float(value) for value in values]
    if len(values) < 2:
        return 0.0
    mean = average(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adaptive difficulty experiment.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Consecutive games per scripted player.")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="Repeated adaptive runs per scripted player.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use --seed -1 for no fixed seed.")
    parser.add_argument("--start-difficulty", type=int, default=DEFAULT_START_DIFFICULTY, help="Initial difficulty level for each player type.")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES, help="Candidate mazes considered per generated level.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum simulated steps per episode.")
    parser.add_argument("--mcts-rollouts", type=int, default=DEFAULT_MCTS_ROLLOUTS, help="Rollouts per MCTS action during experiments.")
    parser.add_argument("--mcts-depth", type=int, default=DEFAULT_MCTS_DEPTH, help="Rollout depth for MCTS during experiments.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--visualization", type=Path, help="Optional Matplotlib visualization output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = None if args.seed < 0 else args.seed
    ghost_mcts.ROLLOUTS_PER_ACTION = args.mcts_rollouts
    ghost_mcts.ROLLOUT_DEPTH = args.mcts_depth

    rows = run_adaptive_experiment(
        episodes=args.episodes,
        seed=seed,
        start_difficulty=args.start_difficulty,
        candidates=args.candidates,
        max_steps=args.max_steps,
        repetitions=args.repetitions,
    )
    print_results(rows)

    if args.output is not None:
        write_csv(args.output, rows)
        print(f"\nSaved adaptive difficulty CSV to {args.output}")
    if args.visualization is not None:
        write_visualization(rows, args.visualization)
        print(f"Saved visualization to {args.visualization}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
