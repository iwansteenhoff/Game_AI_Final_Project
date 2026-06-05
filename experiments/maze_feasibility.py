"""Maze feasibility experiment for the Pacman PCG system.

Generates multiple mazes per difficulty level and reports structural
playability metrics that can be used in the assignment write-up.
"""

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

from game.difficulty import DIFFICULTY_LEVELS, get_config
from game.maze_generator import analyze_maze, generate_maze, is_valid_maze


DEFAULT_WIDTH = 21
DEFAULT_HEIGHT = 21
DEFAULT_SAMPLES = 100


@dataclass
class LevelResult:
    level: int
    requested_samples: int
    generated_samples: int
    valid_samples: int
    connected_samples: int
    reachable_pellet_samples: int
    average_open_cells: float
    average_pellets: float
    average_dead_end_ratio: float
    average_junction_count: float
    average_branching_factor: float
    average_corridor_length: float
    average_nearest_ghost_distance: float
    average_collection_distance: float
    average_maze_difficulty_score: float
    std_dead_end_ratio: float
    std_branching_factor: float
    std_corridor_length: float
    std_nearest_ghost_distance: float
    std_collection_distance: float
    std_maze_difficulty_score: float

    @property
    def generation_rate(self) -> float:
        return self.generated_samples / max(1, self.requested_samples)

    @property
    def valid_rate(self) -> float:
        return self.valid_samples / max(1, self.requested_samples)


def run_experiment(
    samples: int,
    width: int,
    height: int,
    seed: int | None,
) -> list[LevelResult]:
    if seed is not None:
        random.seed(seed)

    results = []
    for level in sorted(DIFFICULTY_LEVELS):
        config = get_config(level)
        analyses = []
        valid_samples = 0
        connected_samples = 0
        reachable_pellet_samples = 0

        for _ in range(samples):
            try:
                maze = generate_maze(width, height, config)
            except RuntimeError:
                continue

            analysis = analyze_maze(maze.grid, maze.pacman_start, maze.ghost_starts, config)
            analyses.append(analysis)

            if is_valid_maze(
                maze.grid,
                maze.pacman_start,
                maze.ghost_starts,
                maze.ghost_respawn_location,
                config,
            ):
                valid_samples += 1
            if analysis.connected:
                connected_samples += 1
            if analysis.all_pellets_reachable:
                reachable_pellet_samples += 1

        results.append(
            LevelResult(
                level=level,
                requested_samples=samples,
                generated_samples=len(analyses),
                valid_samples=valid_samples,
                connected_samples=connected_samples,
                reachable_pellet_samples=reachable_pellet_samples,
                average_open_cells=average(a.open_cell_count for a in analyses),
                average_pellets=average(a.pellet_count for a in analyses),
                average_dead_end_ratio=average(a.dead_end_ratio for a in analyses),
                average_junction_count=average(a.junction_count for a in analyses),
                average_branching_factor=average(a.average_branching_factor for a in analyses),
                average_corridor_length=average(a.average_corridor_length for a in analyses),
                average_nearest_ghost_distance=average(a.nearest_ghost_distance for a in analyses),
                average_collection_distance=average(a.estimated_collection_distance for a in analyses),
                average_maze_difficulty_score=average(a.difficulty_score for a in analyses),
                std_dead_end_ratio=stddev(a.dead_end_ratio for a in analyses),
                std_branching_factor=stddev(a.average_branching_factor for a in analyses),
                std_corridor_length=stddev(a.average_corridor_length for a in analyses),
                std_nearest_ghost_distance=stddev(a.nearest_ghost_distance for a in analyses),
                std_collection_distance=stddev(a.estimated_collection_distance for a in analyses),
                std_maze_difficulty_score=stddev(a.difficulty_score for a in analyses),
            )
        )

    return results


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


def print_results(results: list[LevelResult]) -> None:
    header = (
        "level gen% valid% conn% pellets% open pellets dead_end junctions "
        "branch corridor ghost_dist collect_dist maze_score"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.level:>5} "
            f"{result.generation_rate:>4.0%} "
            f"{result.valid_rate:>6.0%} "
            f"{result.connected_samples / max(1, result.requested_samples):>5.0%} "
            f"{result.reachable_pellet_samples / max(1, result.requested_samples):>8.0%} "
            f"{result.average_open_cells:>4.0f} "
            f"{result.average_pellets:>7.0f} "
            f"{result.average_dead_end_ratio:>8.3f} "
            f"{result.average_junction_count:>9.1f} "
            f"{result.average_branching_factor:>6.2f} "
            f"{result.average_corridor_length:>8.2f} "
            f"{result.average_nearest_ghost_distance:>10.1f} "
            f"{result.average_collection_distance:>12.1f} "
            f"{result.average_maze_difficulty_score:>10.2f}"
        )


def write_csv(results: list[LevelResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(LevelResult.__dataclass_fields__) + ["generation_rate", "valid_rate"])
        writer.writeheader()
        for result in results:
            row = {
                field: getattr(result, field)
                for field in LevelResult.__dataclass_fields__
            }
            row["generation_rate"] = result.generation_rate
            row["valid_rate"] = result.valid_rate
            writer.writerow(row)


def write_visualization(results: list[LevelResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    levels = [result.level for result in results]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle("Maze Feasibility Experiment", fontsize=20, fontweight="bold")
    figure.patch.set_facecolor("#f8fafc")

    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.grid(True, color="#e5e7eb", linewidth=0.9)
        axis.set_xlabel("Difficulty level")
        axis.set_xticks(levels)
        for spine in axis.spines.values():
            spine.set_color("#d1d5db")

    plot_mean_with_std(
        axes[0, 0],
        levels,
        [result.average_maze_difficulty_score for result in results],
        [result.std_maze_difficulty_score for result in results],
        "#d9480f",
        "Maze score",
        clamp_high=5.0,
    )
    axes[0, 0].set_title("Estimated Maze Difficulty", fontweight="bold")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_ylim(0, 5)
    axes[0, 0].legend()

    plot_mean_with_std(
        axes[0, 1],
        levels,
        [result.valid_rate for result in results],
        [rate_std(result.valid_rate) for result in results],
        "#2f9e44",
        "Valid",
    )
    connected_rates = [result.connected_samples / max(1, result.requested_samples) for result in results]
    plot_mean_with_std(
        axes[0, 1],
        levels,
        connected_rates,
        [rate_std(rate) for rate in connected_rates],
        "#1971c2",
        "Connected",
    )
    pellet_rates = [result.reachable_pellet_samples / max(1, result.requested_samples) for result in results]
    plot_mean_with_std(
        axes[0, 1],
        levels,
        pellet_rates,
        [rate_std(rate) for rate in pellet_rates],
        "#7048e8",
        "Pellets reachable",
    )
    axes[0, 1].set_title("Feasibility Rates", fontweight="bold")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0, 1].legend()

    plot_mean_with_std(
        axes[1, 0],
        levels,
        [result.average_dead_end_ratio * 10.0 for result in results],
        [result.std_dead_end_ratio * 10.0 for result in results],
        "#c92a2a",
        "Dead-end ratio x10",
    )
    plot_mean_with_std(
        axes[1, 0],
        levels,
        [result.average_branching_factor / 4.0 for result in results],
        [result.std_branching_factor / 4.0 for result in results],
        "#0c8599",
        "Branching / 4",
    )
    plot_mean_with_std(
        axes[1, 0],
        levels,
        [result.average_corridor_length / 10.0 for result in results],
        [result.std_corridor_length / 10.0 for result in results],
        "#5f3dc4",
        "Corridor / 10",
    )
    axes[1, 0].set_title("Maze Structure", fontweight="bold")
    axes[1, 0].set_ylabel("Normalized value")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    bar_width = 0.36
    left_positions = [level - bar_width / 2 for level in levels]
    right_positions = [level + bar_width / 2 for level in levels]
    axes[1, 1].bar(
        left_positions,
        [result.average_nearest_ghost_distance for result in results],
        yerr=[result.std_nearest_ghost_distance for result in results],
        width=bar_width,
        color="#1864ab",
        capsize=4,
        label="Nearest ghost",
    )
    axes[1, 1].bar(
        right_positions,
        [result.average_collection_distance / 10.0 for result in results],
        yerr=[result.std_collection_distance / 10.0 for result in results],
        width=bar_width,
        color="#e67700",
        capsize=4,
        label="Collection / 10",
    )
    axes[1, 1].set_title("Distance Metrics", fontweight="bold")
    axes[1, 1].set_ylabel("Distance")
    axes[1, 1].legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_mean_with_std(
    axis,
    x_values,
    means,
    stds,
    color: str,
    label: str,
    clamp_high: float = 1.0,
) -> None:
    lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
    upper = [min(clamp_high, mean + std) for mean, std in zip(means, stds)]
    axis.plot(x_values, means, marker="o", linewidth=2.5, color=color, label=label)
    axis.fill_between(x_values, lower, upper, color=color, alpha=0.16, linewidth=0)


def rate_std(rate: float) -> float:
    return (max(0.0, rate * (1.0 - rate))) ** 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the maze feasibility experiment.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Mazes to generate per difficulty level.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Maze width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Maze height.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use --seed -1 for no fixed seed.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--visualization", type=Path, help="Optional Matplotlib visualization output path, such as .png, .pdf, or .svg.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = None if args.seed < 0 else args.seed
    results = run_experiment(args.samples, args.width, args.height, seed)
    print_results(results)
    if args.output is not None:
        write_csv(results, args.output)
        print(f"\nSaved CSV results to {args.output}")
    if args.visualization is not None:
        write_visualization(results, args.visualization)
        print(f"Saved visualization to {args.visualization}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
