"""Replot adaptive-difficulty CSV results using Pacman agent names."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.difficulty import TARGET_HIGH, TARGET_LOW


AGENTS = ("random", "greedy", "cautious")
COLORS = {
    "random": "#c92a2a",
    "greedy": "#1971c2",
    "cautious": "#2f9e44",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = average(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def plot_metric(
    axis,
    episodes: list[int],
    rows: list[dict[str, str]],
    value_fn: Callable[[dict[str, str]], float],
    color: str,
    label: str,
    clamp_high: float = 1.0,
) -> None:
    means = []
    stds = []
    for episode in episodes:
        values = [
            value_fn(row)
            for row in rows
            if int(row["episode"]) == episode
        ]
        means.append(average(values))
        stds.append(stddev(values))

    lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
    upper = [min(clamp_high, mean + std) for mean, std in zip(means, stds)]
    axis.plot(episodes, means, marker="o", linewidth=2.3, color=color, label=label)
    axis.fill_between(episodes, lower, upper, color=color, alpha=0.16)


def create_plot(rows: list[dict[str, str]], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle("Adaptive Difficulty by Pacman Agent", fontsize=20, fontweight="bold")

    for axis in axes.flat:
        axis.grid(True, color="#d1d5db", linewidth=0.9)
        axis.set_xlabel("Episode")

    for agent in AGENTS:
        agent_rows = [row for row in rows if row["pacman_agent"] == agent]
        episodes = sorted({int(row["episode"]) for row in agent_rows})
        color = COLORS[agent]
        plot_metric(axes[0, 0], episodes, agent_rows, lambda row: float(row["difficulty_before"]), color, agent, 5.0)
        plot_metric(axes[0, 1], episodes, agent_rows, lambda row: float(row["performance_score"]), color, agent)
        plot_metric(axes[1, 0], episodes, agent_rows, lambda row: 1.0 if row["won"] == "True" else 0.0, color, agent)
        plot_metric(axes[1, 1], episodes, agent_rows, lambda row: float(row["pellet_completion"]), color, agent)

    axes[0, 0].set_title("Difficulty Over Time", fontweight="bold")
    axes[0, 0].set_ylabel("Difficulty level")
    axes[0, 0].set_ylim(-0.2, 5.2)
    axes[0, 0].legend()

    performance_axis = axes[0, 1]
    performance_axis.set_title("Performance Score Over Time", fontweight="bold")
    performance_axis.set_ylabel("Score")
    performance_axis.set_ylim(0, 1.05)
    performance_axis.axhspan(
        TARGET_LOW,
        TARGET_HIGH,
        color="#74b816",
        alpha=0.30,
        label=f"Target zone ({TARGET_LOW:.2f}-{TARGET_HIGH:.2f})",
        zorder=0,
    )
    performance_axis.axhline(TARGET_LOW, color="#2b8a3e", linestyle="--", linewidth=2)
    performance_axis.axhline(TARGET_HIGH, color="#2b8a3e", linestyle="--", linewidth=2)
    performance_axis.text(
        0.99,
        (TARGET_LOW + TARGET_HIGH) / 2,
        "TARGET ZONE",
        transform=performance_axis.get_yaxis_transform(),
        ha="right",
        va="center",
        color="#1b4332",
        fontweight="bold",
    )
    performance_axis.legend()

    axes[1, 0].set_title("Win Rate Over Time", fontweight="bold")
    axes[1, 0].set_ylabel("Win rate")
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1, 0].legend()

    axes[1, 1].set_title("Pellet Completion Over Time", fontweight="bold")
    axes[1, 1].set_ylabel("Pellet completion")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1, 1].legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/adaptive_difficulty.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/adaptive_difficulty_agents.png"),
    )
    args = parser.parse_args()

    create_plot(load_rows(args.input), args.output)
    print(f"Saved renamed adaptive difficulty plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
