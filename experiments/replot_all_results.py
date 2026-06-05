"""Replot all experiment CSV files with report-sized text."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.difficulty import TARGET_HIGH, TARGET_LOW


RESULTS = PROJECT_ROOT / "results"
AGENTS = ("random", "greedy", "cautious")
COLORS = {"random": "#c92a2a", "greedy": "#1971c2", "cautious": "#2f9e44"}


def load_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def stddev(items: list[float]) -> float:
    average = mean(items)
    return (sum((item - average) ** 2 for item in items) / len(items)) ** 0.5 if len(items) > 1 else 0.0


def setup_plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 21,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "figure.titlesize": 31,
        "lines.linewidth": 3.2,
        "lines.markersize": 9,
    })
    return plt


def style_axes(axes) -> None:
    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        axis.grid(True, color="#d1d5db", linewidth=1.1)
        axis.tick_params(width=1.5, length=6)
        axis.text(
            0.01,
            0.98,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
            zorder=10,
        )


def line_with_std(axis, x, means, stds, color, label, high=1.0) -> None:
    lower = [max(0.0, value - std) for value, std in zip(means, stds)]
    upper = [min(high, value + std) for value, std in zip(means, stds)]
    axis.plot(x, means, marker="o", color=color, label=label)
    axis.fill_between(x, lower, upper, color=color, alpha=0.17)


def plot_maze(plt) -> None:
    from matplotlib.ticker import PercentFormatter

    rows = load_csv("maze_feasibility.csv")
    levels = [int(row["level"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    figure.suptitle("Maze Feasibility Experiment", fontweight="bold")
    style_axes(axes)

    line_with_std(axes[0, 0], levels, values(rows, "average_maze_difficulty_score"), values(rows, "std_maze_difficulty_score"), "#d9480f", "Maze score", 5.0)
    axes[0, 0].set_title("Estimated Maze Difficulty", fontweight="bold")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_ylim(0, 5)
    axes[0, 0].legend()

    requested = values(rows, "requested_samples")
    connected = [float(row["connected_samples"]) / requested[index] for index, row in enumerate(rows)]
    reachable = [float(row["reachable_pellet_samples"]) / requested[index] for index, row in enumerate(rows)]
    line_with_std(axes[0, 1], levels, values(rows, "valid_rate"), [0.0] * len(rows), "#2f9e44", "Valid")
    line_with_std(axes[0, 1], levels, connected, [0.0] * len(rows), "#1971c2", "Safely connected")
    line_with_std(axes[0, 1], levels, reachable, [0.0] * len(rows), "#7048e8", "Pellets reachable")
    axes[0, 1].set_title("Feasibility Rates", fontweight="bold")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].legend()

    structure = (
        ("average_dead_end_ratio", "std_dead_end_ratio", 10.0, "#c92a2a", "Dead-end ratio x10"),
        ("average_branching_factor", "std_branching_factor", 0.25, "#0c8599", "Branching / 4"),
        ("average_corridor_length", "std_corridor_length", 0.1, "#5f3dc4", "Corridor / 10"),
        ("average_spikes", "std_spikes", 0.1, "#e67700", "Spikes / 10"),
    )
    for average_field, std_field, scale, color, label in structure:
        line_with_std(
            axes[1, 0],
            levels,
            [item * scale for item in values(rows, average_field)],
            [item * scale for item in values(rows, std_field)],
            color,
            label,
        )
    axes[1, 0].set_title("Maze Structure", fontweight="bold")
    axes[1, 0].set_ylabel("Normalized value")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    width = 0.36
    axes[1, 1].bar([level - width / 2 for level in levels], values(rows, "average_nearest_ghost_distance"), yerr=values(rows, "std_nearest_ghost_distance"), width=width, color="#1864ab", capsize=6, label="Nearest ghost")
    axes[1, 1].bar([level + width / 2 for level in levels], [item / 10 for item in values(rows, "average_collection_distance")], yerr=[item / 10 for item in values(rows, "std_collection_distance")], width=width, color="#e67700", capsize=6, label="Collection / 10")
    axes[1, 1].set_title("Distance Metrics", fontweight="bold")
    axes[1, 1].set_ylabel("Distance")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Difficulty level")
        axis.set_xticks(levels)
    figure.savefig(RESULTS / "maze_feasibility_large_text.png", dpi=180)
    plt.close(figure)


def plot_difficulty_and_ghosts(plt) -> None:
    from matplotlib.ticker import PercentFormatter

    difficulty = load_csv("difficulty_calibration.csv")
    ghosts = load_csv("ghost_comparison.csv")
    levels = sorted({int(row["difficulty"]) for row in difficulty})
    figure, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    figure.suptitle("Difficulty Calibration and Ghost Comparison", fontweight="bold")
    style_axes(axes)

    for agent in AGENTS:
        rows = [row for row in difficulty if row["pacman_agent"] == agent]
        line_with_std(axes[0, 0], levels, values(rows, "win_rate"), values(rows, "std_win_rate"), COLORS[agent], agent)
        line_with_std(axes[0, 1], levels, values(rows, "average_pellet_completion"), values(rows, "std_pellet_completion"), COLORS[agent], agent)
        line_with_std(axes[1, 0], levels, values(rows, "average_score"), values(rows, "std_score"), COLORS[agent], agent)

    titles = ("Win Rate by Difficulty", "Pellet Completion by Difficulty", "Performance Score by Difficulty")
    labels = ("Win rate", "Pellet completion", "Score")
    for axis, title, label in zip((axes[0, 0], axes[0, 1], axes[1, 0]), titles, labels):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Difficulty level")
        axis.set_ylabel(label)
        axis.set_xticks(levels)
        axis.set_ylim(0, 1.05)
        axis.legend()
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))

    names = [row["ghost_setting"] for row in ghosts]
    positions = list(range(len(names)))
    max_steps = max(values(ghosts, "average_steps_until_end"))
    axes[1, 1].bar([x - 0.24 for x in positions], values(ghosts, "win_rate"), yerr=values(ghosts, "std_win_rate"), width=0.24, color="#2f9e44", capsize=6, label="Win rate")
    axes[1, 1].bar(positions, values(ghosts, "average_pellet_completion"), yerr=values(ghosts, "std_pellet_completion"), width=0.24, color="#1971c2", capsize=6, label="Pellet completion")
    axes[1, 1].bar([x + 0.24 for x in positions], [item / max_steps for item in values(ghosts, "average_steps_until_end")], yerr=[item / max_steps for item in values(ghosts, "std_steps_until_end")], width=0.24, color="#e67700", capsize=6, label="Steps, normalized")
    axes[1, 1].set_title("Ghost Agent Comparison", fontweight="bold")
    axes[1, 1].set_ylabel("Normalized metric")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xticks(positions, names)
    axes[1, 1].legend()

    figure.savefig(RESULTS / "difficulty_and_ghosts_large_text.png", dpi=180)
    plt.close(figure)


def plot_adaptive(plt) -> None:
    from matplotlib.ticker import PercentFormatter

    rows = load_csv("adaptive_difficulty.csv")
    figure, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    figure.suptitle("Adaptive Difficulty by Pacman Agent", fontweight="bold")
    style_axes(axes)

    fields = ("difficulty_before", "performance_score", "won", "pellet_completion")
    for agent in AGENTS:
        agent_rows = [row for row in rows if row["pacman_agent"] == agent]
        episodes = sorted({int(row["episode"]) for row in agent_rows})
        for axis, field, high in zip(axes.flat, fields, (5.0, 1.0, 1.0, 1.0)):
            means, stds = [], []
            for episode in episodes:
                episode_rows = [row for row in agent_rows if int(row["episode"]) == episode]
                episode_values = [1.0 if row[field] == "True" else 0.0 for row in episode_rows] if field == "won" else values(episode_rows, field)
                means.append(mean(episode_values))
                stds.append(stddev(episode_values))
            line_with_std(axis, episodes, means, stds, COLORS[agent], agent, high)

    titles = ("Difficulty Over Time", "Performance Score Over Time", "Win Rate Over Time", "Pellet Completion Over Time")
    labels = ("Difficulty level", "Score", "Win rate", "Pellet completion")
    limits = ((-0.2, 5.2), (0, 1.05), (-0.1, 1.1), (0, 1.05))
    for axis, title, label, limit in zip(axes.flat, titles, labels, limits):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Episode")
        axis.set_ylabel(label)
        axis.set_ylim(*limit)
        axis.legend()
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))

    performance = axes[0, 1]
    performance.axhspan(TARGET_LOW, TARGET_HIGH, color="#74b816", alpha=0.30, label=f"Target zone ({TARGET_LOW:.2f}-{TARGET_HIGH:.2f})", zorder=0)
    performance.axhline(TARGET_LOW, color="#2b8a3e", linestyle="--", linewidth=3)
    performance.axhline(TARGET_HIGH, color="#2b8a3e", linestyle="--", linewidth=3)
    performance.text(0.98, (TARGET_LOW + TARGET_HIGH) / 2, "TARGET ZONE", transform=performance.get_yaxis_transform(), ha="right", va="center", fontsize=20, fontweight="bold", color="#1b4332")
    performance.legend()

    figure.savefig(RESULTS / "adaptive_difficulty_large_text.png", dpi=180)
    plt.close(figure)


def main() -> int:
    plt = setup_plotting()
    plot_maze(plt)
    plot_difficulty_and_ghosts(plt)
    plot_adaptive(plt)
    print("Saved all large-text plots in results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
