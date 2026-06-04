"""Difficulty calibration and ghost-agent comparison experiments."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import ghost_heuristic, ghost_mcts, ghost_random
from game.difficulty import DIFFICULTY_LEVELS, DifficultyConfig, get_config, performance_score
from game.maze_generator import PELLET, POWER, SPIKE, bfs_distances, generate_balanced_maze
from game.pacman_env import PacmanEnv


GRID_WIDTH = 21
GRID_HEIGHT = 21
DEFAULT_EPISODES = 10
DEFAULT_GHOST_COMPARISON_LEVEL = 3
DEFAULT_MAX_STEPS = 250
DEFAULT_MCTS_ROLLOUTS = 6
DEFAULT_MCTS_DEPTH = 5
PACMAN_AGENTS = ("random", "greedy", "cautious")
GHOST_SETTINGS = (
    ("random", "random", 0.0),
    ("heuristic", "heuristic", 1.0),
    ("mcts", "mcts", 1.0),
)


@dataclass(frozen=True)
class EpisodeResult:
    won: bool
    steps: int
    pellets_collected: int
    total_pellets: int
    power_pellets_collected: int
    ghosts_eaten: int
    score: float
    average_ghost_distance: float

    @property
    def pellet_completion(self) -> float:
        return self.pellets_collected / max(1, self.total_pellets)


@dataclass(frozen=True)
class DifficultyCalibrationResult:
    pacman_agent: str
    difficulty: int
    episodes: int
    win_rate: float
    average_pellet_completion: float
    average_survival_time: float
    average_score: float
    average_ghosts_eaten: float


@dataclass(frozen=True)
class GhostComparisonResult:
    ghost_setting: str
    episodes: int
    win_rate: float
    average_steps_until_end: float
    average_pellet_completion: float
    average_ghost_distance: float


def run_difficulty_calibration(
    episodes: int,
    seed: int | None,
    candidates: int,
    max_steps: int,
) -> list[DifficultyCalibrationResult]:
    if seed is not None:
        random.seed(seed)

    results = []
    for pacman_agent in PACMAN_AGENTS:
        for difficulty in sorted(DIFFICULTY_LEVELS):
            episode_results = [
                run_episode(
                    config=get_config(difficulty),
                    target_difficulty=difficulty,
                    pacman_agent=pacman_agent,
                    candidates=candidates,
                    max_steps=max_steps,
                )
                for _ in range(episodes)
            ]
            results.append(
                DifficultyCalibrationResult(
                    pacman_agent=pacman_agent,
                    difficulty=difficulty,
                    episodes=len(episode_results),
                    win_rate=average(result.won for result in episode_results),
                    average_pellet_completion=average(result.pellet_completion for result in episode_results),
                    average_survival_time=average(result.steps for result in episode_results),
                    average_score=average(result.score for result in episode_results),
                    average_ghosts_eaten=average(result.ghosts_eaten for result in episode_results),
                )
            )
    return results


def run_ghost_comparison(
    episodes: int,
    seed: int | None,
    difficulty: int,
    pacman_agent: str,
    candidates: int,
    max_steps: int,
) -> list[GhostComparisonResult]:
    if seed is not None:
        random.seed(seed)

    base_config = get_config(difficulty)
    grouped_results: dict[str, list[EpisodeResult]] = {name: [] for name, _agent, _aggression in GHOST_SETTINGS}

    for _ in range(episodes):
        maze = generate_balanced_maze(GRID_WIDTH, GRID_HEIGHT, base_config, difficulty, candidates=candidates)
        greedy_solution = count_pellets(maze.grid)

        for name, ghost_agent, aggression in GHOST_SETTINGS:
            config = replace(base_config, ghost_agent=ghost_agent, ghost_aggression=aggression)
            grouped_results[name].append(
                run_episode_on_maze(
                    maze=maze,
                    config=config,
                    pacman_agent=pacman_agent,
                    generated_greedy_solution=greedy_solution,
                    max_steps=max_steps,
                )
            )

    return [
        GhostComparisonResult(
            ghost_setting=name,
            episodes=len(results),
            win_rate=average(result.won for result in results),
            average_steps_until_end=average(result.steps for result in results),
            average_pellet_completion=average(result.pellet_completion for result in results),
            average_ghost_distance=average(result.average_ghost_distance for result in results),
        )
        for name, results in grouped_results.items()
    ]


def run_episode(
    config: DifficultyConfig,
    target_difficulty: int,
    pacman_agent: str,
    candidates: int,
    max_steps: int,
) -> EpisodeResult:
    maze = generate_balanced_maze(GRID_WIDTH, GRID_HEIGHT, config, target_difficulty, candidates=candidates)
    return run_episode_on_maze(maze, config, pacman_agent, count_pellets(maze.grid), max_steps)


def run_episode_on_maze(
    maze,
    config: DifficultyConfig,
    pacman_agent: str,
    generated_greedy_solution: int,
    max_steps: int,
) -> EpisodeResult:
    env = PacmanEnv(
        maze.grid,
        maze.pacman_start,
        maze.ghost_starts,
        max_steps=max_steps,
        maze_difficulty_score=maze.difficulty_score,
        generated_greedy_solution=generated_greedy_solution,
    )
    ghost_distances = []

    while not env.done:
        ghost_distances.append(nearest_ghost_distance(env))
        pacman_action = choose_pacman_action(env, pacman_agent)
        env.step(pacman_action, choose_ghost_actions(env, config))
        for _ in range(config.ghost_speed - 1):
            if env.done:
                break
            env.step("STAY", choose_ghost_actions(env, config))

    score = performance_score(
        env.pellets_collected,
        env.total_pellets,
        env.steps,
        env.max_steps,
        env.traveled_steps,
        env.generated_greedy_solution,
        env.won,
        1 if env.won else 0,
        env.power_pellets_collected,
        int(env.ghosts_eaten),
    )
    return EpisodeResult(
        won=env.won,
        steps=env.steps,
        pellets_collected=env.pellets_collected,
        total_pellets=env.total_pellets,
        power_pellets_collected=env.power_pellets_collected,
        ghosts_eaten=int(env.ghosts_eaten),
        score=score,
        average_ghost_distance=average(ghost_distances),
    )


def choose_pacman_action(env: PacmanEnv, agent_name: str) -> str:
    if agent_name == "greedy":
        return greedy_pellet_action(env)
    if agent_name == "cautious":
        return cautious_greedy_action(env)
    return random_pacman_action(env)


def random_pacman_action(env: PacmanEnv) -> str:
    actions = safe_pacman_actions(env)
    non_stay = [action for action in actions if action != "STAY"]
    return random.choice(non_stay or actions or ["STAY"])


def greedy_pellet_action(env: PacmanEnv) -> str:
    pellets = pellet_positions(env)
    actions = safe_pacman_actions(env)
    if not pellets or not actions:
        return "STAY"

    return min(
        actions,
        key=lambda action: (
            nearest_pellet_distance(env, env.next_position(env.pacman_pos, action), pellets),
            action == "STAY",
        ),
    )


def cautious_greedy_action(env: PacmanEnv) -> str:
    pellets = pellet_positions(env)
    actions = safe_pacman_actions(env)
    if not actions:
        return "STAY"

    def action_score(action: str) -> float:
        next_pos = env.next_position(env.pacman_pos, action)
        pellet_distance = nearest_pellet_distance(env, next_pos, pellets) if pellets else 0
        ghost_distance = min(
            (env.shortest_path_distance(next_pos, ghost_pos) for ghost_pos in env.ghost_positions),
            default=20,
        )
        danger_penalty = max(0, 5 - ghost_distance) * 8
        stay_penalty = 2 if action == "STAY" else 0
        return pellet_distance + danger_penalty + stay_penalty

    return min(actions, key=action_score)


def safe_pacman_actions(env: PacmanEnv) -> list[str]:
    actions = []
    for action in env.legal_actions_for_pacman():
        nx, ny = env.next_position(env.pacman_pos, action)
        if env.grid[ny][nx] != SPIKE:
            actions.append(action)
    return actions


def pellet_positions(env: PacmanEnv) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y, row in enumerate(env.grid)
        for x, tile in enumerate(row)
        if tile in {PELLET, POWER}
    ]


def count_pellets(grid: list[list[str]]) -> int:
    return sum(tile in {PELLET, POWER} for row in grid for tile in row)


def nearest_pellet_distance(env: PacmanEnv, pos: tuple[int, int], pellets: list[tuple[int, int]]) -> int:
    distances = bfs_distances(env.grid, pos)
    return min((distances.get(pellet, 10_000) for pellet in pellets), default=0)


def nearest_ghost_distance(env: PacmanEnv) -> int:
    return min(
        (env.shortest_path_distance(env.pacman_pos, ghost_pos) for ghost_pos in env.ghost_positions),
        default=0,
    )


def choose_ghost_actions(env: PacmanEnv, config: DifficultyConfig) -> list[str]:
    actions = []
    for ghost_id in range(len(env.ghost_positions)):
        if random.random() > config.ghost_aggression:
            actions.append(ghost_random.choose_action(env, ghost_id))
        elif config.ghost_agent == "mcts":
            actions.append(ghost_mcts.choose_action(env, ghost_id))
        elif config.ghost_agent == "heuristic":
            actions.append(ghost_heuristic.choose_action(env, ghost_id))
        else:
            actions.append(ghost_random.choose_action(env, ghost_id))
    return actions


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def print_difficulty_results(results: list[DifficultyCalibrationResult]) -> None:
    print("\nDifficulty calibration")
    print("agent difficulty episodes win% pellet% steps score ghosts_eaten")
    print("------------------------------------------------------------")
    for result in results:
        print(
            f"{result.pacman_agent:>8} {result.difficulty:>10} {result.episodes:>8} "
            f"{result.win_rate:>4.0%} {result.average_pellet_completion:>7.0%} "
            f"{result.average_survival_time:>5.1f} {result.average_score:>5.2f} "
            f"{result.average_ghosts_eaten:>12.2f}"
        )


def print_ghost_results(results: list[GhostComparisonResult]) -> None:
    print("\nGhost comparison")
    print("ghost episodes win% pellet% steps avg_ghost_dist")
    print("-----------------------------------------------")
    for result in results:
        print(
            f"{result.ghost_setting:>9} {result.episodes:>8} {result.win_rate:>4.0%} "
            f"{result.average_pellet_completion:>7.0%} {result.average_steps_until_end:>5.1f} "
            f"{result.average_ghost_distance:>14.2f}"
        )


def write_visualization(
    difficulty_results: list[DifficultyCalibrationResult],
    ghost_results: list[GhostComparisonResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle("Difficulty Calibration and Ghost Comparison", fontsize=20, fontweight="bold")
    figure.patch.set_facecolor("#f8fafc")

    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.grid(True, color="#e5e7eb", linewidth=0.9)
        for spine in axis.spines.values():
            spine.set_color("#d1d5db")

    levels = sorted(DIFFICULTY_LEVELS)
    colors = {"random": "#2f9e44", "greedy": "#1971c2", "cautious": "#7048e8"}

    for agent in PACMAN_AGENTS:
        agent_rows = [row for row in difficulty_results if row.pacman_agent == agent]
        axes[0, 0].plot(levels, [row.win_rate for row in agent_rows], marker="o", linewidth=2.5, color=colors[agent], label=agent)
        axes[0, 1].plot(levels, [row.average_pellet_completion for row in agent_rows], marker="o", linewidth=2.5, color=colors[agent], label=agent)
        axes[1, 0].plot(levels, [row.average_score for row in agent_rows], marker="o", linewidth=2.5, color=colors[agent], label=agent)

    axes[0, 0].set_title("Win Rate by Difficulty", fontweight="bold")
    axes[0, 0].set_xlabel("Difficulty level")
    axes[0, 0].set_ylabel("Win rate")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_xticks(levels)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0, 0].legend()

    axes[0, 1].set_title("Pellet Completion by Difficulty", fontweight="bold")
    axes[0, 1].set_xlabel("Difficulty level")
    axes[0, 1].set_ylabel("Pellet completion")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xticks(levels)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0, 1].legend()

    axes[1, 0].set_title("Performance Score by Difficulty", fontweight="bold")
    axes[1, 0].set_xlabel("Difficulty level")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xticks(levels)
    axes[1, 0].legend()

    ghost_names = [row.ghost_setting for row in ghost_results]
    x_positions = range(len(ghost_names))
    axes[1, 1].bar(
        [x - 0.24 for x in x_positions],
        [row.win_rate for row in ghost_results],
        width=0.24,
        color="#2f9e44",
        label="Win rate",
    )
    axes[1, 1].bar(
        x_positions,
        [row.average_pellet_completion for row in ghost_results],
        width=0.24,
        color="#1971c2",
        label="Pellet completion",
    )
    max_steps = max((row.average_steps_until_end for row in ghost_results), default=1.0)
    axes[1, 1].bar(
        [x + 0.24 for x in x_positions],
        [row.average_steps_until_end / max_steps for row in ghost_results],
        width=0.24,
        color="#e67700",
        label="Steps, normalized",
    )
    axes[1, 1].set_title("Ghost Agent Comparison", fontweight="bold")
    axes[1, 1].set_ylabel("Normalized metric")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xticks(list(x_positions), ghost_names)
    axes[1, 1].legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run difficulty calibration and ghost comparison experiments.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Episodes per condition.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use --seed -1 for no fixed seed.")
    parser.add_argument("--candidates", type=int, default=2, help="Candidate mazes considered per generated level.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum simulated steps per episode.")
    parser.add_argument("--mcts-rollouts", type=int, default=DEFAULT_MCTS_ROLLOUTS, help="Rollouts per MCTS action during experiments.")
    parser.add_argument("--mcts-depth", type=int, default=DEFAULT_MCTS_DEPTH, help="Rollout depth for MCTS during experiments.")
    parser.add_argument("--ghost-difficulty", type=int, default=DEFAULT_GHOST_COMPARISON_LEVEL, help="Fixed difficulty for ghost comparison.")
    parser.add_argument("--ghost-pacman-agent", choices=PACMAN_AGENTS, default="cautious", help="Pacman baseline used for ghost comparison.")
    parser.add_argument("--difficulty-output", type=Path, help="Optional CSV path for difficulty calibration results.")
    parser.add_argument("--ghost-output", type=Path, help="Optional CSV path for ghost comparison results.")
    parser.add_argument("--visualization", type=Path, help="Optional Matplotlib visualization output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = None if args.seed < 0 else args.seed
    ghost_mcts.ROLLOUTS_PER_ACTION = args.mcts_rollouts
    ghost_mcts.ROLLOUT_DEPTH = args.mcts_depth
    difficulty_results = run_difficulty_calibration(args.episodes, seed, args.candidates, args.max_steps)
    ghost_results = run_ghost_comparison(args.episodes, seed, args.ghost_difficulty, args.ghost_pacman_agent, args.candidates, args.max_steps)

    print_difficulty_results(difficulty_results)
    print_ghost_results(ghost_results)

    if args.difficulty_output is not None:
        write_csv(args.difficulty_output, difficulty_results)
        print(f"\nSaved difficulty calibration CSV to {args.difficulty_output}")
    if args.ghost_output is not None:
        write_csv(args.ghost_output, ghost_results)
        print(f"Saved ghost comparison CSV to {args.ghost_output}")
    if args.visualization is not None:
        write_visualization(difficulty_results, ghost_results, args.visualization)
        print(f"Saved visualization to {args.visualization}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
