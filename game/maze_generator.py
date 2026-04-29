"""Procedural maze generation and validation."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from game.difficulty import DifficultyConfig

WALL = "#"
EMPTY = " "
PELLET = "."
POWER = "o"

Position = tuple[int, int]


@dataclass(frozen=True)
class GeneratedMaze:
    grid: list[list[str]]
    pacman_start: Position
    ghost_starts: list[Position]


def generate_maze(
    width: int,
    height: int,
    config: DifficultyConfig,
    max_attempts: int = 250,
) -> GeneratedMaze:
    width = _make_odd(max(11, width))
    height = _make_odd(max(11, height))

    for _ in range(max_attempts):
        grid = _create_wall_grid(width, height)
        _carve_depth_first_maze(grid)
        _open_extra_cells(grid, config.maze_openness)

        open_cells = _walkable_cells(grid)
        if len(open_cells) < (width * height * 0.25):
            continue

        pacman_start = _choose_pacman_start(open_cells)
        distances = bfs_distances(grid, pacman_start)
        ghost_starts = _choose_ghost_starts(
            open_cells,
            distances,
            config.ghost_count,
            config.min_start_distance,
        )
        if len(ghost_starts) != config.ghost_count:
            continue

        _add_pellets(grid, {pacman_start, *ghost_starts})
        _add_power_pellets(grid, config.power_pellets, distances)

        if is_valid_maze(grid, pacman_start, ghost_starts, config):
            return GeneratedMaze(grid, pacman_start, ghost_starts)

    raise RuntimeError("Could not generate a valid maze after many attempts.")


def bfs_distances(grid: list[list[str]], start: Position) -> dict[Position, int]:
    queue: deque[Position] = deque([start])
    distances = {start: 0}

    while queue:
        pos = queue.popleft()
        for neighbor in neighbors(grid, pos):
            if neighbor not in distances:
                distances[neighbor] = distances[pos] + 1
                queue.append(neighbor)

    return distances


def shortest_path_distance(grid: list[list[str]], start: Position, goal: Position) -> int:
    if start == goal:
        return 0
    distances = bfs_distances(grid, start)
    return distances.get(goal, 10_000)


def neighbors(grid: list[list[str]], pos: Position) -> list[Position]:
    x, y = pos
    candidates = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
    return [
        (nx, ny)
        for nx, ny in candidates
        if 0 <= ny < len(grid)
        and 0 <= nx < len(grid[0])
        and grid[ny][nx] != WALL
    ]


def is_valid_maze(
    grid: list[list[str]],
    pacman_start: Position,
    ghost_starts: list[Position],
    config: DifficultyConfig,
) -> bool:
    distances = bfs_distances(grid, pacman_start)
    open_cells = _walkable_cells(grid)

    if any(cell not in distances for cell in open_cells):
        return False

    pellets = [
        (x, y)
        for y, row in enumerate(grid)
        for x, tile in enumerate(row)
        if tile in {PELLET, POWER}
    ]
    if not pellets or any(pellet not in distances for pellet in pellets):
        return False

    if len(neighbors(grid, pacman_start)) < 2:
        return False

    if any(distances.get(ghost, 0) < config.min_start_distance for ghost in ghost_starts):
        return False

    junction_count = sum(1 for cell in open_cells if len(neighbors(grid, cell)) >= 3)
    min_junctions = max(4, (len(grid) * len(grid[0])) // 60)
    return junction_count >= min_junctions


def _create_wall_grid(width: int, height: int) -> list[list[str]]:
    return [[WALL for _ in range(width)] for _ in range(height)]


def _carve_depth_first_maze(grid: list[list[str]]) -> None:
    width = len(grid[0])
    height = len(grid)
    start = (1, 1)
    grid[start[1]][start[0]] = EMPTY
    stack = [start]

    while stack:
        x, y = stack[-1]
        candidates = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and grid[ny][nx] == WALL:
                candidates.append((nx, ny, dx, dy))

        if not candidates:
            stack.pop()
            continue

        nx, ny, dx, dy = random.choice(candidates)
        grid[y + dy // 2][x + dx // 2] = EMPTY
        grid[ny][nx] = EMPTY
        stack.append((nx, ny))


def _open_extra_cells(grid: list[list[str]], openness: float) -> None:
    width = len(grid[0])
    height = len(grid)
    probability = max(0.0, min(1.0, openness - 0.35))

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] == WALL and random.random() < probability:
                adjacent_open = sum(
                    grid[ny][nx] != WALL
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                )
                if adjacent_open >= 2:
                    grid[y][x] = EMPTY


def _walkable_cells(grid: list[list[str]]) -> list[Position]:
    return [
        (x, y)
        for y, row in enumerate(grid)
        for x, tile in enumerate(row)
        if tile != WALL
    ]


def _choose_pacman_start(open_cells: list[Position]) -> Position:
    return min(open_cells, key=lambda cell: cell[0] + cell[1])


def _choose_ghost_starts(
    open_cells: list[Position],
    distances: dict[Position, int],
    ghost_count: int,
    min_distance: int,
) -> list[Position]:
    candidates = [cell for cell in open_cells if distances.get(cell, 0) >= min_distance]
    candidates.sort(key=lambda cell: distances.get(cell, 0), reverse=True)
    random.shuffle(candidates[: min(8, len(candidates))])

    starts: list[Position] = []
    for cell in candidates:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 3 for other in starts):
            starts.append(cell)
        if len(starts) == ghost_count:
            break
    return starts


def _add_pellets(grid: list[list[str]], blocked: set[Position]) -> None:
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            if tile == EMPTY and (x, y) not in blocked:
                grid[y][x] = PELLET


def _add_power_pellets(
    grid: list[list[str]],
    count: int,
    distances: dict[Position, int],
) -> None:
    pellet_cells = [
        pos
        for pos, dist in distances.items()
        if dist > 4 and grid[pos[1]][pos[0]] == PELLET
    ]
    pellet_cells.sort(key=lambda pos: distances[pos], reverse=True)
    selected = pellet_cells[: max(0, count)]
    for x, y in selected:
        grid[y][x] = POWER


def _make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1
