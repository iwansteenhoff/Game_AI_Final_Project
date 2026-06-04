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
SPIKE = "^"

Position = tuple[int, int]


@dataclass(frozen=True)
class GeneratedMaze:
    grid: list[list[str]]
    pacman_start: Position
    ghost_starts: list[Position]
    ghost_respawn_location: Position


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
        
        _apply_wrap_portals(grid, max_pairs=3)

        if not _has_enough_walls(grid):
            continue
        
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

        respawn_location = _choose_ghost_respawn_location(
            grid,
            distances,
            config.min_start_distance,
            {pacman_start, *ghost_starts},
        )
        if respawn_location is None:
            continue

        _add_pellets(grid, {pacman_start})
        _add_power_pellets(grid, config.power_pellets, distances)

        spike_count = getattr(config, "spike_count", 0)
        protected = {pacman_start, *ghost_starts}
        _add_spikes(grid, spike_count, protected, distances)
        # _enforce_wrap_consistency(grid)

        if is_valid_maze(grid, pacman_start, ghost_starts, respawn_location, config):
            return GeneratedMaze(grid, pacman_start, ghost_starts, respawn_location)

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

# Simulate wrap-around connectivity for validation
def wrap_neighbors(grid: list[list[str]], pos: Position) -> list[Position]:
    width = len(grid[0])
    height = len(grid)
    x, y = pos
    result = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        # Wrap horizontally
        if nx < 0:
            nx = width - 1
        elif nx >= width:
            nx = 0
        # Wrap vertically
        if ny < 0:
            ny = height - 1
        elif ny >= height:
            ny = 0
        if grid[ny][nx] != WALL:
            result.append((nx, ny))
    return result

# BFS with wrap-around
def bfs_wrap(grid: list[list[str]], start: Position) -> dict[Position, int]:
    queue: deque[Position] = deque([start])
    distances = {start: 0}
    while queue:
        pos = queue.popleft()
        for neighbor in wrap_neighbors(grid,pos):
            if neighbor not in distances:
                distances[neighbor] = distances[pos] + 1
                queue.append(neighbor)
    return distances



def is_valid_maze(
    grid: list[list[str]],
    pacman_start: Position,
    ghost_starts: list[Position],
    ghost_respawn_location: Position,
    config: DifficultyConfig,
) -> bool:
    distances = bfs_wrap(grid, pacman_start)
    open_cells = _walkable_cells(grid)

    # Ensure all open cells are reachable from the start without wrap-around
    if any(cell not in distances for cell in open_cells):
        print("Validation failed: Not all open cells are reachable from the start.")
        return False

    # Ensure all pellets and power pellets are reachable
    pellets = [
        (x, y)
        for y, row in enumerate(grid)
        for x, tile in enumerate(row)
        if tile in {PELLET, POWER}
    ]
    if not pellets or any(pellet not in distances for pellet in pellets):
        print("Validation failed: Not all pellets are reachable.")
        return False

    # Pacman must have at least 2 possible moves from the start
    if len(wrap_neighbors(grid, pacman_start)) < 2:
        print("Validation failed: Pacman does not have enough initial moves.")
        return False

    # Ghosts must start at a minimum distance from Pacman
    if any(distances.get(ghost, 0) < config.min_start_distance for ghost in ghost_starts):
        print("Validation failed: Ghosts do not start at a minimum distance from Pacman.")
        return False

    # Respawn location must be a wall next to open space and far enough from Pacman.
    if grid[ghost_respawn_location[1]][ghost_respawn_location[0]] != WALL:
        print("Validation failed: Ghost respawn location must be inside a wall.")
        return False

    adjacent_open_cells = [neighbor for neighbor in neighbors(grid, ghost_respawn_location)]
    if not adjacent_open_cells:
        print("Validation failed: Ghost respawn location is not next to open space.")
        return False

    if min(distances.get(cell, 0) for cell in adjacent_open_cells) < config.min_start_distance:
        print("Validation failed: Ghost respawn location is too close to Pacman.")
        return False

    # Maintain minimum junctions for maze complexity
    junction_count = sum(1 for cell in open_cells if len(wrap_neighbors(grid, cell)) >= 3)
    min_junctions = max(4, (len(grid) * len(grid[0])) // 60)
    return junction_count >= min_junctions

def _has_enough_walls(grid):
    total = len(grid) * len(grid[0])
    walls = sum(cell == WALL for row in grid for cell in row)
    return walls > total * 0.2

def _create_wall_grid(width: int, height: int) -> list[list[str]]:
    return [[WALL for _ in range(width)] for _ in range(height)]

def _carve_depth_first_maze(grid: list[list[str]]) -> None:
    width = len(grid[0])
    height = len(grid)
    # start = (1, 1)
    start = (random.randrange(1, width-1), random.randrange(1, height-1))
    grid[start[1]][start[0]] = EMPTY
    stack : list[tuple[int, int]] = [start]
    visited = {start}

    while stack:
        x, y = stack[-1]
        candidates = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                if grid[ny][nx] == WALL:
                    candidates.append((nx, ny, dx, dy))

        if not candidates:
            stack.pop()
            continue

        nx, ny, dx, dy = random.choice(candidates)

        grid[y + dy // 2][x + dx // 2] = EMPTY
        grid[ny][nx] = EMPTY

        stack.append((nx, ny))
        visited.add((nx, ny))

def _open_extra_cells(grid: list[list[str]], openness: float) -> None:
    width = len(grid[0])
    height = len(grid)
    probability = max(0.0, min(1.0, openness - 0.35))

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] == WALL and random.random() < probability:
                adjacent = sum(
                    grid[y + dy][x + dx] != WALL
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))
                )
                if adjacent >= 3:
                    grid[y][x] = EMPTY

def _apply_wrap_portals(grid, max_pairs=3):
    width = len(grid[0])
    height = len(grid)

    # horizontal portals
    ys = random.sample(range(1, height - 1), k=max_pairs)
    while not (all(b - a >= 1 for a, b in zip(ys, ys[1:]))):
        ys = random.sample(range(1, height - 1), k=max_pairs)
    for y in ys:
        grid[y][0] = EMPTY
        grid[y][width - 1] = EMPTY
        i = 0
        while len(neighbors(grid, (0+i, y))) == 0:
            grid[y][i+1] = EMPTY
            i += 1
        i = 0
        while len(neighbors(grid, (width - 1-i, y))) == 0: 
            grid[y][width - 2-i] = EMPTY
            i += 1          

    # vertical portals
    xs = random.sample(range(1, width - 1), k=max_pairs)
    while not (all(b - a >= 1 for a, b in zip(xs, xs[1:]))):
        xs = random.sample(range(1, width - 1), k=max_pairs)
    for x in xs:
        grid[0][x] = EMPTY
        grid[height - 1][x] = EMPTY
        i = 0
        while len(neighbors(grid, (x, 0+i))) == 0:
            grid[i+1][x] = EMPTY
            i += 1
        i = 0
        while len(neighbors(grid, (x, height - 1-i))) == 0:
            grid[height - 2-i][x] = EMPTY
            i += 1

def _walkable_cells(grid: list[list[str]]) -> list[Position]:
    return [
        (x, y)
        for y, row in enumerate(grid)
        for x, tile in enumerate(row)
        if tile != WALL
    ]

def _add_spikes(
    grid: list[list[str]],
    count: int,
    protected: set[Position],
    distances: dict[Position, int],
) -> None:
    if count <= 0:
        return

    candidates = [
        pos
        for pos, dist in distances.items()
        if dist >= 4
        and grid[pos[1]][pos[0]] == PELLET
        and pos not in protected
        and len(wrap_neighbors(grid, pos)) >= 2
    ]
    candidates.sort(key=lambda p: distances[p], reverse=True)
    random.shuffle(candidates[: min(count * 4, len(candidates))])

    def _safe_walkable_count(center: Position, radius: int = 2) -> int:
        sx, sy = center
        safe = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = sx + dx, sy + dy
                if ny < 0 or ny >= len(grid) or nx < 0 or nx >= len(grid[0]):
                    continue
                cell = grid[ny][nx]
                if cell != WALL and cell != SPIKE:
                    safe += 1
        return safe

    def _would_block_path(spike_pos: Position) -> bool:
        x, y = spike_pos
        grid[y][x] = SPIKE
        walkable = _walkable_cells(grid)
        grid[y][x] = PELLET
        if not walkable:
            return True

        if not walkable:
            return True

        start = walkable[0]
        visited = {start}
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()
            for nx, ny in wrap_neighbors(grid, (cx, cy)):
                if (nx, ny) not in visited and grid[ny][nx] != WALL and grid[ny][nx] != SPIKE:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) < len(walkable)

    placed = 0
    for x, y in candidates:
        if placed >= count:
            break

        if _safe_walkable_count((x, y), radius=1) < 5:
            continue

        if any(grid[ny][nx] == SPIKE for nx, ny in wrap_neighbors(grid, (x, y))):
            continue

        if _would_block_path((x, y)):
            continue
        grid[y][x] = SPIKE
        placed += 1

def _choose_pacman_start(open_cells: list[Position]) -> Position:
    return min(open_cells, key=lambda cell: cell[0] + cell[1])

def _choose_ghost_starts(
    open_cells: list[Position],
    distances: dict[Position, int],
    ghost_count: int,
    min_distance: int,
) -> list[Position]:
    candidates = [cell for cell in open_cells if distances.get(
        cell, 0) >= min_distance]
    candidates.sort(key=lambda cell: distances.get(cell, 0), reverse=True)
    random.shuffle(candidates[: min(8, len(candidates))])

    starts: list[Position] = []
    for cell in candidates:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 3 for other in starts):
            starts.append(cell)
        if len(starts) == ghost_count:
            break
    return starts

def _choose_ghost_respawn_location(
    grid: list[list[str]],
    distances: dict[Position, int],
    min_distance: int,
    blocked: set[Position],
) -> Position | None:
    candidates: list[tuple[Position, int]] = []
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            if tile != WALL or (x, y) in blocked:
                continue

            adjacent_open_cells = neighbors(grid, (x, y))
            if not adjacent_open_cells:
                continue

            nearest_distance = min(distances.get(cell, 0) for cell in adjacent_open_cells)
            if nearest_distance >= min_distance:
                candidates.append(((x, y), nearest_distance))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    shortlist = candidates[: min(8, len(candidates))]
    random.shuffle(shortlist)
    return shortlist[0][0]

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
    if count <= 0:
        return

    pellet_cells = [
        pos
        for pos, dist in distances.items()
        if dist > 4 and grid[pos[1]][pos[0]] == PELLET
    ]
    if not pellet_cells:
        return

    width = len(grid[0])
    height = len(grid)
    center = (width // 2, height // 2)

    def _distance_to_selected(cell: Position, selected: list[Position]) -> int:
        if not selected:
            return 10_000
        return min(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) for other in selected)

    selected: list[Position] = []

    middle_candidate = min(
        pellet_cells,
        key=lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1]),
    )
    if abs(middle_candidate[0] - center[0]) + abs(middle_candidate[1] - center[1]) <= 2:
        selected.append(middle_candidate)

    while len(selected) < count:
        remaining = [cell for cell in pellet_cells if cell not in selected]
        if not remaining:
            break
        next_cell = max(
            remaining,
            key=lambda cell: (
                _distance_to_selected(cell, selected),
                distances[cell],
            ),
        )
        selected.append(next_cell)

    for x, y in selected:
        grid[y][x] = POWER

def _make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1
