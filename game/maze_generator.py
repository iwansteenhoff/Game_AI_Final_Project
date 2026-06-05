"""Procedural maze generation and validation."""

from __future__ import annotations

import math
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
    difficulty_score: float = 0.0


@dataclass(frozen=True)
class MazeAnalysis:
    connected: bool
    all_pellets_reachable: bool
    open_cell_count: int
    pellet_count: int
    dead_end_ratio: float
    junction_count: int
    average_branching_factor: float
    average_corridor_length: float
    nearest_ghost_distance: int
    estimated_collection_distance: int
    difficulty_score: float


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
        distances = bfs_wrap(grid, pacman_start)
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

        _add_pellets(grid, {pacman_start, *ghost_starts})
        _add_power_pellets(grid, config.power_pellets, distances)

        spike_count = getattr(config, "spike_count", 0)
        protected = {pacman_start, *ghost_starts}
        _add_spikes(grid, spike_count, protected, distances)
        # _enforce_wrap_consistency(grid)

        if is_valid_maze(grid, pacman_start, ghost_starts, respawn_location, config):
            return GeneratedMaze(grid, pacman_start, ghost_starts, respawn_location)

    raise RuntimeError("Could not generate a valid maze after many attempts.")


def generate_balanced_maze(
    width: int,
    height: int,
    config: DifficultyConfig,
    target_difficulty: int,
    candidates: int = 12,
) -> GeneratedMaze:
    best_maze: GeneratedMaze | None = None
    best_distance = float("inf")

    for _ in range(max(1, candidates)):
        maze = generate_maze(width, height, config)
        analysis = analyze_maze(
            maze.grid, maze.pacman_start, maze.ghost_starts, config)
        target_score = float(target_difficulty)
        distance = abs(analysis.difficulty_score - target_score)

        if distance < best_distance:
            best_distance = distance
            best_maze = GeneratedMaze(
                maze.grid,
                maze.pacman_start,
                maze.ghost_starts,
                maze.ghost_respawn_location,
                analysis.difficulty_score,
            )

    if best_maze is None:
        raise RuntimeError("Could not generate a balanced maze.")
    return best_maze


def analyze_maze(
    grid: list[list[str]],
    pacman_start: Position,
    ghost_starts: list[Position],
    config: DifficultyConfig,
) -> MazeAnalysis:
    distances = bfs_wrap_avoiding_spikes(grid, pacman_start)
    open_cells = _safe_walkable_cells(grid)
    pellets = [
        (x, y)
        for y, row in enumerate(grid)
        for x, tile in enumerate(row)
        if tile in {PELLET, POWER}
    ]
    connected = all(cell in distances for cell in open_cells)
    all_pellets_reachable = all(pellet in distances for pellet in pellets)

    branch_counts = [len(safe_wrap_neighbors(grid, cell))
                     for cell in open_cells]
    dead_ends = sum(1 for count in branch_counts if count == 1)
    junction_count = sum(1 for count in branch_counts if count >= 3)
    dead_end_ratio = dead_ends / max(1, len(open_cells))
    average_branching_factor = sum(branch_counts) / max(1, len(branch_counts))
    average_corridor_length = _average_corridor_length(grid, open_cells)
    nearest_ghost_distance = min(
        (distances.get(ghost, 10_000) for ghost in ghost_starts),
        default=10_000,
    )
    estimated_collection_distance = _estimate_collection_distance(
        grid, pacman_start, pellets)

    score = _maze_difficulty_score(
        config,
        dead_end_ratio,
        average_branching_factor,
        average_corridor_length,
        nearest_ghost_distance,
        estimated_collection_distance,
        len(pellets),
    )

    return MazeAnalysis(
        connected,
        all_pellets_reachable,
        len(open_cells),
        len(pellets),
        dead_end_ratio,
        junction_count,
        average_branching_factor,
        average_corridor_length,
        nearest_ghost_distance,
        estimated_collection_distance,
        score,
    )


def shortest_path_distance(grid: list[list[str]], start: Position, goal: Position) -> int:
    if start == goal:
        return 0
    distances = bfs_wrap(grid, start)
    return distances.get(goal, 10_000)

# ---------------------SHORTEST CLEAR USING GREEDY STRATEGY WITH 2 OPT REFINEMENT------------------------------


@staticmethod
def solve(maze: GeneratedMaze) -> int:
    '''
    Solves the maze using a greedy strategy followed by 2-opt refinement.
    Returns the total distance of the route that collects all pellets and power pellets.
    This route's length is meant to be compared with the total amount of moves the player makes.
    '''
    ball_positions = [
        (x, y)
        for y, row in enumerate(maze.grid)
        for x, tile in enumerate(row)
        if tile in {PELLET, POWER}
    ]

    all_nodes = [maze.pacman_start] + ball_positions
    matrix = {pos: bfs_wrap_avoiding_spikes(maze.grid, pos)
              for pos in all_nodes}

    route = greedy_route(maze.pacman_start, ball_positions, matrix)
    route = two_opt(maze.pacman_start, route, matrix)

    return route_cost(maze.pacman_start, route, matrix)


def greedy_route(pacman_start: Position, ball_positions: list[Position], matrix: dict[Position, dict[Position, int]]) -> list[Position]:
    unvisited = set(ball_positions)
    route = []
    current = pacman_start

    while unvisited:
        nearest = min(unvisited, key=lambda pos: matrix[current][pos])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return route


def route_cost(pacman_start: Position, route: list[Position], matrix: dict[Position, dict[Position, int]]) -> int:
    if not route:
        return 0
    total = matrix[pacman_start][route[0]]
    for i in range(len(route) - 1):
        total += matrix[route[i]][route[i + 1]]
    return total


def two_opt_swap(route: list[Position], i: int, j: int) -> list[Position]:
    return route[:i] + route[i:j+1][::-1] + route[j+1:]


def two_opt_gain(pacman_start: Position, route: list[Position], i: int, j: int, matrix: dict[Position, dict[Position, int]]) -> int:
    # Determine the node before index i
    before_i = pacman_start if i == 0 else route[i - 1]
    # Determine the node after index j
    after_j = route[j + 1] if j + 1 < len(route) else None

    old_cost = matrix[before_i][route[i]]
    if after_j is not None:
        old_cost += matrix[route[j]][after_j]

    new_cost = matrix[before_i][route[j]]
    if after_j is not None:
        new_cost += matrix[route[i]][after_j]

    return old_cost - new_cost  # positive means improvement


def two_opt(pacman_start: Position, route: list[Position], matrix: dict[Position, dict[Position, int]]) -> list[Position]:
    route = list(route)
    improved = True

    while improved:
        improved = False
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                if two_opt_gain(pacman_start, route, i, j, matrix) > 0:
                    route = two_opt_swap(route, i, j)
                    improved = True

    return route

# ---------------------END OF SHORTEST CLEAR CODE---------------------------------------------------------


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
        for neighbor in wrap_neighbors(grid, pos):
            if neighbor not in distances:
                distances[neighbor] = distances[pos] + 1
                queue.append(neighbor)
    return distances


def safe_wrap_neighbors(grid: list[list[str]], pos: Position) -> list[Position]:
    return [
        neighbor
        for neighbor in wrap_neighbors(grid, pos)
        if grid[neighbor[1]][neighbor[0]] != SPIKE
    ]


def bfs_wrap_avoiding_spikes(
    grid: list[list[str]],
    start: Position,
) -> dict[Position, int]:
    if grid[start[1]][start[0]] in {WALL, SPIKE}:
        return {}

    queue: deque[Position] = deque([start])
    distances = {start: 0}
    while queue:
        pos = queue.popleft()
        for neighbor in safe_wrap_neighbors(grid, pos):
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
    distances = bfs_wrap_avoiding_spikes(grid, pacman_start)
    open_cells = _safe_walkable_cells(grid)

    # Ensure all open cells are reachable from the start with gameplay wrap-around
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
    if len(safe_wrap_neighbors(grid, pacman_start)) < 2:
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

    adjacent_open_cells = [
        neighbor
        for neighbor in neighbors(grid, ghost_respawn_location)
        if grid[neighbor[1]][neighbor[0]] != SPIKE
    ]
    if not adjacent_open_cells:
        print("Validation failed: Ghost respawn location is not next to open space.")
        return False

    if min(distances.get(cell, 0) for cell in adjacent_open_cells) < config.min_start_distance:
        print("Validation failed: Ghost respawn location is too close to Pacman.")
        return False

    # Maintain minimum junctions for maze complexity
    junction_count = sum(1 for cell in open_cells if len(
        safe_wrap_neighbors(grid, cell)) >= 3)
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
    stack: list[tuple[int, int]] = [start]
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
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
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


def _safe_walkable_cells(grid: list[list[str]]) -> list[Position]:
    return [
        (x, y)
        for y, row in enumerate(grid)
        for x, tile in enumerate(row)
        if tile not in {WALL, SPIKE}
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
    top_candidates = candidates[: min(8, len(candidates))]
    random.shuffle(top_candidates)
    candidates = top_candidates + candidates[min(8, len(candidates)):]

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
        walkable = _safe_walkable_cells(grid)
        reachable = (
            bfs_wrap_avoiding_spikes(grid, walkable[0])
            if walkable
            else {}
        )
        grid[y][x] = PELLET
        return not walkable or any(cell not in reachable for cell in walkable)

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

            nearest_distance = min(distances.get(cell, 0)
                                   for cell in adjacent_open_cells)
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

    width, height = len(grid[0]), len(grid)
    cx, cy = width // 2, height // 2

    middle_row_pellets = []
    for pos in distances:
        x, y = pos
        is_pellet = grid[y][x] == PELLET
        is_in_middle_rows = abs(y - cy) <= 1
        if is_pellet and is_in_middle_rows:
            middle_row_pellets.append(pos)

    if not middle_row_pellets:
        return

    def manhattan_distance_to_center(pos):
        return abs(pos[0] - cx) + abs(pos[1] - cy)

    center_pellet = min(middle_row_pellets, key=manhattan_distance_to_center)

    grid[center_pellet[1]][center_pellet[0]] = POWER

    pellet_cells = [pos for pos in distances if grid[pos[1]][pos[0]] == PELLET]

    for i in range(count):
        angle = (2 * math.pi * i) / count
        dx, dy = math.cos(angle), math.sin(angle)

        best = max(
            pellet_cells,
            key=lambda pos: (pos[0] - cx) * dx + (pos[1] - cy) * dy,
        )
        grid[best[1]][best[0]] = POWER


def _make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _average_corridor_length(grid: list[list[str]], open_cells: list[Position]) -> float:
    corridor_cells = [cell for cell in open_cells if len(
        safe_wrap_neighbors(grid, cell)) == 2]
    if not corridor_cells:
        return 0.0
    visited: set[Position] = set()
    lengths: list[int] = []

    for start in corridor_cells:
        if start in visited:
            continue

        length = 0
        stack = [start]
        while stack:
            cell = stack.pop()
            if cell in visited or len(safe_wrap_neighbors(grid, cell)) != 2:
                continue
            visited.add(cell)
            length += 1
            stack.extend(neighbor for neighbor in safe_wrap_neighbors(
                grid, cell) if neighbor not in visited)

        if length:
            lengths.append(length)

    return sum(lengths) / max(1, len(lengths))


def _estimate_collection_distance(
    grid: list[list[str]],
    start: Position,
    pellets: list[Position],
) -> int:
    if not pellets:
        return 0

    remaining = set(pellets)
    current = start
    total = 0

    while remaining:
        distances = bfs_wrap_avoiding_spikes(grid, current)
        nearest = min(
            remaining, key=lambda pellet: distances.get(pellet, 10_000))
        distance = distances.get(nearest, 10_000)
        if distance >= 10_000:
            break
        total += distance
        current = nearest
        remaining.remove(nearest)

    return total


def _maze_difficulty_score(
    config: DifficultyConfig,
    dead_end_ratio: float,
    average_branching_factor: float,
    average_corridor_length: float,
    nearest_ghost_distance: int,
    estimated_collection_distance: int,
    pellet_count: int,
) -> float:
    agent_score = {"random": 0.3, "heuristic": 1.8,
                   "mcts": 2.7}.get(config.ghost_agent, 1.0)
    ghost_pressure = min(1.5, config.ghost_count * 0.35) + \
        (0.4 * max(0, config.ghost_speed - 1))
    dead_end_pressure = min(1.0, dead_end_ratio * 4.0)
    corridor_pressure = min(0.8, average_corridor_length / 12.0)
    branch_relief = min(0.8, max(0.0, average_branching_factor - 2.0) * 0.6)
    start_relief = min(0.8, nearest_ghost_distance / 20.0)
    collection_pressure = min(
        0.8, estimated_collection_distance / max(1, pellet_count * 8))
    power_relief = min(0.8, config.power_pellets * 0.12)

    raw_score = (
        agent_score
        + ghost_pressure
        + dead_end_pressure
        + corridor_pressure
        + collection_pressure
        - branch_relief
        - start_relief
        - power_relief
    )
    return max(0.0, min(5.0, raw_score))
