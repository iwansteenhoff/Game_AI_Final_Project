"""Grid-based Pacman-like game environment."""

from __future__ import annotations

from dataclasses import dataclass

from game.maze_generator import PELLET, POWER, SPIKE, WALL, Position

ACTION_DELTAS: dict[str, Position] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "STAY": (0, 0),
}
DEFAULT_GHOST_RESPAWN_DELAY = 180


@dataclass
class StepResult:
    won: bool
    lost: bool
    pellet_collected: bool
    power_collected: bool
    ghosts_eaten: int = 0
    ghost_eaten: bool = False


@dataclass
class MotionState:
    direction: str = "STAY"
    next_direction: str = "STAY"
    position: Position = (0, 0)
    home_position: Position = (0, 0)
    move_from: Position = (0, 0)
    move_to: Position = (0, 0)
    move_progress: float = 0.0
    moving: bool = False
    just_reached_tile: bool = False
    respawn_timer: int = 0
    pixel_pos: list[float] | None = None


class PacmanEnv:
    def __init__(
        self,
        grid: list[list[str]],
        pacman_start: Position,
        ghost_starts: list[Position],
        ghost_respawn_location: Position,
        max_steps: int = 700,
        frighten_duration: int = 2400,
        ghost_respawn_delay: int = DEFAULT_GHOST_RESPAWN_DELAY,
        move_speed: float = 0.1,
        cell_size: int = 28,  # For pixel calculations
        maze_difficulty_score: float = 0.0,
        generated_greedy_solution: int = 2**32,
    ) -> None:
        self.initial_grid = [row[:] for row in grid]
        self.grid = [row[:] for row in self.initial_grid]
        self.pacman_start = pacman_start
        self.ghost_starts = ghost_starts[:]
        self.ghost_respawn_location = ghost_respawn_location
        self.max_steps = max_steps
        self.maze_difficulty_score = maze_difficulty_score
        self.frighten_duration = frighten_duration
        self.ghost_respawn_delay = max(1, ghost_respawn_delay)
        self.move_speed = move_speed
        self.cell_size = cell_size
        self.pacman_state = MotionState()
        self.ghost_states: list[MotionState] = []
        self.pacman_pos = pacman_start
        self.ghost_positions = ghost_starts[:]
        self.steps = 0
        self.won = False
        self.lost = False
        self.end_reason: str | None = None
        self.pellets_collected = 0
        self.total_pellets = 0
        self.ghost_frighten_timers: list[int] = []
        self.ghost_respawn_timers: list[int] = []
        self.pacman_pixel_pos: list[float] = [0.0, 0.0]
        self.ghost_dirs: list[str] = []
        self.ghost_next_dirs: list[str] = []
        self.ghost_pixel_pos: list[list[float]] = []
        self.pacman_dir = "STAY"
        self.pacman_next_dir = "STAY"
        self.generated_greedy_solution = generated_greedy_solution
        self.reset()

    def reset(self) -> None:
        self.grid = [row[:] for row in self.initial_grid]
        self.pacman_pos = self.pacman_start
        self.ghost_positions = self.ghost_starts[:]
        self.steps = 0
        self.traveled_steps = 0
        self.won = False
        self.lost = False
        self.end_reason = None
        self.pellets_collected = 0
        self.power_pellets_collected = 0
        self.ghosts_eaten = 0
        self.deaths_to_ghost = 0
        self.power_timer = 0
        self.total_pellets = sum(tile in {PELLET, POWER}
                                 for row in self.grid for tile in row)
        self.ghost_frighten_timers: list[int] = [0] * len(self.ghost_starts)
        self.ghost_respawn_timers = [0] * len(self.ghost_starts)

        self.pacman_state = MotionState(
            position=self.pacman_pos,
            home_position=self.pacman_pos,
            move_from=self.pacman_pos,
            move_to=self.pacman_pos,
            pixel_pos=[
                float(self.pacman_pos[0] *
                      self.cell_size + self.cell_size // 2),
                float(self.pacman_pos[1] *
                      self.cell_size + self.cell_size // 2),
            ],
        )
        self.ghost_states = [
            MotionState(
                position=ghost_pos,
                home_position=self.ghost_respawn_location,
                move_from=ghost_pos,
                move_to=ghost_pos,
                pixel_pos=[
                    float(ghost_pos[0] * self.cell_size + self.cell_size // 2),
                    float(ghost_pos[1] * self.cell_size + self.cell_size // 2),
                ],
            )
            for ghost_pos in self.ghost_positions
        ]
        self._sync_motion_attributes()

    def _sync_motion_attributes(self) -> None:
        self.pacman_dir = self.pacman_state.direction
        self.pacman_next_dir = self.pacman_state.next_direction
        self.just_reached_tile = self.pacman_state.just_reached_tile
        self.pacman_pixel_pos = self.pacman_state.pixel_pos or [0.0, 0.0]
        self.ghost_respawn_timers = [
            state.respawn_timer for state in self.ghost_states]
        self.ghost_dirs = [state.direction for state in self.ghost_states]
        self.ghost_next_dirs = [
            state.next_direction for state in self.ghost_states]
        self.ghost_pixel_pos = [state.pixel_pos or [0.0, 0.0]
                                for state in self.ghost_states]

    def _advance_motion_state(
        self,
        state: MotionState,
        speed: float,
        allow_respawn_tile: bool,
    ) -> MotionState:

        state.just_reached_tile = False

        if state.respawn_timer > 0:
            state.respawn_timer -= 1
            state.direction = "STAY"
            state.next_direction = "STAY"
            state.moving = False
            state.move_progress = 0.0
            state.position = state.home_position
            state.move_from = state.home_position
            state.move_to = state.home_position
            state.pixel_pos = [
                state.home_position[0] * self.cell_size + self.cell_size // 2,
                state.home_position[1] * self.cell_size + self.cell_size // 2,
            ]
            if state.respawn_timer > 0:
                return state

        if not state.moving:
            x, y = state.position
            if state.next_direction != state.direction and state.next_direction in self.legal_actions_from((x, y), allow_respawn_tile):
                state.direction = state.next_direction

            if state.direction != "STAY":
                target = self.next_position(
                    (x, y), state.direction, allow_respawn_tile)
                if target != (x, y):
                    state.move_from = (x, y)
                    state.move_to = target
                    state.move_progress = 0.0
                    state.moving = True
                else:
                    state.direction = "STAY"

        if state.moving:
            state.move_progress += speed
            if state.move_progress >= 1.0:
                state.position = state.move_to
                state.move_from = state.position
                state.move_to = state.position
                state.move_progress = 0.0
                state.moving = False
                state.just_reached_tile = True
            else:
                state.just_reached_tile = False

        fx, fy = state.move_from
        tx, ty = state.move_to
        t = state.move_progress

        # A portal move crosses one cell beyond the board edge. Interpolating
        # directly between wrapped tile indices would sweep across the board.
        dx = tx - fx
        dy = ty - fy
        if abs(dx) > 1:
            dx = -1 if tx > fx else 1
        if abs(dy) > 1:
            dy = -1 if ty > fy else 1

        interp_x = fx + dx * t
        interp_y = fy + dy * t
        state.pixel_pos = [
            float(interp_x * self.cell_size + self.cell_size // 2),
            float(interp_y * self.cell_size + self.cell_size // 2),
        ]
        return state

    def move_pacman_smooth(self, speed: float) -> None:
        self.pacman_state.next_direction = self.pacman_next_dir
        self.pacman_state = self._advance_motion_state(
            self.pacman_state, speed, False)
        self.pacman_pos = self.pacman_state.position
        self._sync_motion_attributes()

    def move_ghosts_smooth(self, ghost_actions: list[str], speed: float) -> None:
        for index, action in enumerate(ghost_actions):
            if index < len(self.ghost_states):
                self.ghost_states[index].next_direction = action
                self.ghost_states[index] = self._advance_motion_state(
                    self.ghost_states[index], speed, True)

        self.ghost_positions = [state.position for state in self.ghost_states]
        self._sync_motion_attributes()

    @property
    def frightened_ghosts(self) -> set[int]:
        return {i for i, t in enumerate(self.ghost_frighten_timers) if t > 0}

    def step(self, pacman_action: str, ghost_actions: list[str]) -> StepResult:
        if self.done:
            return StepResult(self.won, self.lost, False, False)

        old_pacman_pos = self.pacman_pos
        pellet_collected = False
        power_collected = False
        ghosts_eaten = 0
        ghost_eaten = False
        self.steps += 1
        if self.power_timer > 0:
            self.power_timer -= 1

        self.pacman_next_dir = pacman_action
        self.move_pacman_smooth(self.move_speed)
        self.move_ghosts_smooth(ghost_actions, self.move_speed)
        self.ghost_frighten_timers = [max(0, t - 1)
                                      for t in self.ghost_frighten_timers]

        if self.pacman_state.just_reached_tile:
            px, py = self.pacman_pos

            if self.grid[py][px] == SPIKE:
                self.lost = True
                self.end_reason = "Hit a spike"
                return StepResult(False, True, pellet_collected, power_collected, ghosts_eaten)

            if self.grid[py][px] in {PELLET, POWER}:
                power_collected = self.grid[py][px] == POWER
                pellet_collected = True
                self.grid[py][px] = " "
                self.pellets_collected += 1

                if power_collected:
                    self.power_pellets_collected += 1
                    self.power_timer = self.frighten_duration
                    self.ghost_frighten_timers = [
                        self.frighten_duration if pos is not None else 0
                        for pos in self.ghost_positions
                    ]

        ghost_eaten = self._resolve_ghost_collision()
        if self.lost:
            self.deaths_to_ghost += 1
            return StepResult(False, True, pellet_collected, power_collected, ghosts_eaten, ghost_eaten)

        if old_pacman_pos != self.pacman_pos:
            self.traveled_steps += 1

        if self.pellets_collected >= self.total_pellets:
            self.won = True
            self.end_reason = "All pellets collected"
        elif self.steps >= self.max_steps:
            self.lost = True
            self.end_reason = "Time limit reached"

        return StepResult(self.won, self.lost, pellet_collected, power_collected, ghosts_eaten, ghost_eaten)

    def _send_ghost_home(self, ghost_id: int) -> None:
        state = self.ghost_states[ghost_id]
        state.respawn_timer = self.ghost_respawn_delay
        state.direction = "STAY"
        state.next_direction = "STAY"
        state.moving = False
        state.move_progress = 0.0
        state.position = self.ghost_respawn_location
        state.home_position = self.ghost_respawn_location
        state.move_from = self.ghost_respawn_location
        state.move_to = self.ghost_respawn_location
        state.pixel_pos = [
            float(self.ghost_respawn_location[0] *
                  self.cell_size + self.cell_size // 2),
            float(self.ghost_respawn_location[1] *
                  self.cell_size + self.cell_size // 2),
        ]
        self.ghost_frighten_timers[ghost_id] = 0
        self.ghost_positions[ghost_id] = state.position
        self.ghost_respawn_timers[ghost_id] = state.respawn_timer

    @property
    def done(self) -> bool:
        return self.won or self.lost

    def legal_actions_for_pacman(self) -> list[str]:
        return self.legal_actions_from(self.pacman_pos, False)

    def legal_actions_for_ghost(self, ghost_id: int) -> list[str]:
        return self.legal_actions_from(self.ghost_positions[ghost_id], True)

    @property
    def just_reached_tile(self) -> bool:
        return self.pacman_state.just_reached_tile

    @just_reached_tile.setter
    def just_reached_tile(self, value: bool) -> None:
        self.pacman_state.just_reached_tile = value

    @property
    def ghost_just_reached_tiles(self) -> list[bool]:
        return [state.just_reached_tile for state in self.ghost_states]

    def legal_actions_from(self, pos: Position, allow_respawn_tile: bool) -> list[str]:
        return [
            action
            for action in ACTION_DELTAS
            if action == "STAY" or self.next_position(pos, action, allow_respawn_tile) != pos
        ]

    def next_ghost_position(self, ghost_id: int, action: str) -> Position:
        return self.next_position(self.ghost_positions[ghost_id], action, True)

    def next_position(self, pos: Position, action: str, allow_respawn_tile: bool = False) -> Position:
        dx, dy = ACTION_DELTAS.get(action, (0, 0))
        width = len(self.grid[0])
        height = len(self.grid)
        nx = (pos[0] + dx) % width
        ny = (pos[1] + dy) % height
        if self.grid[ny][nx] == WALL and not (allow_respawn_tile and (nx, ny) == self.ghost_respawn_location):
            return pos
        # TODO: Keep track of total steps in current level
        return nx, ny

    def shortest_path_distance(self, start: Position, goal: Position) -> int:
        return self._shortest_path_distance_for_ghosts(start, goal)

    def _shortest_path_distance_for_ghosts(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0

        queue: list[Position] = [start]
        distances = {start: 0}

        while queue:
            x, y = queue.pop(0)
            for dx, dy in ACTION_DELTAS.values():
                if dx == 0 and dy == 0:
                    continue

                nx = (x + dx) % len(self.grid[0])
                ny = (y + dy) % len(self.grid)
                next_pos = (nx, ny)

                if next_pos in distances:
                    continue

                if self.grid[ny][nx] == WALL and next_pos != self.ghost_respawn_location:
                    continue

                distances[next_pos] = distances[(x, y)] + 1
                if next_pos == goal:
                    return distances[next_pos]
                queue.append(next_pos)

        return 10_000

    @property
    def powered(self) -> bool:
        return self.power_timer > 0

    def _resolve_ghost_collision(self) -> bool:
        colliding_ghosts = [
            index
            for index, state in enumerate(self.ghost_states)
            if state.respawn_timer <= 0 and self._pixel_distance(
                self.pacman_state.pixel_pos,
                state.pixel_pos,
            ) <= self.cell_size * 0.55
        ]
        if not colliding_ghosts:
            return False

        if not self.powered:
            self.lost = True
            self.end_reason = "Caught by a ghost"
            return False

        eaten_any = False
        for index in colliding_ghosts:
            self._send_ghost_home(index)
            self.ghosts_eaten += 1
            eaten_any = True
        return eaten_any

    def _pixel_distance(
        self,
        first: list[float] | None,
        second: list[float] | None,
    ) -> float:
        if first is None or second is None:
            return float("inf")

        board_width = len(self.grid[0]) * self.cell_size
        board_height = len(self.grid) * self.cell_size
        dx = abs(first[0] - second[0]) % board_width
        dy = abs(first[1] - second[1]) % board_height
        dx = min(dx, board_width - dx)
        dy = min(dy, board_height - dy)
        return (dx * dx + dy * dy) ** 0.5
