"""Grid-based Pacman-like game environment."""

from __future__ import annotations

from dataclasses import dataclass

from game.maze_generator import PELLET, POWER, SPIKE, WALL, Position, shortest_path_distance

ACTION_DELTAS: dict[str, Position] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "STAY": (0, 0),
}


@dataclass
class StepResult:
    won: bool
    lost: bool
    pellet_collected: bool
    power_collected: bool
    ghost_eaten: bool = False


class PacmanEnv:
    def __init__(
        self,
        grid: list[list[str]],
        pacman_start: Position,
        ghost_starts: list[Position],
        max_steps: int = 700,
        maze_difficulty_score: float = 0.0,
        frighten_duration: int = 20,
        generated_greedy_solution: int = 2**32,
    ) -> None:
        self.initial_grid = [row[:] for row in grid]
        self.grid = [row[:] for row in self.initial_grid]
        self.pacman_start = pacman_start
        self.ghost_starts = ghost_starts[:]
        self.max_steps = max_steps
        self.maze_difficulty_score = maze_difficulty_score
        self.frighten_duration = frighten_duration
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
        self.pellets_collected = 0
        self.power_pellets_collected = 0
        self.ghosts_eaten = False
        self.deaths_to_ghost = 0
        self.power_timer = 0
        self.total_pellets = sum(tile in {PELLET, POWER} for row in self.grid for tile in row)

    def step(self, pacman_action: str, ghost_actions: list[str]) -> StepResult:
        if self.done:
            return StepResult(self.won, self.lost, False, False, False)
        old_x, old_y = self.pacman_pos
        pellet_collected = False
        power_collected = False
        ghost_eaten = False
        self.steps += 1
        if self.power_timer > 0:
            self.power_timer -= 1

        self.ghost_frighten_timers = [max(0, t - 1)
                                      for t in self.ghost_frighten_timers]

        self.pacman_pos = self.next_position(self.pacman_pos, pacman_action)
        px, py = self.pacman_pos

        if self.grid[py][px] == SPIKE:
            self.lost = True
            return StepResult(False, True, pellet_collected, power_collected, ghost_eaten)

        if self.grid[py][px] in {PELLET, POWER}:
            power_collected = self.grid[py][px] == POWER
            pellet_collected = True
            self.grid[py][px] = " "
            self.pellets_collected += 1
            if power_collected:
                self.power_pellets_collected += 1
                self.power_timer = 30

        ghost_eaten = self._resolve_ghost_collision()
        if self.lost:
            self.deaths_to_ghost += 1
            return StepResult(False, True, pellet_collected, power_collected, ghost_eaten)

        for index, action in enumerate(ghost_actions):
            if index < len(self.ghost_positions):
                self.ghost_positions[index] = self.next_position(
                    self.ghost_positions[index], action
                )

        ghost_eaten = self._resolve_ghost_collision() or ghost_eaten
        if self.lost:
            self.deaths_to_ghost += 1
        elif self.pellets_collected >= self.total_pellets:
            self.won = True
        elif self.steps >= self.max_steps:
            self.lost = True
        if (old_x, old_y) != self.pacman_pos:
            self.traveled_steps += 1

        return StepResult(self.won, self.lost, pellet_collected, power_collected, ghost_eaten)

    @property
    def done(self) -> bool:
        return self.won or self.lost

    def legal_actions_for_pacman(self) -> list[str]:
        return self.legal_actions_from(self.pacman_pos)

    def legal_actions_for_ghost(self, ghost_id: int) -> list[str]:
        return self.legal_actions_from(self.ghost_positions[ghost_id])

    def legal_actions_from(self, pos: Position) -> list[str]:
        return [
            action
            for action in ACTION_DELTAS
            if action == "STAY" or self.next_position(pos, action) != pos
        ]

    def next_ghost_position(self, ghost_id: int, action: str) -> Position:
        return self.next_position(self.ghost_positions[ghost_id], action)

    def next_position(self, pos: Position, action: str) -> Position:
        dx, dy = ACTION_DELTAS.get(action, (0, 0))
        nx, ny = pos[0] + dx, pos[1] + dy
        if self.grid[ny][nx] == WALL:
            return pos
        ###### TODO: Keep track of total steps in current level
        return nx, ny

    def shortest_path_distance(self, start: Position, goal: Position) -> int:
        return shortest_path_distance(self.grid, start, goal)

    @property
    def powered(self) -> bool:
        return self.power_timer > 0

    def _resolve_ghost_collision(self) -> bool:
        if self.pacman_pos not in self.ghost_positions:
            return False

        if not self.powered:
            self.lost = True
            return False

        eaten_any = False
        for index, ghost_pos in enumerate(self.ghost_positions):
            if ghost_pos == self.pacman_pos:
                self.ghost_positions[index] = self.ghost_starts[index]
                self.ghosts_eaten += 1
                eaten_any = True
        return eaten_any
