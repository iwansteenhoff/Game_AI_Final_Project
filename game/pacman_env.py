"""Grid-based Pacman-like game environment."""

from __future__ import annotations

from dataclasses import dataclass

from game.maze_generator import PELLET, POWER, WALL, Position, shortest_path_distance

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


class PacmanEnv:
    def __init__(
        self,
        grid: list[list[str]],
        pacman_start: Position,
        ghost_starts: list[Position],
        max_steps: int = 700,
    ) -> None:
        self.grid = [row[:] for row in grid]
        self.pacman_start = pacman_start
        self.ghost_starts = ghost_starts[:]
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> None:
        self.pacman_pos = self.pacman_start
        self.ghost_positions = self.ghost_starts[:]
        self.steps = 0
        self.won = False
        self.lost = False
        self.pellets_collected = 0
        self.total_pellets = sum(tile in {PELLET, POWER} for row in self.grid for tile in row)

    def step(self, pacman_action: str, ghost_actions: list[str]) -> StepResult:
        if self.done:
            return StepResult(self.won, self.lost, False, False)

        pellet_collected = False
        power_collected = False
        self.steps += 1

        self.pacman_pos = self.next_position(self.pacman_pos, pacman_action)
        px, py = self.pacman_pos
        if self.grid[py][px] in {PELLET, POWER}:
            power_collected = self.grid[py][px] == POWER
            pellet_collected = True
            self.grid[py][px] = " "
            self.pellets_collected += 1

        if self.pacman_pos in self.ghost_positions:
            self.lost = True
            return StepResult(False, True, pellet_collected, power_collected)

        for index, action in enumerate(ghost_actions):
            if index < len(self.ghost_positions):
                self.ghost_positions[index] = self.next_position(self.ghost_positions[index], action)

        if self.pacman_pos in self.ghost_positions:
            self.lost = True
        elif self.pellets_collected >= self.total_pellets:
            self.won = True
        elif self.steps >= self.max_steps:
            self.lost = True

        return StepResult(self.won, self.lost, pellet_collected, power_collected)

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
        return nx, ny

    def shortest_path_distance(self, start: Position, goal: Position) -> int:
        return shortest_path_distance(self.grid, start, goal)
