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
    ghosts_eaten: int = 0


class PacmanEnv:
    def __init__(
        self,
        grid: list[list[str]],
        pacman_start: Position,
        ghost_starts: list[Position],
        max_steps: int = 700,
        frighten_duration: int = 20,
        generated_greedy_solution: int = 2**32,
    ) -> None:
        self.grid = [row[:] for row in grid]
        self.pacman_start = pacman_start
        self.ghost_starts = ghost_starts[:]
        self.max_steps = max_steps
        self.frighten_duration = frighten_duration
        self.generated_greedy_solution = generated_greedy_solution
        self.reset()

    def reset(self) -> None:
        self.pacman_pos = self.pacman_start
        self.ghost_positions = self.ghost_starts[:]
        self.steps = 0
        self.traveled_steps = 0
        self.won = False
        self.lost = False
        self.pellets_collected = 0
        self.total_pellets = sum(tile in {PELLET, POWER}
                                 for row in self.grid for tile in row)

        self.ghost_frighten_timers: list[int] = [0] * len(self.ghost_starts)

    @property
    def frightened_ghosts(self) -> set[int]:
        return {i for i, t in enumerate(self.ghost_frighten_timers) if t > 0}

    def step(self, pacman_action: str, ghost_actions: list[str]) -> StepResult:
        if self.done:
            return StepResult(self.won, self.lost, False, False)

        pellet_collected = False
        power_collected = False
        ghosts_eaten = 0
        self.steps += 1

        self.ghost_frighten_timers = [max(0, t - 1)
                                      for t in self.ghost_frighten_timers]

        old_x, old_y = self.pacman_pos
        self.pacman_pos = self.next_position(self.pacman_pos, pacman_action)
        px, py = self.pacman_pos

        if self.grid[py][px] == SPIKE:
            self.lost = True
            return StepResult(False, True, pellet_collected, power_collected, ghosts_eaten)

        if self.grid[py][px] in {PELLET, POWER}:
            power_collected = self.grid[py][px] == POWER
            pellet_collected = True
            self.grid[py][px] = " "
            self.pellets_collected += 1
            if power_collected:
                self.ghost_frighten_timers = [
                    self.frighten_duration if pos is not None else 0
                    for pos in self.ghost_positions
                ]

        ghosts_eaten += self._resolve_pacman_ghost_collisions()
        if self.lost:
            return StepResult(False, True, pellet_collected, power_collected, ghosts_eaten)

        for index, action in enumerate(ghost_actions):
            if index < len(self.ghost_positions):
                self.ghost_positions[index] = self.next_position(
                    self.ghost_positions[index], action
                )

        ghosts_eaten += self._resolve_pacman_ghost_collisions()
        if self.lost:
            return StepResult(False, True, pellet_collected, power_collected, ghosts_eaten)

        if (old_x, old_y) != self.pacman_pos:
            self.traveled_steps += 1

        if self.pellets_collected >= self.total_pellets:
            self.won = True
        elif self.steps >= self.max_steps:
            self.lost = True

        return StepResult(self.won, self.lost, pellet_collected, power_collected, ghosts_eaten)

    def _resolve_pacman_ghost_collisions(self) -> int:
        eaten = 0
        indices_to_remove = []
        for i, ghost_pos in enumerate(self.ghost_positions):
            if ghost_pos == self.pacman_pos:
                if self.ghost_frighten_timers[i] > 0:
                    indices_to_remove.append(i)
                    eaten += 1
                else:
                    self.lost = True
                    return eaten

        # TODO: Handle what happens when ghosts are eaten (respawn? removed from game?)
        # For now, we just remove them from the game.
        for i in reversed(indices_to_remove):
            self.ghost_positions.pop(i)
            self.ghost_frighten_timers.pop(i)
        return eaten

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
