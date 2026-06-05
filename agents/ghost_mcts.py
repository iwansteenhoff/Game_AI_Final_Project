"""Small rollout-based ghost controller for the hardest difficulty.

This is intentionally lightweight rather than a full academic MCTS
implementation. It samples short future trajectories for each legal first move
and chooses the move with the best expected chase outcome.
"""

from __future__ import annotations

import random

from game.pacman_env import ACTION_DELTAS

ROLLOUTS_PER_ACTION = 18
ROLLOUT_DEPTH = 8


def choose_action(state, ghost_id: int, difficulty: int | None = None) -> str:
    legal_actions = state.legal_actions_for_ghost(ghost_id)
    if not legal_actions:
        return "STAY"

    best_action = "STAY"
    best_score = float("-inf")

    for action in legal_actions:
        total = 0.0
        for _ in range(ROLLOUTS_PER_ACTION):
            total += _rollout_score(state, ghost_id, action)
        average = total / ROLLOUTS_PER_ACTION

        if average > best_score:
            best_score = average
            best_action = action

    return best_action


def _rollout_score(state, ghost_id: int, first_action: str) -> float:
    ghost_pos = state.next_ghost_position(ghost_id, first_action)
    pacman_pos = state.pacman_pos

    if ghost_pos == pacman_pos:
        return 100.0

    for depth in range(ROLLOUT_DEPTH):
        pacman_pos = _random_next_position(state, pacman_pos)
        ghost_pos = _greedy_or_random_next_position(state, ghost_pos, pacman_pos)

        if ghost_pos == pacman_pos:
            return 80.0 - depth

    distance = state.shortest_path_distance(ghost_pos, pacman_pos)
    return -float(distance)


def _random_next_position(state, pos):
    actions = _legal_actions_from_grid(state, pos)
    action = random.choice(actions) if actions else "STAY"
    return _next_position_from_grid(state, pos, action)


def _greedy_or_random_next_position(state, pos, target):
    actions = _legal_actions_from_grid(state, pos)
    if not actions:
        return pos

    if random.random() < 0.25:
        return _next_position_from_grid(state, pos, random.choice(actions))

    return min(
        (_next_position_from_grid(state, pos, action) for action in actions),
        key=lambda candidate: state.shortest_path_distance(candidate, target),
    )


def _legal_actions_from_grid(state, pos):
    return [
        action
        for action in ACTION_DELTAS
        if action == "STAY" or _next_position_from_grid(state, pos, action) != pos
    ]


def _next_position_from_grid(state, pos, action):
    dx, dy = ACTION_DELTAS.get(action, (0, 0))
    width = len(state.grid[0])
    height = len(state.grid)
    nx = (pos[0] + dx) % width
    ny = (pos[1] + dy) % height
    respawn_location = getattr(state, "ghost_respawn_location", None)
    if state.grid[ny][nx] == "#" and (nx, ny) != respawn_location:
        return pos
    return nx, ny
