"""Shortest-path chasing ghost controller."""

from __future__ import annotations


def choose_action(state, ghost_id: int, difficulty: int | None = None) -> str:
    best_action = "STAY"
    best_distance = float("inf")

    for action in state.legal_actions_for_ghost(ghost_id):
        next_pos = state.next_ghost_position(ghost_id, action)
        distance = state.shortest_path_distance(next_pos, state.pacman_pos)

        if distance < best_distance:
            best_distance = distance
            best_action = action

    return best_action
