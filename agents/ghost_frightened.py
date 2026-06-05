"""Fleeing controller used while a ghost is frightened."""

from __future__ import annotations

import random


def choose_action(state, ghost_id: int) -> str:
    legal_actions = [
        action
        for action in state.legal_actions_for_ghost(ghost_id)
        if action != "STAY"
    ]
    if not legal_actions:
        return "STAY"

    distances = {
        action: state.shortest_path_distance(
            state.next_ghost_position(ghost_id, action),
            state.pacman_pos,
        )
        for action in legal_actions
    }
    furthest_distance = max(distances.values())
    furthest_actions = [
        action for action, distance in distances.items()
        if distance == furthest_distance
    ]
    return random.choice(furthest_actions)
