"""Collaborative Predictive MCTS Ghost Controller.

Based on the IEEE IGIC/CEC Ms. Pac-Man vs Ghost Team Competition research.
Includes Hammer & Anvil strategy, Flocking penalties, Power Pellet evasion,
and Computational Budget Scaling.
"""

from __future__ import annotations
import random
from game.pacman_env import ACTION_DELTAS

def choose_action(state, ghost_id: int, difficulty: int = 5) -> str:
    legal_actions = state.legal_actions_for_ghost(ghost_id)
    if not legal_actions:
        return "STAY"

    # --- SCALING DIFFICULTY: The Computational Budget ---
    # Easy: Short-sighted and shallow simulations
    # Hard: Deep, elaborate trap-setting
    if difficulty <= 3:
        rollouts_per_action = 5
        rollout_depth = 4
    elif difficulty == 4:
        rollouts_per_action = 10
        rollout_depth = 8
    else:  # Level 5+ (The Boss)
        rollouts_per_action = 20
        rollout_depth = 12
    # ----------------------------------------------------

    is_fleeing = getattr(state, "power_pellet_active", False)

    # THE HAMMER: Ghost 0 uses rule-based chasing (or fleeing)
    if ghost_id == 0:
        if is_fleeing:
            return max(
                legal_actions,
                key=lambda a: state.shortest_path_distance(
                    state.next_ghost_position(ghost_id, a), state.pacman_pos
                )
            )
        else:
            return min(
                legal_actions,
                key=lambda a: state.shortest_path_distance(
                    state.next_ghost_position(ghost_id, a), state.pacman_pos
                )
            )

    # THE ANVIL: All other ghosts use Collaborative MCTS
    best_action = "STAY"
    best_score = float("-inf")

    for action in legal_actions:
        total = 0.0
        for _ in range(rollouts_per_action):
            # Pass the dynamic rollout_depth into the scoring function
            total += _rollout_score(state, ghost_id, action, is_fleeing, rollout_depth)
        average = total / rollouts_per_action

        if average > best_score:
            best_score = average
            best_action = action

    return best_action


def _rollout_score(state, ghost_id: int, first_action: str, is_fleeing: bool, rollout_depth: int) -> float:
    ghost_pos = state.next_ghost_position(ghost_id, first_action)
    pacman_pos = state.pacman_pos

    other_ghosts = [pos for i, pos in enumerate(state.ghost_positions) if i != ghost_id]

    if ghost_pos == pacman_pos:
        return -1000.0 if is_fleeing else 100.0

    # Use the dynamic depth here!
    for depth in range(rollout_depth):
        pacman_pos = _evasive_next_position(state, pacman_pos, ghost_pos)
        ghost_pos = _greedy_or_random_next_position(state, ghost_pos, pacman_pos, is_fleeing)

        if ghost_pos == pacman_pos:
            return -800.0 + depth if is_fleeing else 80.0 - depth

    distance_to_pacman = state.shortest_path_distance(ghost_pos, pacman_pos)
    score = float(distance_to_pacman) if is_fleeing else -float(distance_to_pacman)

    if not is_fleeing:
        for teammate_pos in other_ghosts:
            dist_to_teammate = abs(ghost_pos[0] - teammate_pos[0]) + abs(ghost_pos[1] - teammate_pos[1])
            if dist_to_teammate < 3:
                score -= 15.0  

    return score


def _evasive_next_position(state, pacman_pos, ghost_pos):
    actions = _legal_actions_from_grid(state, pacman_pos)
    if not actions:
        return pacman_pos

    if random.random() < 0.15:
        return _next_position_from_grid(state, pacman_pos, random.choice(actions))

    return max(
        (_next_position_from_grid(state, pacman_pos, action) for action in actions),
        key=lambda candidate: state.shortest_path_distance(candidate, ghost_pos),
    )


def _greedy_or_random_next_position(state, pos, target, is_fleeing):
    actions = _legal_actions_from_grid(state, pos)
    if not actions:
        return pos

    if random.random() < 0.20:
        return _next_position_from_grid(state, pos, random.choice(actions))

    if is_fleeing:
        return max(
            (_next_position_from_grid(state, pos, action) for action in actions),
            key=lambda candidate: state.shortest_path_distance(candidate, target),
        )
    else:
        return min(
            (_next_position_from_grid(state, pos, action) for action in actions),
            key=lambda candidate: state.shortest_path_distance(candidate, target),
        )


def _legal_actions_from_grid(state, pos):
    return [
        action
        for action in ACTION_DELTAS
        if action != "STAY" and _next_position_from_grid(state, pos, action) != pos
    ]


def _next_position_from_grid(state, pos, action):
    dx, dy = ACTION_DELTAS.get(action, (0, 0))
    nx, ny = pos[0] + dx, pos[1] + dy
    if state.grid[ny][nx] == "#":
        return pos
    return nx, ny