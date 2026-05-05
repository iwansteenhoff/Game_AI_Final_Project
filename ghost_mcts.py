"""Collaborative Predictive MCTS Ghost Controller.

Based on the IEEE IGIC/CEC Ms. Pac-Man vs Ghost Team Competition research.
This implementation uses a "Hammer and Anvil" multi-agent strategy combined 
with Flocking/Dispersion penalties to trap the player.
"""

from __future__ import annotations
import random
from game.pacman_env import ACTION_DELTAS

ROLLOUTS_PER_ACTION = 20
ROLLOUT_DEPTH = 12

def choose_action(state, ghost_id: int) -> str:
    legal_actions = state.legal_actions_for_ghost(ghost_id)
    if not legal_actions:
        return "STAY"

    # THE HAMMER: Ghost 0 uses aggressive rule-based chasing to flush the player out.
    if ghost_id == 0:
        return min(
            legal_actions,
            key=lambda a: state.shortest_path_distance(
                state.next_ghost_position(ghost_id, a), state.pacman_pos
            )
        )

    # THE ANVIL: All other ghosts use Collaborative MCTS to set traps.
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

    # Isolate the positions of all other teammate ghosts
    other_ghosts = [pos for i, pos in enumerate(state.ghost_positions) if i != ghost_id]

    if ghost_pos == pacman_pos:
        return 100.0

    for depth in range(ROLLOUT_DEPTH):
        # Pac-Man actively runs away in the simulation (predictive evasion)
        pacman_pos = _evasive_next_position(state, pacman_pos, ghost_pos)
        ghost_pos = _greedy_or_random_next_position(state, ghost_pos, pacman_pos)

        if ghost_pos == pacman_pos:
            return 80.0 - depth  # Reward faster traps

    # Evaluate the state at the end of the rollout
    distance_to_pacman = state.shortest_path_distance(ghost_pos, pacman_pos)
    score = -float(distance_to_pacman)

    # ACADEMIC UPGRADE: The Flocking/Dispersion Penalty
    # Penalize the MCTS score heavily if this ghost ends up too close to its teammates.
    # This prevents the "conga line" effect and forces encirclement.
    for teammate_pos in other_ghosts:
        dist_to_teammate = abs(ghost_pos[0] - teammate_pos[0]) + abs(ghost_pos[1] - teammate_pos[1])
        if dist_to_teammate < 3:
            score -= 15.0  

    return score


def _evasive_next_position(state, pacman_pos, ghost_pos):
    """Simulates a highly evasive Pac-Man attempting to maximize survival distance."""
    actions = _legal_actions_from_grid(state, pacman_pos)
    if not actions:
        return pacman_pos

    # 15% of the time, Pac-Man makes a mistake, keeping the search tree wide
    if random.random() < 0.15:
        return _next_position_from_grid(state, pacman_pos, random.choice(actions))

    # 85% of the time, Pac-Man plays perfectly defensively
    return max(
        (_next_position_from_grid(state, pacman_pos, action) for action in actions),
        key=lambda candidate: state.shortest_path_distance(candidate, ghost_pos),
    )


def _greedy_or_random_next_position(state, pos, target):
    """Simulates a ghost moving toward the target with slight randomization."""
    actions = _legal_actions_from_grid(state, pos)
    if not actions:
        return pos

    if random.random() < 0.20:
        return _next_position_from_grid(state, pos, random.choice(actions))

    return min(
        (_next_position_from_grid(state, pos, action) for action in actions),
        key=lambda candidate: state.shortest_path_distance(candidate, target),
    )


def _legal_actions_from_grid(state, pos):
    """Finds legal actions, strictly forbidding the STAY action to maintain momentum."""
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