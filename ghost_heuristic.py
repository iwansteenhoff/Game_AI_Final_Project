"""Advanced Heuristic ghost controller with flanking, fleeing, and difficulty scaling."""

from __future__ import annotations
import random

def choose_action(state, ghost_id: int, difficulty: int = 4) -> str:
    legal_actions = state.legal_actions_for_ghost(ghost_id)
    
    valid_moves = [a for a in legal_actions if a != "STAY"]
    if not valid_moves:
        return "STAY"

    # Scaling difficulity: The Mistake Rate (Epsilon-Greedy)
    
    mistake_chance = 0.0    # 0% mistake rate
    if difficulty == 2:
        mistake_chance = 0.30   #30% mistake rate
    elif difficulty == 3:
        mistake_chance = 0.10    # 10% mistake rate

    if random.random() < mistake_chance:
        return random.choice(valid_moves)
    

    # power_pellet_active
    
    is_fleeing = getattr(state, "power_pellet_active", False)

    best_action = valid_moves[0]
    # we want the HIGHEST score. If chasing, we want the LOWEST score.
    best_score = float("-inf") if is_fleeing else float("inf")

    other_ghosts = [
        pos for i, pos in enumerate(state.ghost_positions) if i != ghost_id
    ]

    for action in valid_moves:
        next_pos = state.next_ghost_position(ghost_id, action)
        distance_to_pacman = state.shortest_path_distance(next_pos, state.pacman_pos)
        score = float(distance_to_pacman)

        # The Flanking/Dispersion Penalty (Only apply if chasing)
        if not is_fleeing and other_ghosts:
            closest_teammate_dist = min(
                abs(next_pos[0] - g[0]) + abs(next_pos[1] - g[1]) 
                for g in other_ghosts
            )
            if closest_teammate_dist < 4:
                score += 15.0  

        # Evaluate the best move based on whether we are fleeing or hunting
        if is_fleeing:
            # We want to MAXIMIZE distance from Pac-Man
            if score > best_score:
                best_score = score
                best_action = action
        else:
            # We want to MINIMIZE distance to Pac-Man (with flanking penalties)
            if score < best_score:
                best_score = score
                best_action = action



    return best_action