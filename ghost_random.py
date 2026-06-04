from __future__ import annotations
import random

# Memory to track where the ghost just came from
_last_positions: dict[int, tuple[int, int]] = {}

def choose_action(state, ghost_id: int) -> str:
    current_pos = state.ghost_positions[ghost_id]
    legal_actions = state.legal_actions_for_ghost(ghost_id)
    
    # 1. Never stand still if we can help it
    valid_moves = [a for a in legal_actions if a != "STAY"]
    
    # 2. THE ANTI-JITTER RULE
    # The ghost cannot reverse its direction unless it hits a dead end
    previous_pos = _last_positions.get(ghost_id)
    if previous_pos and len(valid_moves) > 1:
        forward_moves = [
            a for a in valid_moves 
            if state.next_ghost_position(ghost_id, a) != previous_pos
        ]
        # Only apply the filter if we actually have forward moves left
        if forward_moves:
            valid_moves = forward_moves

    # Fallback if completely trapped
    if not valid_moves:
        _last_positions[ghost_id] = current_pos
        return "STAY"

    # 3. Pick a completely random direction from the remaining forward paths
    chosen_action = random.choice(valid_moves)
    
    # Save memory for next turn
    _last_positions[ghost_id] = current_pos
    return chosen_action
