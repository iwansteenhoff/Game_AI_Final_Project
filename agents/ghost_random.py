"""Random ghost controller."""

from __future__ import annotations

import random


def choose_action(state, ghost_id: int) -> str:
    actions = state.legal_actions_for_ghost(ghost_id)
    return random.choice(actions) if actions else "STAY"
