import numpy as np

from splendor_gym.engine import initial_state, legal_moves, apply_action
from splendor_gym.engine.encode import TOTAL_ACTIONS
from splendor_gym.engine.state import COLOR_INDEX


def test_initial_legal_moves_nonempty():
	state = initial_state(seed=42)
	mask = legal_moves(state)
	assert len(mask) == TOTAL_ACTIONS
	assert any(mask), "At least one legal move at start"


def test_take3_or_take2_changes_bank_correctly():
	state = initial_state(seed=1)
	mask = legal_moves(state)
	action = next(i for i, v in enumerate(mask) if v == 1)
	prev_sum = sum(state.bank)
	next_state = apply_action(state, action)
	if action < 10:
		assert sum(next_state.bank) == prev_sum - 3
	elif 10 <= action < 15:
		assert sum(next_state.bank) == prev_sum - 2
	assert next_state.turn_count == state.turn_count + 1


def test_take2_requires_four_in_bank():
	state = initial_state(seed=0)
	mask = legal_moves(state)
	for ci in range(5):
		if state.bank[ci] < 4:
			assert mask[10 + ci] == 0


def test_token_limit_enforced_to_ten():
	state = initial_state(seed=0)
	p = state.players[state.to_play]
	# Give player many tokens to exceed limit
	p.tokens = [5, 5, 5, 5, 5, 0]
	next_state = apply_action(state, 0)  # any legal action style, will trigger end of turn
	assert sum(next_state.players[state.to_play - 1].tokens) <= 10 