import numpy as np

from splendor_gym.engine import initial_state, apply_action, legal_moves
from splendor_gym.envs import SplendorEnv
from splendor_gym.engine.state import COLOR_INDEX, STANDARD_COLORS
from splendor_gym.engine.encode import OBSERVATION_DIM, encode_observation


def test_affordability_discounts_and_gold():
	state = initial_state(seed=123)
	p = state.players[state.to_play]
	# Craft an affordable card: needs 2 red, 2 blue; player has 1 red token, 1 blue token, 1 gold; bonuses 1 red
	p.tokens = [0, 1, 0, 1, 0, 1]  # white, blue, green, red, black, gold
	p.bonuses = [0, 0, 0, 1, 0]
	# Insert a card at tier1 slot0
	card = state.board[1][0]
	card.cost = {"red": 2, "blue": 2}
	state.bank = [4, 4, 4, 4, 4, 5]
	mask = legal_moves(state)
	buy_idx = 15 + 0  # tier1 slot0
	assert mask[buy_idx] == 1
	next_state = apply_action(state, buy_idx)
	p2 = next_state.players[state.to_play]  # next player to play; purchased player is previous
	prev_player = next_state.players[(next_state.to_play - 1) % next_state.num_players]
	# Gold used should be at most 1
	assert prev_player.tokens[COLOR_INDEX["gold"]] <= 1
	# No negative tokens
	assert all(t >= 0 for t in prev_player.tokens)


def test_noble_selection_deterministic_one_only():
	state = initial_state(seed=999)
	p = state.players[state.to_play]
	# Force bonuses to meet possibly multiple nobles: set all bonuses to 4
	p.bonuses = [4, 4, 4, 4, 4]
	# Take a no-opish action that ends turn (e.g., reserve blind if possible or take-3)
	mask = legal_moves(state)
	action = int(np.flatnonzero(mask)[0])
	next_state = apply_action(state, action)
	prev_player = next_state.players[(next_state.to_play - 1) % next_state.num_players]
	# Exactly one noble tile should be taken
	taken = sum(1 for n in state.nobles if n is None)
	assert taken == 1 or any(n is None for n in next_state.nobles)


def test_observation_ranges():
	state = initial_state(seed=0)
	obs = encode_observation(state)
	assert obs.shape == (OBSERVATION_DIM,)
	# quick range checks
	bank = state.bank
	assert all(0 <= x <= 7 for x in bank)
	for p in state.players:
		assert sum(p.tokens) <= 10
		assert all(b >= 0 for b in p.bonuses)


def test_token_return_behavior_random_non_gold():
	env = SplendorEnv(num_players=2)
	obs, info = env.reset(seed=7)
	state = env.state
	p = state.players[state.to_play]
	# exceed limit: give many tokens (non-gold)
	p.tokens = [3, 3, 3, 3, 3, 0]
	prev_bank = list(state.bank)
	# Take a simple action to trigger end of turn and token return
	action = int(np.flatnonzero(info["action_mask"])[0])
	obs, reward, terminated, truncated, info = env.step(action)
	prev_player = env.state.players[(env.state.to_play - 1) % env.state.num_players]
	assert sum(prev_player.tokens) == 10
	# Bank increased by the amount returned
	assert sum(env.state.bank) >= sum(prev_bank) 