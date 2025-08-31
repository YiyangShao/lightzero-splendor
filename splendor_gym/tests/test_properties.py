import numpy as np
from gymnasium.utils.env_checker import check_env
import warnings

from splendor_gym.envs import SplendorEnv
from splendor_gym.engine import initial_state, legal_moves, apply_action
from splendor_gym.engine.encode import TOTAL_ACTIONS


def scripted_policy(mask):
	# pick lowest-index legal action
	legal = np.flatnonzero(mask)
	return int(legal[0]) if len(legal) > 0 else 0


def test_env_api_ok():
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=UserWarning)
		env = SplendorEnv(num_players=2, seed=123)
		check_env(env)


def test_deterministic_reset_and_first_actions():
	env1 = SplendorEnv(num_players=2)
	env2 = SplendorEnv(num_players=2)
	obs1, info1 = env1.reset(seed=42)
	obs2, info2 = env2.reset(seed=42)
	assert np.array_equal(obs1, obs2)
	for _ in range(5):
		a1 = scripted_policy(info1["action_mask"])
		a2 = scripted_policy(info2["action_mask"])
		assert a1 == a2
		obs1, r1, t1, tr1, info1 = env1.step(a1)
		obs2, r2, t2, tr2, info2 = env2.step(a2)
		assert np.array_equal(obs1, obs2)
		assert (r1, t1, tr1) == (r2, t2, tr2)


def test_mask_invariants_basic():
	state = initial_state(seed=7)
	mask = legal_moves(state)
	# take-2 only if bank[color] >= 4
	for ci in range(5):
		assert (mask[10 + ci] == 1) == (state.bank[ci] >= 4)
	# reserve blind only if deck not empty and reserved < 3
	p = state.players[state.to_play]
	for tier in (1, 2, 3):
		idx = 39 + (tier - 1)
		assert (mask[idx] == 1) == (len(state.decks[tier]) > 0 and len(p.reserved) < 3)
	# buy/reserve visible masked off when slot empty
	for tier in (1, 2, 3):
		for slot in range(4):
			buy_idx = 15 + (tier - 1) * 4 + slot
			res_idx = 27 + (tier - 1) * 4 + slot
			present = state.board[tier][slot] is not None
			if not present:
				assert mask[buy_idx] == 0 and mask[res_idx] == 0 