import numpy as np

from splendor_gym.envs import SplendorEnv
from splendor_gym.engine import legal_moves


def test_no_legal_move_draw():
	env = SplendorEnv(seed=0)
	obs, info = env.reset()
	# Empty bank
	env.state.bank[:] = [0, 0, 0, 0, 0, 0]
	# Current player tokens exactly 10, no gold
	p = env.state.players[env.state.to_play]
	p.tokens[:] = [10, 0, 0, 0, 0, 0]
	# Reserve 3 cards
	p.reserved = env.state.decks[1][:3]
	# Make visible slots unbuyable: clear all
	for t in (1, 2, 3):
		env.state.board[t] = [None, None, None, None]
	m = legal_moves(env.state)
	assert not np.any(m)
	# Step should result in draw
	obs, reward, terminated, truncated, info = env.step(0)
	assert terminated and reward == 0 and env.state.winner_index is None 