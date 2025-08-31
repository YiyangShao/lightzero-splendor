import numpy as np

from splendor_gym.tests.utils import make_env, mask_from_state
from splendor_gym.engine.state import COLOR_INDEX


def test_take_two_when_only_two_colors():
	env = make_env(seed=123)
	obs, info = env.reset(seed=123)
	# Empty bank first
	env.state.bank[:] = [0, 0, 0, 0, 0, 0]
	# Only 2 colors with ≥1 token: diamond (white) and emerald (green)
	env.state.bank[COLOR_INDEX["white"]] = 1
	env.state.bank[COLOR_INDEX["green"]] = 2
	m = mask_from_state(env)
	legal_take_actions = [i for i in range(0, 10) if m[i] == 1]
	assert len(legal_take_actions) == 1
	a = legal_take_actions[0]
	obs, reward, terminated, truncated, info = env.step(a)
	last_player = env.state.players[(env.state.to_play - 1) % env.state.num_players]
	assert last_player.tokens[COLOR_INDEX["white"]] + last_player.tokens[COLOR_INDEX["green"]] == 2


def test_take_one_when_only_one_color():
	env = make_env(seed=123)
	obs, info = env.reset(seed=123)
	# Empty bank first
	env.state.bank[:] = [0, 0, 0, 0, 0, 0]
	env.state.bank[COLOR_INDEX["black"]] = 3  # onyx
	m = mask_from_state(env)
	legal_take_actions = [i for i in range(0, 10) if m[i] == 1]
	assert len(legal_take_actions) == 1
	a = legal_take_actions[0]
	obs, reward, terminated, truncated, info = env.step(a)
	last_player = env.state.players[(env.state.to_play - 1) % env.state.num_players]
	assert last_player.tokens[COLOR_INDEX["black"]] == 1 