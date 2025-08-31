import numpy as np
import pytest
import time
import warnings

from gymnasium.utils.env_checker import check_env
from splendor_gym.engine.encode import OBSERVATION_DIM
from splendor_gym.tests.utils import make_env, get_mask


def test_api_checker():
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=UserWarning)
		env = make_env()
		check_env(env)


def test_reset_shapes_and_types():
	env = make_env()
	obs, info = env.reset(seed=123)
	assert env.observation_space.contains(obs)
	assert obs.shape == env.observation_space.shape == (OBSERVATION_DIM,)
	assert env.action_space.n == 45
	m = get_mask(info)
	assert m.shape == (env.action_space.n,)
	assert m.dtype == np.int8


def test_step_shapes_and_types():
	env = make_env()
	obs, info = env.reset(seed=123)
	mask = get_mask(info)
	if mask.any():
		a = int(np.where(mask == 1)[0][0])
		obs2, r, terminated, truncated, info2 = env.step(a)
		assert env.observation_space.contains(obs2)
		assert isinstance(r, (int, float))
		assert isinstance(terminated, (bool, np.bool_))
		assert isinstance(truncated, (bool, np.bool_))
		m2 = get_mask(info2)
		assert m2.shape == (env.action_space.n,)


def test_deterministic_reset_with_seed():
	env1 = make_env()
	env2 = make_env()
	obs1, info1 = env1.reset(seed=42)
	obs2, info2 = env2.reset(seed=42)
	np.testing.assert_array_equal(obs1, obs2)
	np.testing.assert_array_equal(get_mask(info1), get_mask(info2))


def test_deterministic_scripted_prefix():
	env1 = make_env()
	env2 = make_env()
	o1, i1 = env1.reset(seed=999)
	o2, i2 = env2.reset(seed=999)

	def scripted_step(env, obs, info, moves=20):
		hist = []
		for _ in range(moves):
			m = get_mask(info)
			if not m.any():
				obs, r, term, trunc, info = env.step(0)
				hist.append((None, r, term, trunc))
				break
			a = int(np.where(m == 1)[0][0])
			obs, r, term, trunc, info = env.step(a)
			hist.append((a, r, term, trunc))
			if term or trunc:
				break
		return hist

	h1 = scripted_step(env1, o1, i1, moves=20)
	h2 = scripted_step(env2, o2, i2, moves=20)
	assert h1 == h2


def test_action_space_contains_and_mask_alignment():
	env = make_env()
	obs, info = env.reset(seed=123)
	mask = get_mask(info)
	assert mask.shape[0] == env.action_space.n
	for _ in range(20):
		a = env.action_space.sample()
		assert env.action_space.contains(a)


def test_step_after_terminated_raises():
	env = make_env()
	obs, info = env.reset(seed=123)
	terminated_state = False
	for _ in range(2000):
		m = get_mask(info)
		if not m.any():
			obs, r, term, trunc, info = env.step(0)
			if term or trunc:
				terminated_state = True
				break
		a = int(np.where(m == 1)[0][0])
		obs, r, term, trunc, info = env.step(a)
		if term or trunc:
			terminated_state = True
			break
	assert terminated_state
	with pytest.raises((AssertionError, ValueError, RuntimeError)):
		# After termination, step should not be allowed
		env.step(0)


def test_masked_on_actions_execute_masked_off_do_not():
	env = make_env()
	obs, info = env.reset(seed=321)
	mask = get_mask(info)
	legal_indices = np.where(mask == 1)[0].tolist()
	if legal_indices:
		a = int(legal_indices[0])
		_obs2, _r2, _t2, _tr2, _info2 = env.step(a)
	illegal_indices = np.where(mask == 0)[0].tolist()
	if illegal_indices:
		# Illegal masked-off actions should not raise but return a small penalty and same obs
		_obs3, r3, t3, tr3, info3 = env.step(int(illegal_indices[0]))
		assert r3 <= 0
		assert not t3 and not tr3


def test_render_no_crash():
	env = make_env()
	env.render_mode = "human"
	obs, info = env.reset(seed=0)
	env.render()
	env.close()


@pytest.mark.slow
def test_rollout_perf_smoke():
	env = make_env()
	obs, info = env.reset(seed=123)
	n = 20000
	t0 = time.time()
	steps = 0
	while steps < n:
		m = get_mask(info)
		if not m.any():
			obs, r, term, trunc, info = env.step(0)
			if term or trunc:
				obs, info = env.reset(seed=steps)
			continue
		a = int(np.where(m == 1)[0][0])
		obs, r, term, trunc, info = env.step(a)
		steps += 1
		if term or trunc:
			obs, info = env.reset(seed=steps)
	elapsed = time.time() - t0
	sps = steps / elapsed
	print(f"SPS={sps:.0f}")
	assert sps > 8000 