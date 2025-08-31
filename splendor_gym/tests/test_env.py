import numpy as np
import gymnasium as gym

from splendor_gym.envs import SplendorEnv
from splendor_gym.engine.encode import TOTAL_ACTIONS, OBSERVATION_DIM


def test_reset_and_step_shapes():
	env = SplendorEnv(num_players=2)
	obs, info = env.reset(seed=123)
	assert isinstance(obs, np.ndarray)
	assert obs.shape == (OBSERVATION_DIM,)
	assert "action_mask" in info
	assert info["action_mask"].shape == (TOTAL_ACTIONS,)

	# take a legal action
	mask = info["action_mask"]
	action = int(np.flatnonzero(mask)[0])
	obs, reward, terminated, truncated, info = env.step(action)
	assert isinstance(obs, np.ndarray)
	assert obs.shape == (OBSERVATION_DIM,)
	assert isinstance(reward, float)
	assert isinstance(terminated, bool)
	assert isinstance(truncated, bool)
	assert "action_mask" in info


def test_random_rollout_no_crash():
	env = SplendorEnv(num_players=2)
	obs, info = env.reset(seed=0)
	for _ in range(50):
		if info["action_mask"].sum() == 0:
			break
		action = int(np.random.choice(np.flatnonzero(info["action_mask"])) )
		obs, reward, terminated, truncated, info = env.step(action)
		if terminated or truncated:
			break 