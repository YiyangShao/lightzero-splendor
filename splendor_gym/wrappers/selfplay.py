import gymnasium as gym
import numpy as np


class SelfPlayWrapper(gym.Wrapper):
	"""
	Expose a single-agent MDP by:
	- Returning obs/action_mask for the current player only
	- Internally playing the opponent's move whenever to_play == 1
	- Returning reward from the current player's POV (+1 win, -1 loss, 0 otherwise)
	"""

	def __init__(self, env: gym.Env, opponent_policy, random_starts: bool = True, opponent_supplier=None):
		super().__init__(env)
		self.opponent_policy = opponent_policy
		self.random_starts = random_starts
		self.opponent_supplier = opponent_supplier
		self._opp_policy = opponent_policy

	def reset(self, **kwargs):
		# Sample opponent policy for this episode if supplier provided
		if self.opponent_supplier is not None:
			self._opp_policy = self.opponent_supplier()
		else:
			self._opp_policy = self.opponent_policy

		obs, info = self.env.reset(**kwargs)
		# Optional random starts: if opponent to play and coin flip, let opponent move once
		if self.random_starts and info.get("to_play", 0) == 1 and np.random.rand() < 0.5:
			a = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(a)
			if term or trunc:
				return obs, info
		# Advance until it's our turn (to_play == 0) or terminal
		while info.get("to_play", 0) == 1:
			a = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(a)
			if term or trunc:
				break
		return obs, info

	def step(self, action):
		# Our move
		obs, reward, term, trunc, info = self.env.step(action)
		if term or trunc:
			# Reward already from winner POV; correct perspective: if winner is us, reward=+1 else -1 or 0 for draw
			return obs, reward, term, trunc, info
		# Opponent move
		if info.get("to_play", 0) == 1:
			a = self._opp_policy(obs, info)
			obs, opp_reward, term, trunc, info = self.env.step(a)
			# Env reward is already from our (player 0) POV; pass through on terminal, else 0
			reward = opp_reward if (term or trunc) else 0.0
			return obs, reward, term, trunc, info
		# Shouldn't happen, but return zero reward
		return obs, 0.0, term, trunc, info


def random_opponent(obs, info):
	mask = info.get("action_mask")
	if mask is None:
		return 0
	legal = np.flatnonzero(mask)
	if len(legal) == 0:
		return 0
	return int(np.random.choice(legal)) 