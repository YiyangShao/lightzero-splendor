from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

from ..engine import (
	SplendorState,
	legal_moves,
	apply_action,
	is_terminal,
	winner,
	initial_state,
)
from ..engine.state import TOKEN_COLORS, STANDARD_COLORS
from ..engine.encode import TOTAL_ACTIONS, OBSERVATION_DIM, encode_observation


class SplendorEnv(gym.Env):
	metadata = {"render_modes": ["human"], "name": "Splendor-v0"}

	def __init__(self, num_players: int = 2, render_mode: str | None = None, seed: int | None = None):
		super().__init__()
		if num_players != 2:
			raise NotImplementedError("Current env supports 2 players only.")
		self.num_players = num_players
		self.render_mode = render_mode

		self.action_space = spaces.Discrete(TOTAL_ACTIONS)
		self.observation_space = spaces.Box(
			low=0, high=50, shape=(OBSERVATION_DIM,), dtype=np.int32
		)

		self.state: SplendorState | None = None
		self.current_player: int = 0

	def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
		super().reset(seed=seed)
		engine_seed = int(self.np_random.integers(0, 2**31 - 1))  # type: ignore[attr-defined]
		self.state = initial_state(num_players=self.num_players, seed=engine_seed)
		self.current_player = self.state.to_play
		obs = encode_observation(self.state)
		mask = np.array(legal_moves(self.state), dtype=np.int8)
		info = {"action_mask": mask, "to_play": self.state.to_play}
		return obs, info

	def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
		assert self.state is not None, "Call reset() first"
		if is_terminal(self.state):
			raise RuntimeError("Cannot call step() after episode termination. Call reset().")
		mask = legal_moves(self.state)
		if np.sum(mask) == 0:
			self.state.game_over = True
			self.state.winner_index = None
			self.state.to_play = 0
			obs = encode_observation(self.state)
			return obs, 0.0, True, False, {"action_mask": np.zeros(self.action_space.n, dtype=np.int8), "to_play": self.state.to_play, "draw": True}
		if not (0 <= action < self.action_space.n):
			raise ValueError("Action out of bounds for action_space")
		if mask[action] != 1:
			obs = encode_observation(self.state)
			return obs, -0.01, False, False, {"illegal_action": True, "action_mask": np.array(mask, dtype=np.int8), "to_play": self.state.to_play}

		self.state = apply_action(self.state, action)
		obs = encode_observation(self.state)
		terminated = is_terminal(self.state)
		reward = 0.0
		if terminated:
			w = winner(self.state)
			# Draw penalty if turn limit reached
			if w is None and getattr(self.state, "turn_limit_reached", False):
				reward = -0.1
			else:
				reward = 0.0 if w is None else (1.0 if w == self.current_player else -1.0)
		mask = np.array(legal_moves(self.state), dtype=np.int8) if not terminated else np.zeros(self.action_space.n, dtype=np.int8)
		info = {"action_mask": mask, "to_play": self.state.to_play}
		if terminated and getattr(self.state, "turn_limit_reached", False):
			info["turn_limit"] = True
		return obs, float(reward), bool(terminated), False, info

	def render(self):
		if self.render_mode not in ("human", None):
			return
		assert self.state is not None
		p = self.state.players[self.state.to_play]
		print(f"Turn {self.state.turn_count} — Player {self.state.to_play}")
		print(f"Bank: {dict(zip(TOKEN_COLORS, self.state.bank))}")
		print(f"You: tokens={dict(zip(TOKEN_COLORS, p.tokens))} bonuses={dict(zip(STANDARD_COLORS, p.bonuses))} pp={p.prestige}")  # noqa: E501


def make(num_players: int = 2, render_mode: str | None = None, seed: int | None = None) -> SplendorEnv:
	return SplendorEnv(num_players=num_players, render_mode=render_mode, seed=seed) 