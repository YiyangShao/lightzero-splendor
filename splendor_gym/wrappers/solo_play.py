import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Optional


class SoloPlayWrapper(gym.Wrapper):
	"""
	Solo play wrapper where only player 0 makes real moves, player 1 makes empty moves.
	
	This wrapper is designed for training agents to play optimally against no opposition,
	focusing on speed and efficiency. Uses progressive reward structure with exponential 
	incentives for faster games.
	
	Key features:
	- Player 0 (agent) makes normal moves
	- Player 1 (dummy) makes "empty moves" that don't change game state
	- Progressive reward: baseline 50 turns = 0, with exponential bonuses for faster completion
	- Reward rates: 100-40 (+0.2/turn), 40-35 (+1.0/turn), 35-30 (+2.0/turn), 30-20 (+5.0/turn), <20 (+10.0/turn)
	- Info collected from player 1's perspective for observation encoding
	- Perfect for training speed-optimal strategies
	"""

	def __init__(self, env: gym.Env, max_turns: Optional[int] = None):
		"""
		Initialize solo play wrapper.
		
		Args:
			env: Base environment (should be 2-player turn-based)
			max_turns: Maximum turns before forcing termination (None = no limit)
		"""
		super().__init__(env)
		
		# Verify this is a 2-player environment
		if not hasattr(env, 'num_players') or env.num_players != 2:
			raise ValueError("SoloPlayWrapper requires a 2-player environment")
		
		self.max_turns = max_turns
		
		# Tracking for speed-based rewards
		self.turn_count = 0
		self.total_games = 0
		self.fastest_win = None
		self.slowest_win = None

	def reset(self, **kwargs):
		"""Reset environment for new solo play episode."""
		obs, info = self.env.reset(**kwargs)
		
		# Reset episode tracking
		self.turn_count = 0
		
		# If it's player 1's turn initially, handle the empty move to get to player 0
		while info.get("to_play", 0) != 0:
			obs, info = self._make_empty_move()
			if self.env.state and hasattr(self.env.state, 'game_over') and self.env.state.game_over:
				break
		
		return obs, info

	def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
		"""
		Execute a solo play step: agent move + empty opponent move.
		
		Args:
			action: Action taken by the agent (player 0)
			
		Returns:
			obs: Observation after complete turn
			reward: Speed-based reward (higher for faster wins)
			terminated: Whether game has terminated
			truncated: Whether episode was truncated
			info: Enhanced info dict with speed metrics
		"""
		if not hasattr(self.env, 'state') or self.env.state is None:
			raise RuntimeError("Cannot call step() before reset()")
		
		# Verify it's agent's turn
		current_state = self.env.state
		if current_state.to_play != 0:
			raise ValueError("step() requires agent (player 0) to move")
		
		self.turn_count += 1
		
		# Phase 1: Agent move (Player 0)
		agent_obs, agent_reward, done_after_agent, truncated_after_agent, info_after_agent = self.env.step(action)
		
		# Build info with speed metrics
		speed_info = {
			'turn_count': self.turn_count,
			'agent_action': action,
			'total_games': self.total_games,
			'fastest_win': self.fastest_win,
			'slowest_win': self.slowest_win,
			'phase': 'agent_only'
		}
		speed_info.update(info_after_agent)
		
		if done_after_agent or truncated_after_agent:
			# Game ended on agent's move
			final_reward = self._calculate_speed_reward(done_after_agent)
			speed_info.update({
				'game_ended_on': 'agent_move',
				'speed_reward': final_reward,
				'turns_to_completion': self.turn_count
			})
			
			self._update_speed_stats()
			return agent_obs, final_reward, True, truncated_after_agent, speed_info
		
		
		# Check turn limit
		if self.max_turns and self.turn_count >= self.max_turns:
			# Force termination due to turn limit
			final_reward = self._calculate_speed_reward(False)  # Penalty for timeout
			speed_info.update({
				'game_ended_on': 'turn_limit',
				'speed_reward': final_reward,
				'turns_to_completion': self.turn_count,
				'terminated_reason': 'max_turns_reached'
			})
			
			self._update_speed_stats()
			return agent_obs, final_reward, True, True, speed_info
		
		# Phase 2: Empty opponent move (Player 1)
		current_state_after_agent = self.env.state
		if current_state_after_agent.to_play != 1:
			raise ValueError(f"Expected opponent (player 1) to move after agent, got to_play={current_state_after_agent.to_play}")
		
		# Make empty move for opponent
		final_obs, final_info = self._make_empty_move()
		
		# Update info with complete turn data
		speed_info.update(final_info)
		speed_info.update({
			'phase': 'complete_turn',
			'opponent_action': 'empty_move'
		})
		
		# Check if game somehow ended (shouldn't happen with empty moves)
		final_state = self.env.state
		if hasattr(final_state, 'game_over') and final_state.game_over:
			final_reward = self._calculate_speed_reward(True)
			speed_info.update({
				'game_ended_on': 'empty_move',
				'speed_reward': final_reward,
				'turns_to_completion': self.turn_count
			})
			
			self._update_speed_stats()
			return final_obs, final_reward, True, False, speed_info
		
		# Continue playing - no reward for non-terminal turns
		return final_obs, 0.0, False, False, speed_info

	def _make_empty_move(self) -> Tuple[np.ndarray, Dict[str, Any]]:
		"""
		Make an empty move for player 1 that doesn't change the game state.
		
		This advances the turn to player 0 without affecting the board,
		cards, tokens, or any other game elements.
		
		Returns:
			obs: Observation from player 1's perspective (as per wrapper design)
			info: Updated info dict
		"""
		# Store current state before empty move
		current_state = self.env.state
		
		# Simply advance to_play without changing anything else
		# Create a copy of the state with only to_play changed
		new_state = current_state.copy()
		new_state.to_play = (new_state.to_play + 1) % new_state.num_players
		new_state.move_count += 1
		
		# Update environment state
		self.env.state = new_state
		
		# Generate observation from player 1's perspective (as specified)
		from ..engine.encode import encode_observation
		
		# Temporarily set to_play to 1 for observation encoding
		temp_state = new_state.copy()
		temp_state.to_play = 1
		obs = encode_observation(temp_state)
		
		# Generate info
		from ..engine.rules import legal_moves
		info = {
			'action_mask': legal_moves(new_state),
			'to_play': new_state.to_play,
			'move_count': new_state.move_count,
			'turn_count': self.turn_count
		}
		
		return obs, info

	def _calculate_speed_reward(self, won: bool) -> float:
		"""
		Calculate progressive speed-based reward with exponential incentives for faster games.
		
		Reward structure (baseline: 50 turns = 0 reward):
		- 100-40 turns: each turn faster = +0.2 reward  
		- 40-35 turns: each turn faster = +1.0 reward
		- 35-30 turns: each turn faster = +2.0 reward  
		- 30-20 turns: each turn faster = +5.0 reward
		- <20 turns: each turn faster = +10.0 reward (extreme bonus)
		
		Examples:
		- 45 turns: 1.0 reward (5 * 0.2)
		- 38 turns: 4.0 reward (10*0.2 + 2*1.0)  
		- 32 turns: 19.0 reward (10*0.2 + 5*1.0 + 3*2.0)
		- 25 turns: 44.0 reward (10*0.2 + 5*1.0 + 5*2.0 + 5*5.0)
		
		Args:
			won: Whether the agent won the game (unused but kept for consistency)
			
		Returns:
			Progressive reward based on turn count
		"""
		turns = self.turn_count
		
		if turns >= 50:
			# 50+ turns: negative linear penalty
			reward = 50.0 - float(turns)
		else:
			# Progressive bonus structure for sub-50 turn games
			reward = 0.0  # Start from 50-turn baseline
			remaining_turns = 50 - turns
			
			# Apply progressive bonuses from slowest to fastest
			if remaining_turns > 0:
				# 50-40 turns: 0.2 per turn (up to 10 turns * 0.2 = 2.0)
				chunk1 = min(remaining_turns, 10)
				reward += chunk1 * 0.2
				remaining_turns -= chunk1
			
			if remaining_turns > 0:
				# 40-35 turns: 1.0 per turn (up to 5 turns * 1.0 = 5.0)  
				chunk2 = min(remaining_turns, 5)
				reward += chunk2 * 1.0
				remaining_turns -= chunk2
			
			if remaining_turns > 0:
				# 35-30 turns: 2.0 per turn (up to 5 turns * 2.0 = 10.0)
				chunk3 = min(remaining_turns, 5) 
				reward += chunk3 * 2.0
				remaining_turns -= chunk3
			
			if remaining_turns > 0:
				# 30-20 turns: 5.0 per turn (up to 10 turns * 5.0 = 50.0)
				chunk4 = min(remaining_turns, 10)
				reward += chunk4 * 5.0
				remaining_turns -= chunk4
			
			if remaining_turns > 0:
				# <20 turns: 10.0 per turn (extreme bonus for superhuman speed)
				chunk5 = remaining_turns
				reward += chunk5 * 10.0
		
		return reward

	def _update_speed_stats(self):
		"""Update speed statistics for tracking performance."""
		self.total_games += 1
		
		# Update fastest/slowest win records
		current_state = self.env.state
		if hasattr(current_state, 'game_over') and current_state.game_over:
			from ..engine.rules import winner
			w = winner(current_state)
			
			if w == 0:  # Agent won
				if self.fastest_win is None or self.turn_count < self.fastest_win:
					self.fastest_win = self.turn_count
				
				if self.slowest_win is None or self.turn_count > self.slowest_win:
					self.slowest_win = self.turn_count

	def get_wrapper_stats(self) -> Dict[str, Any]:
		"""Get statistics about solo play performance."""
		return {
			'total_games': self.total_games,
			'fastest_win': self.fastest_win,
			'slowest_win': self.slowest_win,
			'current_turn_count': self.turn_count,
			'wrapper_type': 'SoloPlayWrapper'
		}
