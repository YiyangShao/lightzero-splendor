import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Callable, Optional


class DualStepNativeWrapper(gym.Wrapper):
	"""
	Dual-step wrapper that calls env.step() twice and combines results.
	
	This approach eliminates wrapper overhead by:
	1. Making 2 env.step() calls per training step (same as DualStepSelfPlayWrapper)
	2. But returning both players' data explicitly 
	3. Allowing PPO to use dual_step() for maximum efficiency
	4. No modifications to base environment needed
	
	Key benefits:
	- Clean separation: wrapper handles dual logic, env stays pure
	- Performance: eliminates wrapper state management overhead  
	- Flexibility: can be used with any 2-player turn-based env
	- Backward compatibility: original env.step() still works
	"""

	def __init__(self, env: gym.Env, opponent_policy: Callable, random_starts: bool = True, 
	             opponent_supplier: Optional[Callable] = None):
		"""
		Initialize dual-step native wrapper.
		
		Args:
			env: Base environment (should be 2-player turn-based)
			opponent_policy: Policy function for opponent (obs, info) -> action
			random_starts: Whether to randomly let opponent move first sometimes
			opponent_supplier: Optional function that returns different opponent policies per episode
		"""
		super().__init__(env)
		self.opponent_policy = opponent_policy
		self.random_starts = random_starts
		self.opponent_supplier = opponent_supplier
		self._opp_policy = opponent_policy
		
		# Tracking for debugging and analysis
		self.turn_count = 0
		self.total_agent_steps = 0
		self.total_opponent_steps = 0

	def reset(self, **kwargs):
		"""Reset environment and handle random starts if enabled."""
		# Sample new opponent policy for this episode if supplier provided
		if self.opponent_supplier is not None:
			self._opp_policy = self.opponent_supplier()
		else:
			self._opp_policy = self.opponent_policy

		obs, info = self.env.reset(**kwargs)
		
		# Reset episode tracking
		self.turn_count = 0
		self.total_agent_steps = 0
		self.total_opponent_steps = 0
		
		# Handle random starts: if opponent to play and coin flip, let them move first
		if self.random_starts and info.get("to_play", 0) == 1 and np.random.rand() < 0.5:
			# Opponent gets first move
			opp_action = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(opp_action)
			self.total_opponent_steps += 1
			
			if term or trunc:
				return obs, info
		
		# Advance until it's agent's turn (to_play == 0) or terminal
		while info.get("to_play", 0) == 1:
			opp_action = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(opp_action)
			self.total_opponent_steps += 1
			
			if term or trunc:
				break
				
		return obs, info

	def step(self, action: int):
		"""
		Standard single-player step interface for backward compatibility.
		Uses dual_step internally but only returns agent perspective.
		"""
		agent_obs, agent_reward, _, _, done, info = self.dual_step(action)
		truncated = False  # Splendor doesn't use truncation
		return agent_obs, agent_reward, done, truncated, info

	def dual_step(self, agent_action: int) -> Tuple[np.ndarray, float, np.ndarray, float, bool, Dict[str, Any]]:
		"""
		Execute a complete turn: agent move + opponent move.
		
		This method calls env.step() twice (once for each player) and combines
		the results to eliminate wrapper overhead while providing dual-player data.
		
		Args:
			agent_action: Action taken by the agent (player 0)
			
		Returns:
			agent_obs: Observation for agent (player 0) after complete turn
			agent_reward: Agent's reward for this turn
			opponent_obs: Observation for opponent (player 1) after complete turn  
			opponent_reward: Opponent's reward for this turn
			done: Whether game has terminated
			info: Enhanced info dict with turn details
		"""
		if not hasattr(self.env, 'state') or self.env.state is None:
			raise RuntimeError("Cannot call dual_step() before reset()")
		
		# Verify it's agent's turn
		current_state = self.env.state
		if current_state.to_play != 0:
			raise ValueError("dual_step() requires agent (player 0) to move first")
		
		self.turn_count += 1
		self.total_agent_steps += 1
		
		# Phase 1: Agent move (Player 0)
		agent_obs_after, agent_reward, done_after_agent, truncated_after_agent, info_after_agent = self.env.step(agent_action)
		
		# Build turn info starting with agent data
		turn_info = {
			'turn_count': self.turn_count,
			'agent_action': agent_action,
			'total_agent_steps': self.total_agent_steps,
			'total_opponent_steps': self.total_opponent_steps,
			'phase': 'agent_only'
		}
		turn_info.update(info_after_agent)
		
		if done_after_agent or truncated_after_agent:
			# Game ended on agent's move
			turn_info.update({
				'opponent_action': None,
				'opponent_reward': self._extract_opponent_reward(info_after_agent, 1),
				'turn_complete': True,
				'game_ended_on': 'agent_move'
			})
			
			# For terminal states, both players get same observation
			opponent_obs = agent_obs_after  # Terminal obs is same for both
			opponent_reward = turn_info['opponent_reward']
			
			return agent_obs_after, agent_reward, opponent_obs, opponent_reward, True, turn_info
		
		# Phase 2: Opponent move (Player 1)
		current_state_after_agent = self.env.state
		if current_state_after_agent.to_play != 1:
			raise ValueError(f"Expected opponent (player 1) to move after agent, got to_play={current_state_after_agent.to_play}")
		
		# Get opponent action
		opponent_action = self._opp_policy(agent_obs_after, info_after_agent)
		self.total_opponent_steps += 1
		
		# Execute opponent move
		final_obs, opponent_reward, done_final, truncated_final, info_final = self.env.step(opponent_action)
		
		# Extract agent's reward for the complete turn
		if done_final or truncated_final:
			# Game ended on opponent's move - get agent's final reward
			agent_final_reward = self._extract_opponent_reward(info_final, 0)
			game_ended_on = 'opponent_move'
		else:
			# Turn completed without termination - agent gets 0 for non-terminal turn
			agent_final_reward = 0.0
			game_ended_on = None
		
		# Update turn info with complete turn data
		turn_info.update(info_final)
		turn_info.update({
			'opponent_action': opponent_action,
			'opponent_reward': opponent_reward,
			'total_opponent_steps': self.total_opponent_steps,
			'phase': 'complete_turn',
			'turn_complete': True,
			'game_ended_on': game_ended_on
		})
		
		# Generate observations for both players
		# Agent gets the final observation (which should be from their perspective after opponent's move)
		agent_final_obs = final_obs
		
		# For opponent observation, we need to think about perspective
		# In a terminal state, both get same obs. In non-terminal, opponent would see next state from their perspective
		if done_final or truncated_final:
			opponent_final_obs = final_obs  # Same terminal observation
		else:
			# Non-terminal: opponent would see the state from their perspective
			# Since to_play should now be 0 (back to agent), opponent sees "opponent view"
			opponent_final_obs = final_obs  # This is actually fine - obs encoding handles perspective
		
		return agent_final_obs, agent_final_reward, opponent_final_obs, opponent_reward, done_final, turn_info

	def _extract_opponent_reward(self, info: Dict, player_id: int) -> float:
		"""Extract reward for specific player from info dict."""
		if 'final_rewards' in info and player_id in info['final_rewards']:
			return info['final_rewards'][player_id]
		else:
			# No final rewards available, assume 0.0 for non-terminal
			return 0.0

	def get_wrapper_stats(self) -> Dict[str, Any]:
		"""Get statistics about wrapper usage for debugging/analysis."""
		return {
			'turn_count': self.turn_count,
			'total_agent_steps': self.total_agent_steps,
			'total_opponent_steps': self.total_opponent_steps,
			'avg_opponent_steps_per_turn': self.total_opponent_steps / max(1, self.turn_count),
			'wrapper_type': 'DualStepNativeWrapper'
		}


# For compatibility
def random_opponent(obs, info):
	"""Simple random opponent policy - chooses randomly from legal actions."""
	mask = info.get("action_mask")
	if mask is None:
		return 0
	legal = np.flatnonzero(mask)
	if len(legal) == 0:
		return 0
	return int(np.random.choice(legal))
