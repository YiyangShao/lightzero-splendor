import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Callable, Optional


class DualStepSelfPlayWrapper(gym.Wrapper):
	"""
	Advanced SelfPlay wrapper that simulates complete turns (both players) in each step.
	
	Key improvements over SelfPlayWrapper:
	1. Perfect policy consistency - each opponent policy called exactly once per turn
	2. Natural reward attribution - no reward flipping needed  
	3. Better debugging - explicit turn simulation with detailed info
	4. Cleaner logic - simpler code flow without complex reward handling
	5. Extensible - foundation for multi-agent support beyond 2 players
	
	The wrapper simulates a complete turn: agent moves, then opponent moves (if game continues).
	This eliminates policy inconsistency issues where the same (obs, info) could produce
	different opponent actions due to randomness or stateful policies.
	"""

	def __init__(self, env: gym.Env, opponent_policy: Callable, random_starts: bool = True, 
	             opponent_supplier: Optional[Callable] = None):
		"""
		Initialize the dual-step selfplay wrapper.
		
		Args:
			env: The base environment (should be a 2-player game)
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
		self.total_agent_actions = 0
		self.total_opponent_actions = 0

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
		self.total_agent_actions = 0
		self.total_opponent_actions = 0
		
		# Handle random starts: if opponent to play and coin flip, let them move first
		if self.random_starts and info.get("to_play", 0) == 1 and np.random.rand() < 0.5:
			# Opponent gets first move
			opp_action = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(opp_action)
			self.total_opponent_actions += 1
			
			if term or trunc:
				return obs, info
		
		# Advance until it's agent's turn (to_play == 0) or terminal
		while info.get("to_play", 0) == 1:
			opp_action = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(opp_action)
			self.total_opponent_actions += 1
			
			if term or trunc:
				break
				
		return obs, info

	def step(self, agent_action: int) -> Tuple[Any, float, bool, bool, Dict]:
		"""
		Simulate a complete turn: agent moves, then opponent moves (if game continues).
		
		This is the core improvement: we simulate both moves in a single step, ensuring
		perfect consistency and natural reward attribution.
		
		Args:
			agent_action: Action taken by the agent (player 0)
			
		Returns:
			obs: Observation after complete turn (from agent's perspective)
			reward: Agent's reward for this turn (natural, no flipping)
			terminated: Whether game has terminated
			truncated: Whether episode was truncated  
			info: Enhanced info dict with opponent data and turn statistics
		"""
		self.turn_count += 1
		self.total_agent_actions += 1
		
		# Phase 1: Agent move (Player 0)
		obs_after_agent, reward_agent, done_agent, truncated_agent, info_agent = self.env.step(agent_action)
		
		# Prepare enhanced info for return
		turn_info = {
			'turn_count': self.turn_count,
			'agent_action': agent_action,
			'total_agent_actions': self.total_agent_actions,
			'total_opponent_actions': self.total_opponent_actions,
			'phase': 'agent_only'  # Will update if opponent moves
		}
		turn_info.update(info_agent)
		
		if done_agent or truncated_agent:
			# Game ended on agent's move - return agent's natural reward
			turn_info['game_ended_on'] = 'agent_move'
			turn_info['turn_complete'] = True
			return obs_after_agent, reward_agent, done_agent, truncated_agent, turn_info
		
		# Phase 2: Opponent move (Player 1)
		if info_agent.get("to_play", 0) == 1:
			# Get opponent action for this exact game state (called exactly once!)
			opponent_action = self._opp_policy(obs_after_agent, info_agent)
			self.total_opponent_actions += 1
			
			# Apply opponent action  
			obs_final, reward_opponent, done_final, truncated_final, info_final = self.env.step(opponent_action)
			
			# Update info with complete turn data
			turn_info.update(info_final)
			turn_info.update({
				'opponent_action': opponent_action,
				'opponent_reward': reward_opponent,
				'total_opponent_actions': self.total_opponent_actions,
				'phase': 'complete_turn',
				'turn_complete': True
			})
			
			if done_final or truncated_final:
				# Game ended on opponent's move
				turn_info['game_ended_on'] = 'opponent_move'
				
				# Extract agent's final reward from final_rewards if available
				if 'final_rewards' in info_final and 0 in info_final['final_rewards']:
					agent_final_reward = info_final['final_rewards'][0]
				else:
					# Fallback: use reward_agent (should be 0 for non-terminal, correct for terminal)
					agent_final_reward = reward_agent
					
				return obs_final, agent_final_reward, done_final, truncated_final, turn_info
			else:
				# Turn completed without termination - agent gets 0 reward
				return obs_final, 0.0, done_final, truncated_final, turn_info
		else:
			# Invalid state - opponent should be to play after agent's non-terminal move
			raise RuntimeError(
				f"Invalid state after agent move: to_play={info_agent.get('to_play', 'unknown')}, "
				f"expected 1 for opponent. Game state may be corrupted."
			)

	def get_wrapper_stats(self) -> Dict[str, Any]:
		"""Get statistics about wrapper usage for debugging/analysis."""
		return {
			'turn_count': self.turn_count,
			'total_agent_actions': self.total_agent_actions,
			'total_opponent_actions': self.total_opponent_actions,
			'avg_opponent_actions_per_turn': self.total_opponent_actions / max(1, self.turn_count),
			'wrapper_type': 'DualStepSelfPlayWrapper'
		}


# Keep the original random_opponent for compatibility
def random_opponent(obs, info):
	"""Simple random opponent policy - chooses randomly from legal actions."""
	mask = info.get("action_mask")
	if mask is None:
		return 0
	legal = np.flatnonzero(mask)
	if len(legal) == 0:
		return 0
	return int(np.random.choice(legal))
