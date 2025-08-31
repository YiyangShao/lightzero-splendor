from __future__ import annotations

import copy
import pickle
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

# Import Splendor gym engine pieces
from splendor_gym.engine.encode import (
    TOTAL_ACTIONS,
    OBSERVATION_DIM,
    encode_observation,
)
from splendor_gym.engine.state import SplendorState
from splendor_gym.engine import (
    initial_state,
    legal_moves,
    apply_action,
    is_terminal,
    winner,
)


@ENV_REGISTRY.register('splendor')
class SplendorLightZeroEnv(BaseEnv):
    """
    LightZero-compatible Splendor environment.

    - Observation: dict with 'observation' (C,H,W), 'action_mask', 'board' (serialized state), 'current_player_index', 'to_play'
    - Actions: discrete [0, TOTAL_ACTIONS)
    - Modes: default self_play_mode; eval/play_with_bot not implemented (board games bot optional)
    """

    config = dict(
        env_id='Splendor',
        battle_mode='self_play_mode',  # {'self_play_mode', 'play_with_bot_mode', 'eval_mode'}
        render_mode=None,
        channel_last=False,
        scale=True,
        stop_value=0.95,  # win rate goal; not strictly used
        # MCTS backend toggle (for parity with other board games)
        alphazero_mcts_ctree=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None):
        self.cfg = EasyDict(cfg or {})
        # Merge defaults
        for k, v in self.config.items():
            if k not in self.cfg:
                self.cfg[k] = v

        self.battle_mode = self.cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # Used by AlphaZero MCTS simulation env
        self.battle_mode_in_simulation_env = 'self_play_mode'
        self.render_mode = self.cfg.render_mode
        self.channel_last = self.cfg.channel_last
        self.scale = self.cfg.scale
        self.alphazero_mcts_ctree = bool(self.cfg.get('alphazero_mcts_ctree', False))

        # Two-player game indices {0,1}; align with engine's state.to_play
        self.players = [0, 1]
        self._current_player = 0

        # Spaces (AlphaZero uses (C,H,W)); we reshape 225 -> (1, 15, 15)
        self._obs_hw = (15, 15)
        self._observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1, self._obs_hw[0], self._obs_hw[1]),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Discrete(TOTAL_ACTIONS)
        self._reward_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Engine state
        self.state: SplendorState | None = None
        self.start_player_index = 0

    # --------- Required properties ---------
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    # --------- Helpers for LightZero/AlphaZero ---------
    @property
    def legal_actions(self) -> List[int]:
        mask = legal_moves(self.state)
        return [i for i in range(TOTAL_ACTIONS) if mask[i] == 1]

    @property
    def current_player(self) -> int:
        return self._current_player

    @current_player.setter
    def current_player(self, value: int) -> None:
        self._current_player = int(value)

    @property
    def current_player_index(self) -> int:
        return int(self._current_player)

    @property
    def to_play(self) -> int:
        # Next player index in two-player alt-turn setting
        return 1 - self.current_player

    def _encode_obs_grid(self, state: SplendorState) -> Tuple[np.ndarray, np.ndarray]:
        vec = encode_observation(state).astype(np.float32)  # [225]
        grid = vec.reshape(self._obs_hw[0], self._obs_hw[1])  # [H,W]
        raw = grid[None, ...]  # [1,H,W]
        if self.scale:
            # Simple normalization by max token/card value range; keep in [0,1] best-effort
            scaled = (grid / 50.0).clip(0.0, 1.0)[None, ...].astype(np.float32)
        else:
            scaled = raw
        return raw, scaled

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        raw, scaled = self._encode_obs_grid(self.state)
        if self.channel_last:
            # (C,H,W) -> (H,W,C)
            return np.transpose(raw, (1, 2, 0)), np.transpose(scaled, (1, 2, 0))
        return raw, scaled

    # --------- Core API ---------
    def reset(self, start_player_index: int = 0, init_state: Any = None, **kwargs) -> Dict[str, Any]:
        self.start_player_index = int(start_player_index)
        if init_state is None:
            seed = getattr(self, '_seed', None)
            self.state = initial_state(num_players=2, seed=seed)
        else:
            # Accept bytes or SplendorState directly
            if isinstance(init_state, (bytes, bytearray)):
                self.state = pickle.loads(init_state)
            elif isinstance(init_state, SplendorState):
                self.state = copy.deepcopy(init_state)
            else:
                # Fallback: attempt pickle
                try:
                    self.state = pickle.loads(init_state)
                except Exception as e:
                    raise ValueError(f"Unsupported init_state type: {type(init_state)}; error={e}")

        # Set current player
        self._current_player = int(self.state.to_play)

        # Build obs
        action_mask = np.array(legal_moves(self.state), dtype=np.int8)
        obs = {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': pickle.dumps(self.state, protocol=pickle.HIGHEST_PROTOCOL),
            'current_player_index': self.current_player_index,
            'to_play': int(self.current_player),
        }
        return obs

    def step(self, action: int) -> BaseEnvTimestep:
        if action not in self.legal_actions:
            # Strict per user's rule: unexpected conditions must throw.
            raise ValueError(f"Illegal action {action}; legal={self.legal_actions}")

        # Apply action
        self.state = apply_action(self.state, int(action))

        done = bool(is_terminal(self.state))
        rew = 0.0
        info: Dict[str, Any] = {}

        if done:
            w = winner(self.state)
            if w is None:
                rew = 0.0
            else:
                # Reward from perspective of player who just moved (current_player before swap)
                rew = 1.0 if w == self.current_player else -1.0
            info['eval_episode_return'] = float(rew)

        # Swap player turn (engine maintains state.to_play)
        self._current_player = int(self.state.to_play)

        action_mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8) if done else np.array(legal_moves(self.state), dtype=np.int8)
        obs = {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': pickle.dumps(self.state, protocol=pickle.HIGHEST_PROTOCOL),
            'current_player_index': self.current_player_index,
            'to_play': int(self.current_player if self.battle_mode == 'self_play_mode' else -1),
        }

        return BaseEnvTimestep(obs, np.array(rew, dtype=np.float32), done, info)

    # --------- Simulation helpers for AlphaZero MCTS ---------
    def simulate_action(self, action: int) -> 'SplendorLightZeroEnv':
        if action not in self.legal_actions:
            raise ValueError(f"action {action} is not legal")
        next_env = copy.deepcopy(self)
        next_env.state = apply_action(copy.deepcopy(self.state), int(action))
        next_env._current_player = int(next_env.state.to_play)
        # keep other fields
        return next_env

    def simulate_action_v2(self, board: Any, start_player_index: int, action: int) -> Tuple[bytes, List[int]]:
        # board is the serialized state we provided in obs['board']
        state = pickle.loads(board) if isinstance(board, (bytes, bytearray)) else copy.deepcopy(board)
        if action not in [i for i in range(TOTAL_ACTIONS) if legal_moves(state)[i] == 1]:
            raise ValueError(f"action {action} is not legal for given board")
        new_state = apply_action(state, int(action))
        new_board = pickle.dumps(new_state, protocol=pickle.HIGHEST_PROTOCOL)
        new_legal = [i for i in range(TOTAL_ACTIONS) if legal_moves(new_state)[i] == 1]
        return new_board, new_legal

    def get_done_winner(self) -> Tuple[bool, int]:
        done = bool(is_terminal(self.state))
        if not done:
            return False, -1
        w = winner(self.state)
        if w is None:
            return True, -1
        # Map internal {0,1} to {1,2} as used by LightZero board games
        return True, (1 if int(w) == 0 else 2)

    # --------- Batch env config helpers (collector/evaluator) ---------
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return 'LightZero Splendor Env'

    def close(self) -> None:
        pass

    # DI-engine BaseEnv requirement
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = int(seed)
        self._dynamic_seed = bool(dynamic_seed)
        np.random.seed(self._seed)


