from __future__ import annotations

from ding.utils import POLICY_REGISTRY
from lzero.policy.alphazero import AlphaZeroPolicy


@POLICY_REGISTRY.register('alphazero_splendor')
class SplendorAlphaZeroPolicy(AlphaZeroPolicy):
    """AlphaZero policy with simulation env set to Splendor."""

    def _get_simulation_env(self):
        from zoo.board_games.splendor.envs.splendor_lightzero_env import SplendorLightZeroEnv
        if self._cfg.simulation_env_config_type == 'self_play':
            from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
        else:
            raise NotImplementedError
        self.simulate_env = SplendorLightZeroEnv(splendor_alphazero_config.env)

    def default_model(self):
        # Use our Splendor-specific MLP model
        return 'SplendorAlphaZeroModel', ['zoo.board_games.splendor.model.splendor_alphazero_model']

    def _init_collect(self) -> None:
        super()._init_collect()
        # Ensure ptree MCTS returns a triple to match AlphaZeroPolicy expectations
        if not self._cfg.mcts_ctree:
            orig = self._collect_mcts.get_next_action

            def _wrapper(*args, **kwargs):
                action, probs = orig(*args, **kwargs)
                return action, probs, None

            self._collect_mcts.get_next_action = _wrapper  # type: ignore[attr-defined]

    def _init_eval(self) -> None:
        super()._init_eval()
        if not self._cfg.mcts_ctree:
            orig = self._eval_mcts.get_next_action

            def _wrapper(*args, **kwargs):
                action, probs = orig(*args, **kwargs)
                return action, probs, None

            self._eval_mcts.get_next_action = _wrapper  # type: ignore[attr-defined]


