import os, sys
from easydict import EasyDict

# ensure repo root is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 16
n_episode = 16
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

splendor_alphazero_config = dict(
    exp_name=f'data_az/splendor_alphazero_sp_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False,),
        render_mode=None,
        alphazero_mcts_ctree=mcts_ctree,
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        simulation_env_id='splendor',
        simulation_env_config_type='self_play',
        model=dict(
            model_type='SplendorAlphaZeroModel',
            observation_shape=(1, 15, 15),  # flattened internally
            action_space_size=45,
            policy_hidden_sizes=(512, 256),
            value_hidden_sizes=(512, 256),
        ),
        cuda=True,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        manual_temperature_decay=True,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

splendor_alphazero_config = EasyDict(splendor_alphazero_config)
main_config = splendor_alphazero_config

splendor_alphazero_create_config = dict(
    env=dict(
        type='splendor',
        import_names=['zoo.board_games.splendor.envs.splendor_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero_splendor',
        import_names=['zoo.board_games.splendor.policy.splendor_alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    ),
)

splendor_alphazero_create_config = EasyDict(splendor_alphazero_create_config)
create_config = splendor_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)


