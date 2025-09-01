from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from typing import Dict

import numpy as np
import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.torch_utils import to_tensor
from ding.policy import create_policy

from lzero.entry import eval_alphazero


def decode_action(a: int) -> str:
    # Minimal decoding for readability; matches splendor_gym.engine.encode layout
    from splendor_gym.engine.encode import (
        TAKE3_OFFSET, TAKE3_COUNT,
        TAKE2_OFFSET, TAKE2_COUNT,
        BUY_VISIBLE_OFFSET, BUY_VISIBLE_COUNT,
        RESERVE_VISIBLE_OFFSET, RESERVE_VISIBLE_COUNT,
        RESERVE_BLIND_OFFSET, BUY_RESERVED_OFFSET,
    )
    if BUY_VISIBLE_OFFSET <= a < BUY_VISIBLE_OFFSET + BUY_VISIBLE_COUNT:
        rel = a - BUY_VISIBLE_OFFSET
        tier = rel // 4 + 1
        slot = rel % 4
        return f"BUY_VISIBLE(t{tier},s{slot})"
    if TAKE3_OFFSET <= a < TAKE3_OFFSET + TAKE3_COUNT:
        return f"TAKE3({a-TAKE3_OFFSET})"
    if TAKE2_OFFSET <= a < TAKE2_OFFSET + TAKE2_COUNT:
        color = a - TAKE2_OFFSET
        return f"TAKE2(c{color})"
    if RESERVE_VISIBLE_OFFSET <= a < RESERVE_VISIBLE_OFFSET + RESERVE_VISIBLE_COUNT:
        rel = a - RESERVE_VISIBLE_OFFSET
        tier = rel // 4 + 1
        slot = rel % 4
        return f"RESERVE_VISIBLE(t{tier},s{slot})"
    if a == RESERVE_BLIND_OFFSET:
        return "RESERVE_BLIND(t1)"
    if a == RESERVE_BLIND_OFFSET + 1:
        return "RESERVE_BLIND(t2)"
    if a == RESERVE_BLIND_OFFSET + 2:
        return "RESERVE_BLIND(t3)"
    if BUY_RESERVED_OFFSET <= a < BUY_RESERVED_OFFSET + 3:
        return f"BUY_RESERVED(slot{a - BUY_RESERVED_OFFSET})"
    return f"ACTION({a})"


def run_eval_stats(ckpt_path: str, opponent: str, episodes: int) -> None:
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config as main_config,
        splendor_alphazero_create_config as create_config,
    )
    # Adjust env for eval-mode single-process evals via eval entry
    main_config.env.battle_mode = 'eval_mode'
    main_config.env.evaluator_env_num = max(1, main_config.env.get('evaluator_env_num', 1))
    main_config.env.n_evaluator_episode = main_config.env.evaluator_env_num
    # Set bot type
    main_config.env['bot_action_type'] = 'random' if opponent == 'random' else 'basic'
    # Use base manager for simplicity
    create_config.env_manager.type = 'base'
    # Run eval for the requested number of episodes
    _mean, _returns = eval_alphazero(
        [main_config, create_config],
        seed=0,
        model_path=ckpt_path,
        num_episodes_each_seed=episodes,
        print_seed_details=True,
    )
    return None


def run_single_game_log(ckpt_path: str, opponent: str, log_path: str) -> None:
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config as cfg,
        splendor_alphazero_create_config as create_cfg,
    )
    cfg.env.battle_mode = 'eval_mode'
    cfg.env['bot_action_type'] = 'random' if opponent == 'random' else 'basic'
    # base manager with 1 env for a simple loop
    create_cfg.env_manager.type = 'base'
    cfg.policy.eval_freq = int(1e12)

    compiled = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
    env_fn, _, _ = get_vec_env_setting(compiled.env)
    env = create_env_manager(create_cfg.env_manager, [partial(env_fn, cfg=compiled.env)])
    env.seed(compiled.seed, dynamic_seed=False)
    env.launch()

    policy = create_policy(compiled.policy, enable_field=['eval'])
    # load ckpt
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=compiled.policy.device)
        policy.eval_mode.load_state_dict(sd)
    policy.eval_mode.reset()

    # open log
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"opponent={opponent}\n")
        env.reset()
        policy.eval_mode.reset()
        done_envs: Dict[int, bool] = {}
        while True:
            obs = env.ready_obs
            out = policy.eval_mode.forward(obs)
            actions = {eid: out[eid]['action'] for eid in out}
            # decode first env's action for logging
            if 0 in actions:
                f.write(decode_action(int(actions[0])) + "\n")
            ts = env.step(actions)
            ts = to_tensor(ts, dtype=torch.float32)
            if 0 in ts and ts[0].done:
                f.write(f"final_return={ts[0].info.get('eval_episode_return', 0.0)}\n")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--opponent', type=str, choices=['random', 'basic'], default='random')
    parser.add_argument('--log-txt', type=str, default='data_az/splendor_eval_sample.txt')
    args = parser.parse_args()

    # stats eval
    run_eval_stats(args.ckpt, args.opponent, args.episodes)
    # single game log
    run_single_game_log(args.ckpt, args.opponent, args.log_txt)


if __name__ == '__main__':
    # ensure repo root is in path
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    main()


