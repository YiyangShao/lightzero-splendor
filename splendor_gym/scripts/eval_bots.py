import argparse
import itertools
import json
import numpy as np

from splendor_gym.scripts.eval_suite import (
	greedy_opponent_v1,
	greedy_opponent_v2_factory,
	basic_priority_opponent,
	eval_vs_opponent,
)
from splendor_gym.wrappers.selfplay import random_opponent
from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper


def get_policy(name: str):
	name = name.lower()
	if name in ("random", "rand"):
		return random_opponent
	if name in ("greedy_v1", "greedy1", "greedy"):
		return greedy_opponent_v1
	if name in ("basic_priority", "basic"):
		return basic_priority_opponent
	if name in ("greedy_v2", "greedy2"):
		# env_ref-aware variant not wired here; fall back to env-agnostic form
		return greedy_opponent_v2_factory(None)
	raise ValueError(f"Unknown policy name: {name}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--games", type=int, default=200)
	parser.add_argument("--pairs", type=str, nargs="*", default=[
		"random:greedy_v1",
		"random:basic_priority",
		"greedy_v1:basic_priority",
		"basic_priority:greedy_v1",
		"greedy_v1:random",
		"basic_priority:random",
	])
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--json", type=str, default="")
	args = parser.parse_args()

	results = {}
	for pair in args.pairs:
		left, right = pair.split(":")
		pol_left = get_policy(left)
		pol_right = get_policy(right)

		def make_env():
			# Build a fresh env instance each game; eval_vs_opponent will reset per episode
			env = SplendorEnv(num_players=2)
			env = SelfPlayWrapper(env, opponent_policy=pol_right)
			return env
		res = eval_vs_opponent(make_env, pol_left, n_games=args.games, seed=args.seed)
		key = f"{left}_vs_{right}"
		results[key] = {
			"n": res["n"],
			"wins": res["wins"],
			"losses": res["losses"],
			"draws": res["draws"],
			"win_rate": res["win_rate"],
			"win_rate_ci95": res["win_rate_ci95"],
			"avg_turns": res["avg_turns"],
			"avg_prestige": res["avg_prestige"],
		}
		print(f"{key}: wr={res['win_rate']:.3f}±{res['win_rate_ci95']:.3f}, avg_turns={res['avg_turns']:.2f}, draws={res['draws']}")

	if args.json:
		with open(args.json, "w") as f:
			json.dump(results, f, indent=2)


if __name__ == "__main__":
	main()
