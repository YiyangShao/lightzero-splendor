import argparse
import os
import torch

from splendor_gym.scripts.eval_suite import eval_vs_opponent, basic_priority_opponent, model_greedy_policy_from
from splendor_gym.engine.encode import OBSERVATION_DIM, TOTAL_ACTIONS
from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper


def load_model(path: str, device: str = "cpu"):
	from ppo_splendor import ActorCritic
	model = ActorCritic(OBSERVATION_DIM, TOTAL_ACTIONS).to(device)
	state = torch.load(path, map_location=device, weights_only=True)
	model.load_state_dict(state)
	model.eval()
	return model


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt", type=str, default="runs/ppo_splendor/ppo_splendor_latest.pt")
	parser.add_argument("--games", type=int, default=200)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	if not os.path.exists(args.ckpt):
		raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

	device = "cpu"
	model = load_model(args.ckpt, device=device)
	policy = model_greedy_policy_from(model, device=device)

	def make_env():
		env = SplendorEnv(num_players=2)
		return SelfPlayWrapper(env, opponent_policy=basic_priority_opponent)

	res = eval_vs_opponent(make_env, policy, n_games=args.games, seed=args.seed)
	print({k: v for k, v in res.items() if k in ("n","wins","losses","draws","win_rate","win_rate_ci95","avg_turns","avg_prestige")})


if __name__ == "__main__":
	main()
