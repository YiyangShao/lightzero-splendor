import argparse
import numpy as np
import torch

from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper, random_opponent
from splendor_gym.engine.encode import OBSERVATION_DIM, TOTAL_ACTIONS
from ppo_splendor import ActorCritic, masked_categorical


def load_agent(path: str, device: str = "cpu"):
	agent = ActorCritic(OBSERVATION_DIM, TOTAL_ACTIONS)
	agent.load_state_dict(torch.load(path, map_location=device, weights_only=True))
	agent.eval()
	return agent.to(device)


def ppo_policy(agent, device="cpu"):
	@torch.no_grad()
	def _policy(obs, info):
		obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
		mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
		logits = agent.actor(obs_t)
		probs = masked_categorical(logits, mask)
		action = probs.sample().cpu().numpy()[0]
		return int(action)
	return _policy


def eval_games(num_games: int, model_path: str | None):
	wins = 0
	draws = 0
	losses = 0
	turns = []
	if model_path is None:
		# actor that picks first legal
		def first_legal(obs, info):
			legal = np.flatnonzero(info["action_mask"])
			return int(legal[0]) if len(legal) else 0
		policy = first_legal
	else:
		agent = load_agent(model_path)
		policy = ppo_policy(agent)
	for g in range(num_games):
		env = SplendorEnv(num_players=2)
		env = SelfPlayWrapper(env, opponent_policy=random_opponent)
		obs, info = env.reset(seed=1000 + g)
		term = False
		reward = 0.0
		while not term:
			a = policy(obs, info)
			obs, reward, term, trunc, info = env.step(a)
			if term or trunc:
				break
		# collect turns from base env
		base = env.env
		turns.append(base.state.turn_count)
		if reward > 0:
			wins += 1
		elif reward < 0:
			losses += 1
		else:
			draws += 1
	avg_turns = float(np.mean(turns)) if turns else 0.0
	print(f"vs Random — games={num_games} win={wins} draw={draws} loss={losses} winrate={wins/num_games:.2%} avg_turns={avg_turns:.2f}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--games", type=int, default=200)
	parser.add_argument("--model", type=str, default="runs/ppo_splendor.pt")
	args = parser.parse_args()
	eval_games(args.games, args.model if args.model else None)


if __name__ == "__main__":
	main() 