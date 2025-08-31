import argparse
import numpy as np
import torch

from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper
from splendor_gym.engine.encode import OBSERVATION_DIM, TOTAL_ACTIONS
from ppo_splendor import ActorCritic, masked_categorical
from splendor_gym.strategies.noble_strategy import noble_policy


def load_agent(path: str, device: str = "cpu"):
	agent = ActorCritic(OBSERVATION_DIM, TOTAL_ACTIONS)
	agent.load_state_dict(torch.load(path, map_location=device))
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


def noble_opponent(obs, info):
	return int(noble_policy(obs, info))


def eval_games(num_games: int, model_path: str):
	wins = draws = losses = 0
	turns = []
	for g in range(num_games):
		env = SplendorEnv(num_players=2)
		env = SelfPlayWrapper(env, opponent_policy=noble_opponent)
		obs, info = env.reset(seed=2000 + g)
		agent = load_agent(model_path)
		policy = ppo_policy(agent)
		term = False
		while True:
			a = policy(obs, info)
			obs, r, term, trunc, info = env.step(a)
			if term or trunc:
				if r > 0: wins += 1
				elif r < 0: losses += 1
				else: draws += 1
				break
		base = env.env
		turns.append(base.state.turn_count)
	avg_turns = float(np.mean(turns)) if turns else 0.0
	print(f"vs NobleStrategy — games={num_games} win={wins} draw={draws} loss={losses} winrate={wins/num_games:.2%} avg_turns={avg_turns:.2f}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--games", type=int, default=200)
	parser.add_argument("--model", type=str, default="runs/ppo_splendor.pt")
	args = parser.parse_args()
	eval_games(args.games, args.model)


if __name__ == "__main__":
	main() 