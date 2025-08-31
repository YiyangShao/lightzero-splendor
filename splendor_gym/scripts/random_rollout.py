import argparse
import numpy as np

from splendor_gym.envs import SplendorEnv


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--episodes", type=int, default=3)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	env = SplendorEnv(num_players=2)
	wins = 0
	for ep in range(args.episodes):
		obs, info = env.reset(seed=args.seed + ep)
		done = False
		steps = 0
		while not done and steps < 500:
			mask = info["action_mask"]
			if mask.sum() == 0:
				break
			action = int(np.random.choice(np.flatnonzero(mask)))
			obs, reward, terminated, truncated, info = env.step(action)
			steps += 1
			done = terminated or truncated
		print(f"Episode {ep}: steps={steps} reward={reward}")
		if reward > 0:
			wins += 1
	print(f"Wins: {wins}/{args.episodes}")


if __name__ == "__main__":
	main() 