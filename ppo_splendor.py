 import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from splendor_gym.engine.encode import OBSERVATION_DIM, TOTAL_ACTIONS
from splendor_gym.scripts.eval_suite import (
	model_greedy_policy_from,
	random_opponent,
	greedy_opponent_v1,
	basic_priority_opponent,
)
from training_utils import (
	TrainingLogger,
	CheckpointManager,
	make_env,
	run_evaluation_suite,
	frozen_policy_from,
	linear_lr_schedule,
)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
	# mask shape [B, A]; ensure at least one legal per row
	illegal = (mask < 0.5)
	masked_logits = logits.clone()
	# rows with at least one legal: apply -inf to illegal
	rows_any_legal = (~illegal).any(dim=1)
	if rows_any_legal.any():
		masked_logits[rows_any_legal] = masked_logits[rows_any_legal].masked_fill(
			illegal[rows_any_legal], float("-inf")
		)
	# rows with no legal actions: leave logits unmasked to avoid NaNs; env will handle no-legal at step
	return Categorical(logits=masked_logits)


class ActorCritic(nn.Module):
	def __init__(self, obs_dim: int, act_dim: int):
		super().__init__()
		self.critic = nn.Sequential(
			nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1)
		)
		self.actor = nn.Sequential(
			nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, act_dim)
		)

	def get_value(self, x: torch.Tensor) -> torch.Tensor:
		return self.critic(x)

	def get_action_and_value(self, x: torch.Tensor, mask: torch.Tensor, action: torch.Tensor | None = None):
		logits = self.actor(x)
		probs = masked_categorical(logits, mask)
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action), probs.entropy().mean(), self.critic(x)







def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-timesteps", type=int, default=1_000_000)
	parser.add_argument("--num-envs", type=int, default=16)
	parser.add_argument("--num-steps", type=int, default=128)
	parser.add_argument("--gamma", type=float, default=0.999)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--lr", type=float, default=2.5e-4)
	parser.add_argument("--ent-coef", type=float, default=0.03)
	parser.add_argument("--vf-coef", type=float, default=0.5)
	parser.add_argument("--clip-coef", type=float, default=0.2)
	parser.add_argument("--update-epochs", type=int, default=4)
	parser.add_argument("--minibatch-size", type=int, default=256)
	parser.add_argument("--save-path", type=str, default="runs/ppo_splendor.pt")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--track", action="store_true", help="log to tensorboard")
	parser.add_argument("--log-dir", type=str, default="runs/ppo_splendor")
	parser.add_argument("--eval-every-updates", type=int, default=10)
	parser.add_argument("--eval-games", type=int, default=400)
	parser.add_argument("--lr-anneal", action="store_true")
	parser.add_argument("--train-opponent", type=str, default="basic", choices=["random", "greedy_v1", "basic"], help="Opponent policy for rollouts")
	parser.add_argument("--self-play", dest="self_play", action="store_true", default=True, help="Enable opponent pool self-play for training rollouts (default)")
	parser.add_argument("--no-self-play", dest="self_play", action="store_false", help="Disable self-play; use --train-opponent for static opponent")
	parser.add_argument("--pool-size", type=int, default=12)
	parser.add_argument("--snapshot-every-updates", type=int, default=10)
	parser.add_argument("--p-current", type=float, default=0.25, help="Probability to face current policy in self-play")
	parser.add_argument("--target-kl", type=float, default=0.02)
	parser.add_argument("--vclip", type=float, default=0.2)
	parser.add_argument("--ent-coef-final", type=float, default=0.01)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Device checker output
	try:
		print(f"[device] using: {device}")
		if device.type == "cuda":
			print(f"[device] cuda_available=True name={torch.cuda.get_device_name(0)} count={torch.cuda.device_count()} capability={torch.cuda.get_device_capability(0)}")
		else:
			mps_avail = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
			print(f"[device] mps_available={mps_avail}")
	except Exception as _e:
		print(f"[device] info error: {_e}")

	# Seeding
	rng = np.random.RandomState(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)



	num_envs = args.num_envs
	num_steps = args.num_steps
	# Select training opponent policy
	if args.train_opponent == "random":
		train_opp = random_opponent
	elif args.train_opponent == "greedy_v1":
		train_opp = greedy_opponent_v1
	elif args.train_opponent == "basic":
		train_opp = basic_priority_opponent
	else:
		raise ValueError("Unsupported --train-opponent")

	# Self-play setup
	pool: list[dict] = []  # Opponent pool of state_dict snapshots
	
	def opponent_supplier():
		# Sample current policy with probability p_current
		if (len(pool) == 0) or (np.random.rand() < args.p_current):
			return model_greedy_policy_from(agent, device=device)
		# Pick random frozen snapshot
		idx = int(np.random.randint(0, len(pool)))
		return frozen_policy_from(pool[idx], ActorCritic, OBSERVATION_DIM, TOTAL_ACTIONS, device)

	# Create agent and optimizer BEFORE creating envs (supplier uses agent)
	agent = ActorCritic(OBSERVATION_DIM, TOTAL_ACTIONS).to(device)
	optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

	# Create environments
	if args.self_play:
		envs = gym.vector.SyncVectorEnv([
			make_env(int(rng.randint(1e9)), opponent_supplier=opponent_supplier, random_starts=True) 
			for _ in range(num_envs)
		])
	else:
		envs = gym.vector.SyncVectorEnv([
			make_env(int(rng.randint(1e9)), opponent_policy=train_opp)
			for _ in range(num_envs)
		])

	obs = np.zeros((num_envs, OBSERVATION_DIM), dtype=np.int32)
	masks = np.zeros((num_envs, TOTAL_ACTIONS), dtype=np.int8)
	obs_t, info = envs.reset()
	obs[:] = obs_t
	if isinstance(info, dict) and "action_mask" in info:
		am = info["action_mask"]
		masks[:] = am if isinstance(am, np.ndarray) and am.shape == (num_envs, TOTAL_ACTIONS) else np.stack([am[i] for i in range(num_envs)], axis=0)
	else:
		for i in range(num_envs):
			masks[i] = info[i]["action_mask"]

	# Training setup
	logger = TrainingLogger(args.log_dir, track=args.track)
	checkpoint_manager = CheckpointManager(args.log_dir, logger.run_start_ts)
	
	num_updates = args.total_timesteps // (num_envs * num_steps)
	global_step = 0

	# Initial evaluation for baseline
	print("Running initial evaluation...")
	results = run_evaluation_suite(agent, device, rng, args.eval_games, 0)
	logger.log_evaluation_results(results, global_step)
	logger.update_history(global_step, results, args.lr, 0.0, 0.0, 0.0)
	logger.create_summary_plot(global_step)

	for update in range(num_updates):
		# LR annealing
		if args.lr_anneal:
			progress = 1.0 - (update / max(1, num_updates - 1))
			lr = linear_lr_schedule(args.lr, progress)
			for pg in optimizer.param_groups:
				pg["lr"] = lr

		# Rollout storage
		obs_buf = []
		masks_buf = []
		actions_buf = []
		logprobs_buf = []
		rewards_buf = []
		values_buf = []
		terminals_buf = []

		for step in range(num_steps):
			# Convert observation to tensor (no normalization)
			obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
			mask_tensor = torch.tensor(masks, dtype=torch.float32, device=device)
			with torch.no_grad():
				action, logprob, entropy, value = agent.get_action_and_value(obs_tensor, mask_tensor)
			actions = action.cpu().numpy()
			next_obs, rewards, terms, truncs, infos = envs.step(actions)
			if isinstance(infos, dict) and "action_mask" in infos:
				am = infos["action_mask"]
				next_masks = am if isinstance(am, np.ndarray) and am.shape == (num_envs, TOTAL_ACTIONS) else np.stack([am[i] for i in range(num_envs)], axis=0)
			else:
				next_masks = np.zeros_like(masks)
				for i in range(num_envs):
					next_masks[i] = infos[i]["action_mask"]

			obs_buf.append(obs.copy())
			masks_buf.append(masks.copy())
			actions_buf.append(actions.copy())
			logprobs_buf.append(logprob.detach().cpu().numpy())
			values_buf.append(value.detach().cpu().numpy())
			rewards_buf.append(rewards.copy())
			terminals_buf.append(terms.copy())

			obs = next_obs
			masks = next_masks
			global_step += num_envs

		# Compute returns/advantages (GAE)
		with torch.no_grad():
			obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
			last_values = agent.get_value(obs_tensor).detach().cpu().numpy().squeeze(-1)

		rewards_arr = np.array(rewards_buf)  # [T, N]
		values_arr = np.array(values_buf).squeeze(-1)  # [T, N]
		term_arr = np.array(terminals_buf)  # [T, N]
		advantages = np.zeros_like(rewards_arr)
		lastgaelam = np.zeros(num_envs)
		for t in reversed(range(num_steps)):
			nextnonterminal = 1.0 - term_arr[t]
			nextvalues = last_values if t == num_steps - 1 else values_arr[t + 1]
			delta = rewards_arr[t] + args.gamma * nextvalues * nextnonterminal - values_arr[t]
			advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
		returns = advantages + values_arr

		# Flatten
		obs_flat = np.concatenate(obs_buf, axis=0)
		b_obs = torch.tensor(obs_flat, dtype=torch.float32, device=device)
		b_masks = torch.tensor(np.concatenate(masks_buf, axis=0), dtype=torch.float32, device=device)
		b_actions = torch.tensor(np.concatenate(actions_buf, axis=0), dtype=torch.int64, device=device)
		b_logprobs = torch.tensor(np.concatenate(logprobs_buf, axis=0), dtype=torch.float32, device=device)
		b_returns = torch.tensor(np.concatenate(returns, axis=0), dtype=torch.float32, device=device)
		b_values = torch.tensor(np.concatenate(values_arr, axis=0), dtype=torch.float32, device=device)
		b_advantages = torch.tensor(np.concatenate(advantages, axis=0), dtype=torch.float32, device=device)
		b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

		# PPO update
		batch_size = b_obs.shape[0]
		minibatch_size = min(args.minibatch_size, batch_size)
		# Entropy schedule
		progress_updates = update / max(1, num_updates - 1)
		ent_coef_now = args.ent_coef + (args.ent_coef_final - args.ent_coef) * progress_updates
		for epoch in range(args.update_epochs):
			idxs = torch.randperm(batch_size, device=device)
			for start in range(0, batch_size, minibatch_size):
				mb_idx = idxs[start:start + minibatch_size]
				new_action, new_logprob, entropy, new_value = agent.get_action_and_value(
					b_obs[mb_idx], b_masks[mb_idx], b_actions[mb_idx]
				)
				ratio = (new_logprob - b_logprobs[mb_idx]).exp()
				mb_adv = b_advantages[mb_idx]
				clip_adv = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_adv
				policy_loss = -torch.min(ratio * mb_adv, clip_adv).mean()
				# Value clipping
				v_pred = new_value.squeeze(-1)
				v_pred_clipped = b_values[mb_idx] + torch.clamp(v_pred - b_values[mb_idx], -args.vclip, args.vclip)
				v_loss_unclipped = (v_pred - b_returns[mb_idx]).pow(2)
				v_loss_clipped = (v_pred_clipped - b_returns[mb_idx]).pow(2)
				value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
				entropy_loss = -entropy.mean()
				loss = policy_loss + args.vf_coef * value_loss + ent_coef_now * (-entropy_loss)

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
				optimizer.step()

				# Early stop by target KL
				approx_kl = (b_logprobs[mb_idx] - new_logprob).mean().detach().cpu().item()
				if args.target_kl > 0 and approx_kl > args.target_kl:
					break

		# Save checkpoints each update
		checkpoint_manager.save_checkpoint(agent)

		# Snapshot pool maintenance
		if args.self_play and (update + 1) % max(1, args.snapshot_every_updates) == 0:
			pool.append(agent.state_dict())
			if len(pool) > args.pool_size:
				pool.pop(0)

		# Log training metrics
		logger.log_training_metrics(
			global_step, 
			optimizer.param_groups[0]["lr"],
			policy_loss.item(),
			value_loss.item(), 
			(-entropy_loss).item(),
			approx_kl
		)

		# Periodic evaluation
		if (update + 1) % args.eval_every_updates == 0:
			print(f"Running evaluation at update {update + 1}...")
			results = run_evaluation_suite(agent, device, rng, args.eval_games, update)
			
			# Log results and update history
			logger.log_evaluation_results(results, global_step)
			logger.update_history(
				global_step, results, 
				optimizer.param_groups[0]["lr"],
				policy_loss.item(), 
				value_loss.item(),
				(-entropy_loss).item()
			)
			
			# Create and save plots
			logger.create_summary_plot(global_step)
			
			# Save evaluation checkpoint
			latest_path, ts_path = checkpoint_manager.save_checkpoint(agent)
			print(f"[eval] saved {latest_path} and {ts_path}")

		if (update + 1) % 10 == 0:
			print(f"update={update+1}/{num_updates}")

	# Save final checkpoint
	latest_path, ts_path = checkpoint_manager.save_checkpoint(agent)
	print(f"Saved final {latest_path} and {ts_path}")


if __name__ == "__main__":
	main() 
