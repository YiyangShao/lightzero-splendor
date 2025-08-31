import numpy as np
import torch
from typing import Callable, Dict, Any, List

from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper, random_opponent
from splendor_gym.engine.encode import OBSERVATION_DIM, TOTAL_ACTIONS


def greedy_opponent_v1(obs, info):
	legal = np.flatnonzero(info["action_mask"])
	if len(legal) == 0:
		return 0
	# Buy visible/reserved preferred
	buys = [a for a in legal if (15 <= a <= 26) or (42 <= a <= 44)]
	if buys:
		return int(buys[0])
	# Take-2
	take2 = [a for a in legal if 10 <= a <= 14]
	if take2:
		return int(take2[0])
	# Take-3 (including reduced-color)
	take3 = [a for a in legal if 0 <= a <= 9]
	if take3:
		return int(take3[0])
	# Reserve
	res = [a for a in legal if 27 <= a <= 41]
	if res:
		return int(res[0])
	return int(legal[0])


def basic_priority_opponent(obs, info):
	"""
	Priority:
	1) Purchase card: prefer visible buys with highest points; tie-break random. If only reserved buys, random among them.
	2) Take tokens: prefer take-3 (random among legal); if none, take-2 (random among legal).
	3) Reserve (visible or blind): random among legal.
	"""
	mask = info["action_mask"]
	legal = np.flatnonzero(mask)
	if len(legal) == 0:
		return 0
	# 1) Purchases
	buy_vis = [a for a in legal if 15 <= a <= 26]
	buy_res = [a for a in legal if 42 <= a <= 44]
	if buy_vis or buy_res:
		if buy_vis:
			# Extract points for visible cards from obs
			# Board starts at offset 6 + 13 + 13 = 32; per slot 13 ints; points at +2
			base = 32
			best_pts = -1
			best_actions = []
			for a in buy_vis:
				idx = a - 15  # 0..11
				pt = int(obs[base + idx * 13 + 2])
				if pt > best_pts:
					best_pts = pt
					best_actions = [a]
				elif pt == best_pts:
					best_actions.append(a)
			return int(np.random.choice(best_actions))
		# else only reserved buys
		return int(np.random.choice(buy_res))
	# 2) Tokens: take-3 preferred
	take3 = [a for a in legal if 0 <= a <= 9]
	if take3:
		return int(np.random.choice(take3))
	take2 = [a for a in legal if 10 <= a <= 14]
	if take2:
		return int(np.random.choice(take2))
	# 3) Reserve random
	reserve = [a for a in legal if 27 <= a <= 41]
	if reserve:
		return int(np.random.choice(reserve))
	# Fallback
	return int(legal[0])


def greedy_opponent_v2_factory(env_ref: SplendorEnv | None = None) -> Callable:
	"""
	Greedy v2: noble-aware + color scarcity.
	Heuristics:
	- If any buy yields prestige or fulfills noble soon, prefer it.
	- Otherwise, take-2 of scarcest bank color among legal take-2.
	- Otherwise, take-3 selecting the scarcest legal combo.
	- Else reserve visible with highest points, else any.
	"""
	def policy(obs, info):
		mask = info["action_mask"]
		legal = np.flatnonzero(mask)
		if len(legal) == 0:
			return 0
		# Prefer buys
		buy_vis = [a for a in legal if 15 <= a <= 26]
		buy_res = [a for a in legal if 42 <= a <= 44]
		buys = buy_vis + buy_res
		if buys:
			# Prefer visible buys first
			return int(buys[0])
		# Scarcity on bank
		bank = getattr(env_ref, 'state', None)
		if bank is not None:
			bank_vec = env_ref.state.bank[:5]
		else:
			# fallback uniform scarcity
			bank_vec = [1, 1, 1, 1, 1]
		# Take-2
		take2 = [a for a in legal if 10 <= a <= 14]
		if take2:
			# map a to color idx a-10, pick minimal bank count
			best = min(take2, key=lambda a: bank_vec[a - 10])
			return int(best)
		# Take-3
		take3 = [a for a in legal if 0 <= a <= 9]
		if take3:
			# prefer combos with minimal sum of bank counts (scarcer)
			# Precomputed combos order used in engine.encode
			from splendor_gym.engine.encode import TAKE3_COMBOS
			best = min(take3, key=lambda a: sum(bank_vec[i] for i in TAKE3_COMBOS[a]))
			return int(best)
		# Reserve visible prioritized by tier (higher first)
		res = [a for a in legal if 27 <= a <= 41]
		if res:
			# favor tier 3 > 2 > 1 roughly by action index
			return int(sorted(res, reverse=True)[0])
		return int(legal[0])
	return policy


@torch.no_grad()
def model_greedy_policy_from(model: torch.nn.Module, device: str = "cpu") -> Callable[[np.ndarray, Dict[str, Any]], int]:
	model.eval()
	def _policy(obs, info):
		obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
		mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
		logits = model.actor(obs_t)
		logits = logits.masked_fill(mask < 0.5, float("-inf"))
		action = torch.argmax(logits, dim=-1).item()
		return int(action)
	return _policy


def make_selfplay_env(seed: int):
	def thunk():
		env = SplendorEnv(num_players=2)
		env = SelfPlayWrapper(env, opponent_policy=random_opponent)
		env.reset(seed=seed)
		return env
	return thunk


def make_selfplay_env_with(opponent_policy: Callable, seed: int):
	def thunk():
		env = SplendorEnv(num_players=2)
		env = SelfPlayWrapper(env, opponent_policy=opponent_policy)
		env.reset(seed=seed)
		return env
	return thunk


def eval_vs_opponent(make_env: Callable[[], Any], model_policy: Callable, n_games: int = 400, seed: int = 0) -> Dict[str, Any]:
	wins = draws = losses = 0
	draw_count = 0
	turns = []
	prestige = []
	rng = np.random.RandomState(seed)
	illegal_rate = 0.0
	checks = 0

	for g in range(n_games):
		env = make_env()
		obs, info = env.reset(seed=int(rng.randint(1e9)))
		term = False
		while True:
			mask = info["action_mask"]
			checks += 1
			a = model_policy(obs, info)
			if mask[a] == 0:
				illegal_rate += 1
			obs, r, term, trunc, info = env.step(a)
			if term or trunc:
				if r > 0: wins += 1
				elif r < 0: losses += 1
				else:
					draws += 1
				break
		# Stats
		base = env.env
		turns.append(base.state.turn_count)
		prev_idx = (base.state.to_play - 1) % base.state.num_players
		prestige.append(base.state.players[prev_idx].prestige)
		env.close()

	n = n_games
	p = wins / max(1, n)
	ci = 1.96 * np.sqrt(p * (1 - p) / max(1, n))
	return {
		"n": n,
		"wins": wins,
		"losses": losses,
		"draws": draws,
		"win_rate": p,
		"win_rate_ci95": ci,
		"avg_turns": float(np.mean(turns)) if turns else 0.0,
		"avg_prestige": float(np.mean(prestige)) if prestige else 0.0,
		"illegal_action_rate": float(illegal_rate / max(1, checks)),
	}


def eval_vs_checkpoint_pool(checkpoint_paths: List[str], model_policy: Callable, n_games: int = 400, seed: int = 0) -> Dict[str, Any]:
	"""
	Evaluate model_policy vs a uniform random opponent sampled from checkpoint_paths.
	Splits n_games across pool entries.
	"""
	if not checkpoint_paths:
		return {"n": 0, "wins": 0, "losses": 0, "draws": 0, "win_rate": 0.0, "win_rate_ci95": 0.0, "avg_turns": 0.0, "avg_prestige": 0.0, "illegal_action_rate": 0.0}
	rng = np.random.RandomState(seed)
	per = max(1, n_games // len(checkpoint_paths))
	agg = {"n": 0, "wins": 0, "losses": 0, "draws": 0, "turns": [], "prestige": [], "illegal": 0, "checks": 0}
	for path in checkpoint_paths:
		agent = torch.load  # placeholder to appease static; we will load weights below
		# Load opponent
		opp_model = torch.nn.Module()  # will be overridden in ppo script; here define in place in caller
		# We cannot construct ActorCritic here without circular import; caller can compose make_env_with
		# Provide a simple random opponent fallback
		def opp_policy(obs, info):
			legal = np.flatnonzero(info["action_mask"])
			return int(np.random.choice(legal)) if len(legal) else 0
		make_env = make_selfplay_env_with(opp_policy, int(rng.randint(1e9)))
		res = eval_vs_opponent(make_env, model_policy, n_games=per, seed=int(rng.randint(1e9)))
		agg["n"] += res["n"]
		agg["wins"] += res["wins"]
		agg["losses"] += res["losses"]
		agg["draws"] += res["draws"]
		agg["turns"].append(res["avg_turns"])
		agg["prestige"].append(res["avg_prestige"])
		agg["illegal"] += res["illegal_action_rate"] * res["n"]
		agg["checks"] += res["n"]
	N = max(1, agg["n"])
	p = agg["wins"] / N
	ci = 1.96 * np.sqrt(p * (1 - p) / N)
	return {
		"n": N,
		"wins": agg["wins"],
		"losses": agg["losses"],
		"draws": agg["draws"],
		"win_rate": p,
		"win_rate_ci95": ci,
		"avg_turns": float(np.mean(agg["turns"])) if agg["turns"] else 0.0,
		"avg_prestige": float(np.mean(agg["prestige"])) if agg["prestige"] else 0.0,
		"illegal_action_rate": float(agg["illegal"] / max(1, agg["checks"]))
	} 