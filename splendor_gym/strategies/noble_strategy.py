import numpy as np
from typing import Dict, Any, List, Tuple

from splendor_gym.engine.state import STANDARD_COLORS
from splendor_gym.engine.encode import TAKE3_COMBOS


def parse_obs(obs: np.ndarray) -> Dict[str, Any]:
	# Layout: bank(6), me:tokens(6),bonuses(5),prestige(1),reserved_count(1), opp:tokens(6),bonuses(5),prestige(1),reserved_count(1), board 12x13, nobles 5x6, decks(3), turn, to_play, round_flag
	i = 0
	bank = obs[i:i+6]; i += 6
	me_tokens = obs[i:i+6]; i += 6
	me_bonuses = obs[i:i+5]; i += 5
	me_prestige = int(obs[i]); i += 1
	me_reserved = int(obs[i]); i += 1
	opp_tokens = obs[i:i+6]; i += 6
	opp_bonuses = obs[i:i+5]; i += 5
	opp_prestige = int(obs[i]); i += 1
	opp_reserved = int(obs[i]); i += 1
	# Board cards (12)
	cards = []
	for tier in (1, 2, 3):
		for slot in range(4):
			present = int(obs[i]); i += 1
			if present == 0:
				# skip tier, points, color_onehot(5), cost(5)
				i += 1 + 1 + 5 + 5
				cards.append(None)
				continue
			tier_val = int(obs[i]); i += 1
			points = int(obs[i]); i += 1
			color_onehot = obs[i:i+5]; i += 5
			cost_arr = obs[i:i+5]; i += 5
			color_idx = int(np.argmax(color_onehot))
			cards.append({
				"tier": tier_val,
				"points": points,
				"color_idx": color_idx,
				"cost": {STANDARD_COLORS[k]: int(cost_arr[k]) for k in range(5)}
			})
	# Nobles up to 5
	nobles = []
	for _ in range(5):
		present = int(obs[i]); i += 1
		req = {STANDARD_COLORS[k]: int(obs[i+k]) for k in range(5)}; i += 5
		if present:
			nobles.append(req)
	decks = obs[i:i+3]; i += 3
	turn = int(obs[i]); i += 1
	to_play = int(obs[i]); i += 1
	round_flag = int(obs[i]); i += 1
	return {
		"bank": bank,
		"me_tokens": me_tokens,
		"me_bonuses": me_bonuses,
		"me_prestige": me_prestige,
		"me_reserved": me_reserved,
		"opp_tokens": opp_tokens,
		"opp_bonuses": opp_bonuses,
		"cards": cards,
		"nobles": nobles,
	}


def simulate_noble_after_purchase(bonuses: np.ndarray, card_color_idx: int, nobles: List[Dict[str, int]]) -> bool:
	b = bonuses.copy(); b[card_color_idx] += 1
	for noble in nobles:
		ok = True
		for ci, col in enumerate(STANDARD_COLORS):
			need = noble.get(col, 0)
			if b[ci] < need:
				ok = False
				break
		if ok:
			return True
	return False


def calculate_noble_value(card_color_idx: int, bonuses: np.ndarray, nobles: List[Dict[str, int]]) -> float:
	val = 0.0
	color = STANDARD_COLORS[card_color_idx]
	for noble in nobles:
		req_color = noble.get(color, 0)
		owned = bonuses[card_color_idx]
		if req_color > owned:
			# color contributes towards a noble
			val += 1.0
			# estimate total remaining cards
			cards_needed = 0
			for ci, col in enumerate(STANDARD_COLORS):
				need = noble.get(col, 0) - bonuses[ci]
				cards_needed += max(0, need)
			# bonus shaping
			if cards_needed == 1:
				val += 12
			elif cards_needed == 2:
				val += 7
			elif cards_needed == 3:
				val += 3
			elif 4 <= cards_needed <= 6:
				val += 1
	return val


def calculate_discount_value(card_color_idx: int, bonuses: np.ndarray, prestige: int) -> float:
	# prefer colors we have fewer discounts in; prefer more points when we have many cards
	same_color_cards = bonuses[card_color_idx]
	val = 2.0  # base
	val += max(0.0, 15 - 0.5 * prestige)
	val -= 2.0 * same_color_cards
	return max(0.0, val)


def calculate_token_cost_value(cost: Dict[str, int], bonuses: np.ndarray, me_tokens: np.ndarray) -> float:
	# penalize higher cost and remaining needs
	val = 0.0
	for ci, col in enumerate(STANDARD_COLORS):
		c = cost.get(col, 0)
		discount = bonuses[ci]
		tokens = me_tokens[ci]
		tokens_needed = max(0, c - discount)
		val += (-1.5 * tokens_needed - 0.5 * (tokens_needed ** 1.5))
	return val


def calculate_points_value(points: int, bonuses: np.ndarray) -> float:
	return points * (1 + 0.1 * bonuses.sum())


def evaluate_card(card: Dict[str, Any], bonuses: np.ndarray, me_tokens: np.ndarray, nobles: List[Dict[str, int]], prestige: int) -> float:
	if card is None:
		return -1e9
	nv = calculate_noble_value(card["color_idx"], bonuses, nobles)
	dv = calculate_discount_value(card["color_idx"], bonuses, prestige)
	tcv = calculate_token_cost_value(card["cost"], bonuses, me_tokens)
	pv = calculate_points_value(card["points"], bonuses)
	return nv + dv + tcv + pv


def evaluate_all_cards(cards: List[Dict[str, Any]], bonuses: np.ndarray, me_tokens: np.ndarray, nobles: List[Dict[str, int]], prestige: int) -> Dict[int, float]:
	vals = {}
	for idx, card in enumerate(cards):
		vals[idx] = evaluate_card(card, bonuses, me_tokens, nobles, prestige)
	return vals


def evaluate_tokens(card_vals: Dict[int, float], cards: List[Dict[str, Any]], bonuses: np.ndarray, me_tokens: np.ndarray) -> Dict[str, float]:
	token_values = {c: 1.0 for c in STANDARD_COLORS}
	# Emphasize colors appearing in high-valued cards where require1>0
	ranked = sorted([(idx, v) for idx, v in card_vals.items() if cards[idx] is not None], key=lambda x: -x[1])
	for rank, (idx, v) in enumerate(ranked[:10]):
		card = cards[idx]
		for ci, col in enumerate(STANDARD_COLORS):
			cost = card["cost"].get(col, 0)
			need1 = max(0, cost - bonuses[ci])
			need2 = max(0, cost - bonuses[ci] - me_tokens[ci])
			if need1 > 0:
				base = 1.0 if rank == 0 else (0.5 if rank == 1 else 0.3 if rank == 2 else 0.2)
				if rank > 3:
					base = max(0.05, 0.16 * (0.8 ** (rank - 3)))
				token_values[col] += base if need2 > 0 else base * 0.5
	return token_values


def get_defensive_value(card_idx: int, opp_card_vals: Dict[int, float]) -> float:
	vals = sorted(opp_card_vals.values(), reverse=True)
	if not vals:
		return 0.0
	top = vals[0]
	second = vals[1] if len(vals) > 1 else 0.0
	# If this card is opponent top (approx), add part of margin
	my_val = opp_card_vals.get(card_idx, -1e9)
	if abs(my_val - top) < 1e-6:
		return 0.5 * (top - second if len(vals) > 1 else top)
	return 0.0


def build_actions(mask: np.ndarray) -> List[int]:
	return np.flatnonzero(mask).tolist()


def find_winning_buy(state: Dict[str, Any], legal: List[int]) -> int | None:
	me_prestige = state["me_prestige"]
	bonuses = state["me_bonuses"]
	cards = state["cards"]
	nobles = state["nobles"]
	for a in legal:
		if 15 <= a <= 26:
			card = cards[(a - 15)]
			if card is None:
				continue
			new_prestige = me_prestige + card["points"]
			if simulate_noble_after_purchase(bonuses, card["color_idx"], nobles):
				new_prestige += 3
			if new_prestige >= 15:
				return int(a)
	return None


def handle_almost_winning(state: Dict[str, Any], mask: np.ndarray) -> int | None:
	# Reserve or take tokens to enable a winning buy next turn when missing exactly 1 per color
	me_prestige = state["me_prestige"]
	bonuses = state["me_bonuses"]
	me_tokens = state["me_tokens"]
	cards = state["cards"]
	nobles = state["nobles"]
	legal = np.flatnonzero(mask)
	# Consider visible cards only
	for idx, card in enumerate(cards):
		if card is None:
			continue
		points_after = me_prestige + card["points"]
		if simulate_noble_after_purchase(bonuses, card["color_idx"], nobles):
			points_after += 3
		if points_after < 15:
			continue
		# compute needed per color
		missing = []
		ok = True
		for ci, col in enumerate(STANDARD_COLORS):
			cost = card["cost"].get(col, 0)
			need = max(0, cost - bonuses[ci] - me_tokens[ci])
			if need == 1:
				missing.append(ci)
			elif need > 1:
				ok = False; break
		if ok and 1 <= len(missing) <= 3:
			# If reserve visible for this card exists
			action_res = 27 + idx
			if action_res in legal:
				return int(action_res)
			# Else, try take action that collects any of missing colors
			# take2 or take3
			take2 = [a for a in legal if 10 <= a <= 14]
			for a in take2:
				col = a - 10
				if col in missing:
					return int(a)
			take3 = [a for a in legal if 0 <= a <= 9]
			if take3:
				# choose one that includes any missing
				for a in take3:
					colors = list(TAKE3_COMBOS[a])
					if any(c in colors for c in missing):
						return int(a)
	return None


def evaluate_actions(obs: np.ndarray, info: Dict[str, Any]) -> int:
	mask = info["action_mask"]
	legal = np.flatnonzero(mask).tolist()
	if not legal:
		return 0
	state = parse_obs(obs)
	# 1. Winning move
	wm = find_winning_buy(state, legal)
	if wm is not None:
		return wm
	# 2. Almost winning
	aw = handle_almost_winning(state, mask)
	if aw is not None:
		return aw
	# 3. Structured evaluation
	me_bonuses = state["me_bonuses"]
	me_tokens = state["me_tokens"]
	opp_bonuses = state["opp_bonuses"]
	opp_tokens = state["opp_tokens"]
	cards = state["cards"]
	nobles = state["nobles"]
	# Iterative token value refinement (3 cycles)
	card_vals = evaluate_all_cards(cards, me_bonuses, me_tokens, nobles, state["me_prestige"])
	token_vals = evaluate_tokens(card_vals, cards, me_bonuses, me_tokens)
	card_vals = evaluate_all_cards(cards, me_bonuses, me_tokens, nobles, state["me_prestige"])
	opp_card_vals = evaluate_all_cards(cards, opp_bonuses, opp_tokens, nobles, 0)
	# Score actions
	best_a = legal[0]
	best_v = -1e9
	bank = state["bank"]
	for a in legal:
		val = -1e9
		if 15 <= a <= 26:
			idx = a - 15
			card = cards[idx]
			if card is None:
				continue
			val = card_vals.get(idx, -1e9)
			val += get_defensive_value(idx, opp_card_vals)
		elif 27 <= a <= 38:
			# reserve visible: base on card value + gold token if available
			idx = a - 27
			card = cards[idx]
			if card is None:
				continue
			val = card_vals.get(idx, 0.0) + (3.0 if bank[5] > 0 else 0.0)
		elif 10 <= a <= 14:
			# take-2: sum token values for that color, scarcity weight
			ci = a - 10
			col = STANDARD_COLORS[ci]
			val = token_vals[col] * (2.0 + (1.0 / max(1, bank[ci])))
		elif 0 <= a <= 9:
			colors = TAKE3_COMBOS[a]
			val = sum(token_vals[STANDARD_COLORS[ci]] for ci in colors) / max(1, sum(int(bank[ci] > 0) for ci in colors))
		else:
			# buy reserved or reserve blind; fallback small value
			val = 0.1
		if val > best_v:
			best_v = val
			best_a = a
	return int(best_a)


def noble_policy(obs: np.ndarray, info: Dict[str, Any]) -> int:
	return evaluate_actions(obs, info) 