from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import itertools
import numpy as np

# Canonical color order
TOKEN_COLORS = ["white", "blue", "green", "red", "black", "gold"]
STANDARD_COLORS = TOKEN_COLORS[:-1]
COLOR_INDEX = {c: i for i, c in enumerate(TOKEN_COLORS)}

# Action layout (1-based in docs; 0-based in code):
# 0..9    TAKE3 (10 combos of 3 distinct from 5)
# 10..14  TAKE2 (5 singles)
# 15..26  BUY_VISIBLE (12 = 3 tiers x 4 slots)
# 27..38  RESERVE_VISIBLE (12)
# 39..41  RESERVE_BLIND (tiers 1..3)
# 42..44  BUY_RESERVED (slots 0..2)

TAKE3_OFFSET = 0
TAKE3_COUNT = 10
TAKE2_OFFSET = TAKE3_OFFSET + TAKE3_COUNT
TAKE2_COUNT = 5
BUY_VISIBLE_OFFSET = TAKE2_OFFSET + TAKE2_COUNT
BUY_VISIBLE_COUNT = 12
RESERVE_VISIBLE_OFFSET = BUY_VISIBLE_OFFSET + BUY_VISIBLE_COUNT
RESERVE_VISIBLE_COUNT = 12
RESERVE_BLIND_OFFSET = RESERVE_VISIBLE_OFFSET + RESERVE_VISIBLE_COUNT
RESERVE_BLIND_COUNT = 3
BUY_RESERVED_OFFSET = RESERVE_BLIND_OFFSET + RESERVE_BLIND_COUNT
BUY_RESERVED_COUNT = 3
TOTAL_ACTIONS = BUY_RESERVED_OFFSET + BUY_RESERVED_COUNT

# Precompute TAKE3 combos in fixed lexicographic order of indices 0..4
TAKE3_COMBOS: List[Tuple[int, int, int]] = list(itertools.combinations(range(5), 3))


def encode_take3_index(combo_index: int) -> int:
	return TAKE3_OFFSET + combo_index


def encode_take2_index(color_index: int) -> int:
	return TAKE2_OFFSET + color_index


def encode_buy_visible_index(tier: int, slot: int) -> int:
	return BUY_VISIBLE_OFFSET + (tier - 1) * 4 + slot


def encode_reserve_visible_index(tier: int, slot: int) -> int:
	return RESERVE_VISIBLE_OFFSET + (tier - 1) * 4 + slot


def encode_reserve_blind_index(tier: int) -> int:
	return RESERVE_BLIND_OFFSET + (tier - 1)


def encode_buy_reserved_index(slot: int) -> int:
	return BUY_RESERVED_OFFSET + slot


# Observation encoding helper (finalized layout):
# - Bank (6)
# - Current player: tokens(6), bonuses(5), prestige(1), reserved_count(1)
# - Opponent (first next player): tokens(6), bonuses(5), prestige(1), reserved_count(1)
# - Board 12 cards × (present1, tier1, points1, color_onehot5, cost5) => 12 x 13 = 156
# - Nobles up to 5 × (present1, req5) => 5 x 6 = 30 (we pad to 5)
# - Deck sizes (3)
# - turn_count (1), to_play (1), move_count (1), round_over_flag (1)
# Total length: 6 + (6+5+1+1) + (6+5+1+1) + 156 + 30 + 3 + 4 = 6 + 13 + 13 + 156 + 30 + 3 + 4 = 225

OBSERVATION_DIM = 225


def encode_observation(state) -> np.ndarray:
	from .state import STANDARD_COLORS as SC
	vec: List[int] = []
	# bank
	vec.extend(state.bank)
	# players (current and one opponent summary)
	p = state.players[state.to_play]
	opp = state.players[(state.to_play + 1) % state.num_players]
	vec.extend(p.tokens)
	vec.extend(p.bonuses)
	vec.append(p.prestige)
	vec.append(len(p.reserved))
	vec.extend(opp.tokens)
	vec.extend(opp.bonuses)
	vec.append(opp.prestige)
	vec.append(len(opp.reserved))
	# board
	for tier in (1, 2, 3):
		for slot in range(4):
			card = state.board[tier][slot]
			if card is None:
				vec.extend([0, 0, 0] + [0] * 5 + [0] * 5)
			else:
				vec.append(1)  # present
				vec.append(tier)
				vec.append(card.points)
				onehot = [0] * 5
				onehot[SC.index(card.color)] = 1
				vec.extend(onehot)
				for c in SC:
					vec.append(card.cost.get(c, 0))
	# nobles (pad to 5)
	visible = [n for n in state.nobles if n is not None]
	for i in range(5):
		if i < len(state.nobles) and state.nobles[i] is not None:
			n = state.nobles[i]
			vec.append(1)
			for c in SC:
				vec.append(n.requirements.get(c, 0))
		else:
			vec.extend([0, 0, 0, 0, 0, 0])
	# decks
	for tier in (1, 2, 3):
		vec.append(len(state.decks[tier]))
	# misc
	vec.append(state.turn_count)
	vec.append(state.to_play)
	vec.append(state.move_count)  # Add move count info
	vec.append(1 if (state.game_over and state.to_play == 0) else 0)
	return np.array(vec, dtype=np.int32) 