from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import json
import os

# Token colors: 5 standard + 1 gold (wild)
TOKEN_COLORS = ["white", "blue", "green", "red", "black", "gold"]
STANDARD_COLORS = TOKEN_COLORS[:-1]
COLOR_INDEX = {c: i for i, c in enumerate(TOKEN_COLORS)}
STANDARD_COLOR_INDEX = {c: i for i, c in enumerate(STANDARD_COLORS)}

# Mapping between human names and internal color names
HUMAN_TO_INTERNAL = {
	"diamond": "white",
	"sapphire": "blue",
	"emerald": "green",
	"ruby": "red",
	"onyx": "black",
}
INTERNAL_TO_HUMAN = {v: k for k, v in HUMAN_TO_INTERNAL.items()}

# Splendor defaults for 2 players
DEFAULT_BANK = {
	"white": 4,
	"blue": 4,
	"green": 4,
	"red": 4,
	"black": 4,
	"gold": 5,
}


@dataclass
class Card:
	id: int
	tier: int
	color: str  # one of STANDARD_COLORS
	points: int
	cost: Dict[str, int]


@dataclass
class Noble:
	id: int
	requirements: Dict[str, int]
	points: int = 3


@dataclass
class PlayerState:
	tokens: List[int] = field(default_factory=lambda: [0] * len(TOKEN_COLORS))
	bonuses: List[int] = field(default_factory=lambda: [0] * len(STANDARD_COLORS))
	prestige: int = 0
	reserved: List[Card] = field(default_factory=list)
	revealed_reserved: List[bool] = field(default_factory=list)  # True if reserved from board (public), False if from deck (hidden)
	nobles: List[Noble] = field(default_factory=list)

	def can_afford(self, card: Card) -> Tuple[bool, List[int]]:
		cost_remaining = []
		gold_needed = 0
		for i, color in enumerate(STANDARD_COLORS):
			req = card.cost.get(color, 0)
			discounted = max(0, req - self.bonuses[i])
			pay_with_color = min(self.tokens[i], discounted)
			remaining = discounted - pay_with_color
			cost_remaining.append(discounted)
			gold_needed += remaining
		return (self.tokens[COLOR_INDEX["gold"]] >= gold_needed, cost_remaining)


@dataclass
class SplendorState:
	num_players: int
	bank: List[int]  # tokens by TOKEN_COLORS order
	players: List[PlayerState]
	board: Dict[int, List[Optional[Card]]]  # tier -> 4 visible slots (None if empty)
	decks: Dict[int, List[Card]]  # tier -> remaining deck top at end
	nobles: List[Optional[Noble]]  # visible nobles (None if taken)
	to_play: int = 0
	turn_count: int = 1  # Now represents full rounds (both players move)
	move_count: int = 0  # Total individual moves made
	game_over: bool = False
	winner_index: Optional[int] = None
	turn_limit_reached: bool = False

	def copy(self) -> "SplendorState":
		# shallow copy is sufficient for our immutable Card/Noble objects, deep-copy lists
		return SplendorState(
			num_players=self.num_players,
			bank=list(self.bank),
			players=[PlayerState(tokens=list(p.tokens), bonuses=list(p.bonuses), prestige=p.prestige, reserved=list(p.reserved), revealed_reserved=list(p.revealed_reserved), nobles=list(p.nobles)) for p in self.players],
			board={tier: list(slots) for tier, slots in self.board.items()},
			decks={tier: list(deck) for tier, deck in self.decks.items()},
			nobles=list(self.nobles),
			to_play=self.to_play,
			turn_count=self.turn_count,
			move_count=self.move_count,
			game_over=self.game_over,
			winner_index=self.winner_index,
			turn_limit_reached=self.turn_limit_reached,
		)


# Data loading

def _data_dir() -> str:
	return os.path.join(os.path.dirname(__file__), "data")


def _load_cards_from_json() -> Dict[int, List[Card]]:
	path = os.path.join(_data_dir(), "cards.json")
	if not os.path.exists(path):
		raise FileNotFoundError(f"Missing cards.json at {path}")
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)
	if not raw:
		raise ValueError("cards.json is empty")
	cards: Dict[int, List[Card]] = {1: [], 2: [], 3: []}
	card_id = 0
	for obj in raw:
		tier = int(obj["tier"])  # 1..3
		if tier not in (1, 2, 3):
			raise ValueError("Invalid tier in cards.json")
		points = int(obj.get("points", 0))
		bonus_human = obj.get("bonus")
		if bonus_human is None:
			raise ValueError("Card missing bonus color")
		bonus_internal = HUMAN_TO_INTERNAL.get(bonus_human)
		if bonus_internal not in STANDARD_COLORS:
			raise ValueError(f"Invalid bonus color: {bonus_human}")
		cost_h = obj.get("cost", {})
		cost_internal: Dict[str, int] = {}
		for hname, v in cost_h.items():
			iname = HUMAN_TO_INTERNAL.get(hname)
			if iname is None:
				raise ValueError(f"Invalid cost color: {hname}")
			cost_internal[iname] = int(v)
		cards[tier].append(Card(id=card_id, tier=tier, color=bonus_internal, points=points, cost=cost_internal))
		card_id += 1
	# Basic validations
	if len(cards[1]) == 0 or len(cards[2]) == 0 or len(cards[3]) == 0:
		raise ValueError("cards.json missing tiers")
	# Count validation: base game has 40/30/20 per tiers 1/2/3
	if not (len(cards[1]) == 40 and len(cards[2]) == 30 and len(cards[3]) == 20):
		raise ValueError("cards.json must contain 40/30/20 cards for tiers 1/2/3")
	return cards


def _load_nobles_from_json() -> List[Noble]:
	path = os.path.join(_data_dir(), "nobles.json")
	if not os.path.exists(path):
		raise FileNotFoundError(f"Missing nobles.json at {path}")
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)
	if not raw:
		raise ValueError("nobles.json is empty")
	nobles: List[Noble] = []
	nid = 1000
	for obj in raw:
		points = int(obj.get("points", 3))
		req_h = obj.get("req", {})
		if not req_h:
			raise ValueError("Noble missing requirements")
		req_internal: Dict[str, int] = {}
		for hname, v in req_h.items():
			iname = HUMAN_TO_INTERNAL.get(hname)
			if iname is None:
				raise ValueError(f"Invalid noble req color: {hname}")
			req_internal[iname] = int(v)
		nobles.append(Noble(id=nid, requirements=req_internal, points=points))
		nid += 1
	# Validate count: base game 10 nobles
	if len(nobles) != 10:
		raise ValueError("nobles.json must contain 10 nobles for base game")
	return nobles


def initial_state(num_players: int = 2, seed: int = 0) -> SplendorState:
	local_rng = random.Random(seed)
	cards_by_tier = _load_cards_from_json()
	nobles_list = _load_nobles_from_json()
	# Build decks and board
	decks: Dict[int, List[Card]] = {1: list(cards_by_tier[1]), 2: list(cards_by_tier[2]), 3: list(cards_by_tier[3])}
	board: Dict[int, List[Optional[Card]]] = {1: [None] * 4, 2: [None] * 4, 3: [None] * 4}
	for tier in (1, 2, 3):
		local_rng.shuffle(decks[tier])
		for i in range(4):
			board[tier][i] = decks[tier].pop() if decks[tier] else None
	# Visible nobles count = players + 1, up to 5
	local_rng.shuffle(nobles_list)
	visible_n = min(num_players + 1, len(nobles_list))
	nobles_visible: List[Optional[Noble]] = nobles_list[:visible_n]
	bank = [DEFAULT_BANK[c] for c in TOKEN_COLORS]
	players = [PlayerState() for _ in range(num_players)]
	return SplendorState(
		num_players=num_players,
		bank=bank,
		players=players,
		board=board,
		decks=decks,
		nobles=nobles_visible,
		to_play=0,
		turn_count=1,   # Start at turn 1 (first full round)
		move_count=0,   # Start with no moves made
		game_over=False,
		winner_index=None,
		turn_limit_reached=False,
	) 