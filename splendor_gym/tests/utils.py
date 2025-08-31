import numpy as np

from splendor_gym.envs import SplendorEnv
from splendor_gym.engine import legal_moves
from splendor_gym.engine.state import HUMAN_TO_INTERNAL, COLOR_INDEX, STANDARD_COLORS

H2I = HUMAN_TO_INTERNAL
C2I = COLOR_INDEX


def make_env(seed=123):
	return SplendorEnv(num_players=2, seed=seed)


def mask_from_state(env):
	return np.array(legal_moves(env.state), dtype=np.int8)


def get_mask(info):
	m = info.get("action_mask", None)
	assert m is not None, "info must contain 'action_mask'"
	return np.asarray(m, dtype=np.int8)


def set_bank(env, **kw):
	for hname, v in kw.items():
		iname = H2I.get(hname, hname)
		env.state.bank[C2I[iname]] = int(v)


def set_player_tokens(env, pid, **kw):
	for hname, v in kw.items():
		iname = H2I.get(hname, hname)
		env.state.players[pid].tokens[C2I[iname]] = int(v)


def set_player_bonuses(env, pid, **kw):
	for hname, v in kw.items():
		iname = H2I.get(hname, hname)
		idx = STANDARD_COLORS.index(iname)
		env.state.players[pid].bonuses[idx] = int(v)


def clear_board(env):
	for t in (1, 2, 3):
		env.state.board[t] = [None, None, None, None]


def place_card(env, tier, slot, card):
	env.state.board[tier][slot] = card


def empty_deck(env, tier):
	env.state.decks[tier].clear() 