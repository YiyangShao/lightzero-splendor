# Splendor Gymnasium Environment — Design & Implementation Guide

This document tracks the design and implementation details for a Gymnasium-compatible Splendor environment intended for RL research and self-play training (AlphaZero-style).

## Goals
- RL-ready, Gymnasium-compatible
- Action masking for invalid moves
- Full ruleset: tokens, purchases, reserves, nobles, end conditions
- Deterministic seeds for reproducibility

## Engine Layer
- Pure Python rule engine, framework-agnostic
- State includes: bank (6 tokens), players, board (12 visible cards), decks, nobles, turn, player index
- API:
  - `legal_moves(state) -> mask`
  - `apply_action(state, action) -> new_state`
  - `is_terminal(state) -> bool`
  - `winner(state) -> Optional[int]`

## Actions
Flattened discrete space of size 45 (0-based):
- 0..9: Take-3 distinct colors (10 combos over 5 colors)
- 10..14: Take-2 same color (requires bank[color] ≥ 4)
- 15..26: Buy visible (3 tiers × 4 slots, row-major by tier then slot)
- 27..38: Reserve visible (same 12 slots)
- 39..41: Reserve blind (tiers 1..3)
- 42..44: Buy one of up to 3 reserved cards (slots 0..2)

## Observations
Fixed-size vector `int32` of length 224 with the following layout:
- Bank (6)
- Current player: tokens(6), bonuses(5), prestige(1), reserved_count(1) → 13
- Opponent summary: tokens(6), bonuses(5), prestige(1), reserved_count(1) → 13
- Board: 12 cards × (present1, tier1, points1, color_onehot[5], cost[5]) → 12 × 13 = 156
- Nobles (pad to 5): 5 × (present1, req[5]) → 30
- Deck sizes (3)
- turn_count (1), to_play (1), round_over_flag (1)
Total: 6 + 13 + 13 + 156 + 30 + 3 + 3 = 224

## Rewards
- 0 for all non-terminal steps
- +1 win / -1 loss / 0 draw at terminal

## Turn Handling
- Single-agent per step (2 players supported); `to_play` indicates the acting player
- `step` applies action for `to_play` and switches to next player

## Rendering
- Text render: prints bank and current player snapshot

## Notes
- Strict data: `cards.json` must contain 40/30/20 tiered cards; `nobles.json` must contain 10 nobles
- Action mask is provided via `info["action_mask"]`
- Token limit enforced at end of turn; excess returned randomly from non-gold tokens (gold returned only if unavoidable)

## Roadmap
- Phase 1: Solidify rules, JSON cards/nobles
- Phase 2: Env adapter, masking (done)
- Phase 3: Validation with env checker, random rollouts (done)
- Phase 4: Algorithms integration 