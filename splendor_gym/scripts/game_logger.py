"""
Human-readable game logging for Splendor Gym.

This script runs complete games and outputs detailed, human-readable logs
showing all game states and actions for verification purposes.
"""

import argparse
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from splendor_gym.envs import SplendorEnv
from splendor_gym.engine import (
    SplendorState, 
    legal_moves, 
    apply_action,
    is_terminal,
    winner
)
from splendor_gym.engine.state import (
    TOKEN_COLORS, 
    STANDARD_COLORS, 
    HUMAN_TO_INTERNAL,
    INTERNAL_TO_HUMAN
)
from splendor_gym.engine.encode import (
    TAKE3_COMBOS,
    TAKE3_OFFSET, TAKE3_COUNT,
    TAKE2_OFFSET, TAKE2_COUNT, 
    BUY_VISIBLE_OFFSET, BUY_VISIBLE_COUNT,
    RESERVE_VISIBLE_OFFSET, RESERVE_VISIBLE_COUNT,
    RESERVE_BLIND_OFFSET, RESERVE_BLIND_COUNT,
    BUY_RESERVED_OFFSET, BUY_RESERVED_COUNT
)


@dataclass
class GameLog:
    """Container for game logging information."""
    turn: int
    player: int
    state_before: str
    action: str
    state_after: str
    legal_actions: List[str]


class SplendorGameLogger:
    """Logs Splendor games in compact human-readable format."""
    
    def __init__(self):
        self.logs: List[GameLog] = []
        # Color abbreviations: w=white, b=blue, g=green, r=red, k=black (avoid confusion with blue)
        self.color_abbrev = {"white": "w", "blue": "b", "green": "g", "red": "r", "black": "k", "gold": "G"}
        self.abbrev_to_color = {v: k for k, v in self.color_abbrev.items()}
    
    def _format_card_compact(self, card) -> str:
        """Format card as: g-1pt-2b3r3k (green card, 1 point, costs 2 blue 3 red 3 black)"""
        if not card:
            return "[empty]"
        
        color_abbrev = self.color_abbrev[card.color]
        cost_parts = []
        for color in ["white", "blue", "green", "red", "black"]:
            amount = card.cost.get(color, 0)
            if amount > 0:
                cost_parts.append(f"{amount}{self.color_abbrev[color]}")
        cost_str = "".join(cost_parts) if cost_parts else "free"
        return f"{color_abbrev}-{card.points}pt-{cost_str}"
    
    def _format_cost_compact(self, cost: Dict[str, int]) -> str:
        """Format cost as: 2w3b1g (2 white, 3 blue, 1 green)"""
        if not cost:
            return "free"
        cost_parts = []
        for color in ["white", "blue", "green", "red", "black"]:
            amount = cost.get(color, 0)
            if amount > 0:
                cost_parts.append(f"{amount}{self.color_abbrev[color]}")
        return "".join(cost_parts)
    
    def _format_tokens_compact(self, tokens: List[int]) -> str:
        """Format tokens as: 2w1b3g (2 white, 1 blue, 3 green)"""
        parts = []
        for i, color in enumerate(TOKEN_COLORS):
            if tokens[i] > 0:
                parts.append(f"{tokens[i]}{self.color_abbrev[color]}")
        return "".join(parts) if parts else "none"
    
    def _format_bonuses_compact(self, bonuses: List[int]) -> str:
        """Format bonuses as: 1g2r (1 green, 2 red bonus)"""
        parts = []
        for i, color in enumerate(STANDARD_COLORS):
            if bonuses[i] > 0:
                parts.append(f"{bonuses[i]}{self.color_abbrev[color]}")
        return "".join(parts) if parts else "none"
        
    def decode_action(self, action: int, state: SplendorState) -> str:
        """Convert action number to compact description."""
        
        if TAKE3_OFFSET <= action < TAKE3_OFFSET + TAKE3_COUNT:
            # Take 3 different colors (or reduced version)
            idx = action - TAKE3_OFFSET
            available_colors = [i for i in range(5) if state.bank[i] >= 1]
            
            if len(available_colors) >= 3:
                a, b, c = TAKE3_COMBOS[idx]
                abbrevs = [self.color_abbrev[STANDARD_COLORS[i]] for i in (a, b, c)]
                return f"Take3: {abbrevs[0]}{abbrevs[1]}{abbrevs[2]}"
            elif len(available_colors) == 2:
                abbrevs = [self.color_abbrev[STANDARD_COLORS[i]] for i in available_colors]
                return f"Take2: {abbrevs[0]}{abbrevs[1]} (reduced)"
            elif len(available_colors) == 1:
                abbrev = self.color_abbrev[STANDARD_COLORS[available_colors[0]]]
                return f"Take1: {abbrev} (reduced)"
                
        elif TAKE2_OFFSET <= action < TAKE2_OFFSET + TAKE2_COUNT:
            # Take 2 same color
            color_idx = action - TAKE2_OFFSET
            abbrev = self.color_abbrev[STANDARD_COLORS[color_idx]]
            return f"Take2: {abbrev}{abbrev}"
            
        elif BUY_VISIBLE_OFFSET <= action < BUY_VISIBLE_OFFSET + BUY_VISIBLE_COUNT:
            # Buy visible card
            offset = action - BUY_VISIBLE_OFFSET
            tier = 1 + offset // 4
            slot = offset % 4
            card = state.board[tier][slot]
            if card:
                return f"Buy: T{tier}S{slot+1} {self._format_card_compact(card)}"
            return f"Buy: T{tier}S{slot+1} [empty]"
            
        elif RESERVE_VISIBLE_OFFSET <= action < RESERVE_VISIBLE_OFFSET + RESERVE_VISIBLE_COUNT:
            # Reserve visible card
            offset = action - RESERVE_VISIBLE_OFFSET
            tier = 1 + offset // 4
            slot = offset % 4
            card = state.board[tier][slot]
            if card:
                return f"Reserve: T{tier}S{slot+1} {self._format_card_compact(card)}"
            return f"Reserve: T{tier}S{slot+1} [empty]"
            
        elif RESERVE_BLIND_OFFSET <= action < RESERVE_BLIND_OFFSET + RESERVE_BLIND_COUNT:
            # Reserve from deck
            tier = 1 + (action - RESERVE_BLIND_OFFSET)
            return f"Reserve: T{tier} blind"
            
        elif BUY_RESERVED_OFFSET <= action < BUY_RESERVED_OFFSET + BUY_RESERVED_COUNT:
            # Buy reserved card
            slot = action - BUY_RESERVED_OFFSET
            player = state.players[state.to_play]
            if slot < len(player.reserved):
                card = player.reserved[slot]
                return f"BuyReserved: #{slot+1} {self._format_card_compact(card)}"
            return f"BuyReserved: #{slot+1} [empty]"
            
        return f"Action{action}"
    
    def format_game_state(self, state: SplendorState, player_perspective: int = -1) -> str:
        """Format game state in compact format."""
        lines = []
        
        # Header - much more compact with move info
        move_info = f"M{state.move_count}" if hasattr(state, 'move_count') else ""
        lines.append(f"=== TURN {state.turn_count}{move_info} - P{state.to_play} to move ===")
        
        # Bank - one line
        bank_str = self._format_tokens_compact(state.bank)
        lines.append(f"Bank: {bank_str}")
        
        # Players - much more compact
        for i, player in enumerate(state.players):
            marker = ">>>" if i == state.to_play else "   "
            tokens_str = self._format_tokens_compact(player.tokens)
            bonuses_str = self._format_bonuses_compact(player.bonuses)
            
            # Reserved cards compact
            reserved_str = "none"
            if player.reserved:
                reserved_parts = []
                for card in player.reserved:
                    reserved_parts.append(self._format_card_compact(card))
                reserved_str = ", ".join(reserved_parts)
            
            # Nobles compact
            nobles_str = "none"
            if player.nobles:
                nobles_parts = []
                for noble in player.nobles:
                    req_str = self._format_cost_compact(noble.requirements)
                    nobles_parts.append(f"{noble.points}pt-{req_str}")
                nobles_str = ", ".join(nobles_parts)
            
            lines.append(f"{marker} P{i}: {tokens_str} | bonus:{bonuses_str} | pts:{player.prestige} | reserved:[{reserved_str}] | nobles:[{nobles_str}]")
        
        # Board - much more compact
        lines.append("Board:")
        for tier in [3, 2, 1]:  # Display top tier first
            cards = []
            for slot in range(4):
                card = state.board[tier][slot]
                cards.append(self._format_card_compact(card) if card else "[empty]")
            lines.append(f"  T{tier}: {' | '.join(cards)}")
        
        # Available nobles - one line
        available_nobles = [n for n in state.nobles if n is not None]
        if available_nobles:
            noble_parts = []
            for noble in available_nobles:
                req_str = self._format_cost_compact(noble.requirements)
                noble_parts.append(f"{noble.points}pt-{req_str}")
            lines.append(f"Nobles: {' | '.join(noble_parts)}")
        else:
            lines.append("Nobles: none")
        
        # Deck sizes - one line
        deck_sizes = [f"T{tier}:{len(state.decks[tier])}" for tier in [1, 2, 3]]
        lines.append(f"Decks: {' '.join(deck_sizes)}")
        
        return "\n".join(lines)
    
    def get_legal_actions_description(self, state: SplendorState) -> List[str]:
        """Get human-readable descriptions of all legal actions."""
        mask = legal_moves(state)
        descriptions = []
        for i, legal in enumerate(mask):
            if legal:
                descriptions.append(f"{i}: {self.decode_action(i, state)}")
        return descriptions
    
    def log_game_step(self, state_before: SplendorState, action: int, state_after: SplendorState):
        """Log a single game step."""
        log = GameLog(
            turn=state_before.turn_count,
            player=state_before.to_play,
            state_before=self.format_game_state(state_before),
            action=self.decode_action(action, state_before),
            state_after=self.format_game_state(state_after),
            legal_actions=self.get_legal_actions_description(state_before)
        )
        self.logs.append(log)
    
    def print_game_log(self, show_legal_actions: bool = False, max_turns: int = None):
        """Print the complete game log with full turns (both players)."""
        print("\n" + "=" * 80)
        print("SPLENDOR GAME LOG (Full Rounds)")
        print("=" * 80)
        
        # Group logs by full turns (both players' moves)
        turn_groups = {}
        for log in self.logs:
            if log.turn not in turn_groups:
                turn_groups[log.turn] = []
            turn_groups[log.turn].append(log)
        
        turn_count = 0
        for turn_num in sorted(turn_groups.keys()):
            turn_count += 1
            if max_turns and turn_count > max_turns:
                print(f"\n... (showing first {max_turns} full turns only) ...")
                break
                
            logs_in_turn = turn_groups[turn_num]
            
            print(f"\n{'='*20} TURN {turn_num} {'='*20}")
            
            # Show both players' moves in this turn
            for i, log in enumerate(logs_in_turn):
                half = "FIRST HALF" if i == 0 else "SECOND HALF"
                print(f"\n--- {half} (Player {log.player}) ---")
                print(log.state_before)
                
                if show_legal_actions:
                    print("Legal actions:")
                    for action_desc in log.legal_actions[:10]:  # Show first 10 to save space
                        print(f"  {action_desc}")
                    if len(log.legal_actions) > 10:
                        print(f"  ... ({len(log.legal_actions) - 10} more)")
                    print("")
                
                print(f"P{log.player} ACTION: {log.action}")
                print("")
            
            # Show final state after both players moved
            if logs_in_turn:
                print("--- TURN END STATE ---")
                print(logs_in_turn[-1].state_after)
            print("\n" + "-" * 60 + "\n")


def run_logged_game(policy_type: str = "random", seed: int = 42, max_turns: int = None) -> SplendorGameLogger:
    """Run a complete game with logging."""
    
    # Create environment
    env = SplendorEnv(num_players=2)
    logger = SplendorGameLogger()
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Set up RNG for action selection
    rng = random.Random(seed + 1000)  # Offset seed to avoid correlation
    
    step_count = 0
    max_steps = 1000  # Prevent infinite games
    
    while step_count < max_steps:
        if max_turns and step_count >= max_turns * 2:  # Each turn involves 2 players
            break
            
        # Get current state
        state_before = env.state.copy()
        
        # Choose action based on policy
        mask = info["action_mask"]
        legal_actions = [i for i, legal in enumerate(mask) if legal]
        
        if not legal_actions:
            print("No legal actions available - game should have ended!")
            break
            
        if policy_type == "random":
            action = rng.choice(legal_actions)
        elif policy_type == "first":
            action = legal_actions[0]  # Always pick first legal action
        else:
            # Interactive mode - let user choose
            print(logger.format_game_state(state_before))
            print("\nLegal actions:")
            for i, act_idx in enumerate(legal_actions):
                print(f"{i}: {logger.decode_action(act_idx, state_before)}")
            
            while True:
                try:
                    choice = int(input("Choose action index (0-based): "))
                    if 0 <= choice < len(legal_actions):
                        action = legal_actions[choice]
                        break
                    else:
                        print("Invalid choice, try again.")
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input, try again.")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        state_after = env.state.copy()
        
        # Log this step
        logger.log_game_step(state_before, action, state_after)
        
        # Check if game ended
        if terminated or truncated:
            print(f"\nGAME ENDED after {step_count + 1} steps!")
            if terminated:
                w = winner(state_after)
                if w is not None:
                    print(f"Winner: Player {w}")
                else:
                    print("Game ended in a draw")
            break
            
        step_count += 1
    
    return logger


def main():
    parser = argparse.ArgumentParser(description="Run and log Splendor games for verification")
    parser.add_argument("--policy", type=str, default="random", 
                       choices=["random", "first", "interactive"],
                       help="Policy for action selection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show-legal", action="store_true", 
                       help="Show legal actions at each step")
    parser.add_argument("--max-turns", type=int, help="Maximum turns to show in log")
    parser.add_argument("--output", type=str, help="Save log to file")
    
    args = parser.parse_args()
    
    print(f"Running Splendor game with {args.policy} policy (seed: {args.seed})")
    
    # Run the game
    logger = run_logged_game(args.policy, args.seed, args.max_turns)
    
    # Print or save the log
    if args.output:
        import sys
        original_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            logger.print_game_log(args.show_legal, args.max_turns)
            sys.stdout = original_stdout
        print(f"Game log saved to {args.output}")
    else:
        logger.print_game_log(args.show_legal, args.max_turns)


if __name__ == "__main__":
    main()
