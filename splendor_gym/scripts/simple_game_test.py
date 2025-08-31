"""
Simple script to test and verify Splendor Gym implementation.

This runs a few short games and saves detailed logs for manual inspection.
"""

from game_logger import run_logged_game
import os

def main():
    print("Running Splendor Gym verification tests...")
    print("=" * 50)
    
    # Create logs directory
    os.makedirs("game_logs", exist_ok=True)
    
    # Test different scenarios
    scenarios = [
        ("random_game_1", "random", 42),
        ("random_game_2", "random", 100), 
        ("first_action_game", "first", 42),
    ]
    
    for name, policy, seed in scenarios:
        print(f"\nRunning {name} with {policy} policy (seed: {seed})")
        print("-" * 40)
        
        # Run the game
        logger = run_logged_game(policy, seed, max_turns=20)  # Show first 20 turns
        
        # Save detailed log
        log_file = f"game_logs/{name}_detailed.txt"
        with open(log_file, 'w') as f:
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            logger.print_game_log(show_legal_actions=True)
            sys.stdout = original_stdout
        
        # Save simple log (no legal actions)
        simple_log_file = f"game_logs/{name}_simple.txt" 
        with open(simple_log_file, 'w') as f:
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            logger.print_game_log(show_legal_actions=False)
            sys.stdout = original_stdout
            
        print(f"Saved detailed log: {log_file}")
        print(f"Saved simple log: {simple_log_file}")
        
        # Print summary
        num_turns = len(logger.logs)
        if num_turns > 0:
            last_log = logger.logs[-1]
            print(f"Game lasted {num_turns} turns")
            # Check if we can extract winner info from the final state
        
    print(f"\n✅ All game logs saved in 'game_logs/' directory")
    print("\nTo manually verify:")
    print("1. Check game_logs/*_simple.txt for readable game flow")  
    print("2. Check game_logs/*_detailed.txt for legal actions at each step")
    print("3. Verify token counts, card costs, and rule enforcement")

if __name__ == "__main__":
    main()
