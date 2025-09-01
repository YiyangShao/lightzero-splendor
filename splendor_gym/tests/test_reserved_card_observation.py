"""
Test cases for reserved card observation functionality.

Tests the information asymmetry where:
- Players can see full details of their own reserved cards
- Players can see full details of opponent's revealed reserved cards (reserved from board)
- Players cannot see details of opponent's hidden reserved cards (reserved from deck)
"""

import numpy as np
import pytest

from splendor_gym.envs import SplendorEnv
from splendor_gym.engine.encode import OBSERVATION_DIM, _encode_reserved_card
from splendor_gym.engine.state import STANDARD_COLORS
from splendor_gym.tests.utils import make_env


def get_observation_sections(obs):
    """Parse observation into its sections for easier testing."""
    sections = {}
    
    # Fixed positions in observation
    sections['bank'] = obs[0:6]
    sections['current_player'] = obs[6:19]  # tokens(6) + bonuses(5) + prestige(1) + reserved_count(1)
    sections['opponent'] = obs[19:32]       # tokens(6) + bonuses(5) + prestige(1) + reserved_count(1)
    sections['board'] = obs[32:188]         # 12 cards × 13 dims = 156 elements, starting at 32
    sections['reserved'] = obs[188:272]     # 6 cards × 14 dims = 84 elements, starting at 188
    sections['nobles'] = obs[272:290]       # 3 nobles × 6 dims = 18 elements
    sections['game_state'] = obs[290:297]   # 7 elements
    
    # Split reserved section
    sections['own_reserved'] = sections['reserved'][:42]      # First 3 slots × 14 dims
    sections['opponent_reserved'] = sections['reserved'][42:] # Next 3 slots × 14 dims
    
    return sections


def get_reserved_card_data(reserved_section, slot_index):
    """Extract card data from reserved section for a specific slot (0-2)."""
    start_idx = slot_index * 14
    end_idx = start_idx + 14
    card_data = reserved_section[start_idx:end_idx]
    
    if card_data[0] == 0:  # not present
        return None
    
    return {
        'present': card_data[0],
        'tier': card_data[1], 
        'points': card_data[2],
        'color_onehot': card_data[3:8],
        'cost': card_data[8:13],
        'revealed': card_data[13]
    }


def test_observation_dimension():
    """Test that observation has correct dimensions."""
    env = make_env()
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (OBSERVATION_DIM,), f"Expected shape ({OBSERVATION_DIM},), got {obs.shape}"
    assert OBSERVATION_DIM == 297, f"Expected OBSERVATION_DIM=297, got {OBSERVATION_DIM}"


def test_initial_state_no_reserved_cards():
    """Test that initial state has no reserved cards."""
    env = make_env()
    obs, info = env.reset(seed=42)
    
    sections = get_observation_sections(obs)
    
    # Current player should have 0 reserved cards
    assert sections['current_player'][12] == 0, "Current player should start with 0 reserved cards"
    
    # Opponent should have 0 reserved cards  
    assert sections['opponent'][12] == 0, "Opponent should start with 0 reserved cards"
    
    # All reserved card slots should be empty (all zeros)
    assert np.all(sections['own_reserved'] == 0), "Own reserved slots should be empty initially"
    assert np.all(sections['opponent_reserved'] == 0), "Opponent reserved slots should be empty initially"


def test_reserve_visible_card_information():
    """Test that reserving a visible card shows full information to both players."""
    env = make_env(seed=42)
    obs, info = env.reset(seed=42)
    
    # Player 0 reserves a visible card
    mask = info['action_mask']
    reserve_visible_actions = [i for i in range(len(mask)) if mask[i] and 27 <= i <= 38]
    assert len(reserve_visible_actions) > 0, "Should have visible cards to reserve"
    
    action = reserve_visible_actions[0]
    obs, _, _, _, info = env.step(action)
    
    sections = get_observation_sections(obs)
    
    # Current player (P1) should see P0's reserved card count
    assert sections['current_player'][12] == 0, "Current player (P1) should have 0 reserved cards"
    assert sections['opponent'][12] == 1, "Opponent (P0) should have 1 reserved card"
    
    # P0's reserved card should be visible to P1 (revealed=1)
    opponent_card = get_reserved_card_data(sections['opponent_reserved'], 0)
    assert opponent_card is not None, "Opponent should have a reserved card"
    assert opponent_card['present'] == 1, "Card should be present"
    assert opponent_card['revealed'] == 1, "Card should be revealed (reserved from board)"
    assert opponent_card['tier'] in [1, 2, 3], "Card should have valid tier"
    

def test_reserve_blind_card_information():
    """Test that reserving a blind card hides information from opponent."""
    env = make_env(seed=42)
    obs, info = env.reset(seed=42)
    
    # Player 0 takes some action first
    mask = info['action_mask']
    legal_actions = [i for i in range(len(mask)) if mask[i]]
    obs, _, _, _, info = env.step(legal_actions[0])
    
    # Player 1 reserves a blind card
    mask = info['action_mask']
    reserve_blind_actions = [i for i in range(len(mask)) if mask[i] and 39 <= i <= 41]
    if len(reserve_blind_actions) == 0:
        pytest.skip("No blind reserve actions available in this seed")
    
    action = reserve_blind_actions[0]
    obs, _, _, _, info = env.step(action)
    
    sections = get_observation_sections(obs)
    
    # Current player (P0) should see P1's reserved card count
    assert sections['opponent'][12] == 1, "Opponent (P1) should have 1 reserved card"
    
    # P1's reserved card should be hidden from P0 (all zeros)
    opponent_reserved_slot_0 = sections['opponent_reserved'][:14]
    assert np.all(opponent_reserved_slot_0 == 0), "Hidden reserved card should be all zeros"


def test_own_reserved_cards_always_visible():
    """Test that players can always see full details of their own reserved cards."""
    env = make_env(seed=123)
    obs, info = env.reset(seed=123)
    
    # Player 0 reserves a blind card (hidden from opponent but visible to self)
    mask = info['action_mask']
    reserve_blind_actions = [i for i in range(len(mask)) if mask[i] and 39 <= i <= 41]
    if len(reserve_blind_actions) == 0:
        pytest.skip("No blind reserve actions available in this seed")
    
    action = reserve_blind_actions[0]
    obs, _, _, _, info = env.step(action)
    
    # After P0's move, observation is from P1's perspective
    # P1 should see P0 as opponent with 1 reserved card (but hidden details)
    sections = get_observation_sections(obs)
    assert sections['opponent'][12] == 1, "Opponent (P0) should have 1 reserved card"
    
    # P0's card should be hidden from P1 (all zeros because it was reserved from deck)
    opponent_reserved_slot_0 = sections['opponent_reserved'][:14]
    assert np.all(opponent_reserved_slot_0 == 0), "P0's hidden card should be all zeros from P1's perspective"
    
    # Now P1 takes an action to get back to P0's perspective
    mask = info['action_mask']
    legal_actions = [i for i in range(len(mask)) if mask[i]]
    obs, _, _, _, info = env.step(legal_actions[0])
    
    # Now observation is from P0's perspective - they should see their own card
    sections = get_observation_sections(obs)
    assert sections['current_player'][12] == 1, "Current player (P0) should have 1 reserved card"
    
    own_card = get_reserved_card_data(sections['own_reserved'], 0)
    assert own_card is not None, "Should have own reserved card"
    assert own_card['present'] == 1, "Own card should be present"
    assert own_card['revealed'] == 1, "Own cards are always marked as revealed in observation"
    assert own_card['tier'] in [1, 2, 3], "Own card should have valid tier"


def test_multiple_reserved_cards_mixed_visibility():
    """Test scenario with multiple reserved cards of different visibility."""
    env = make_env(seed=456)
    obs, info = env.reset(seed=456)
    
    # Player 0 reserves visible card
    mask = info['action_mask']
    reserve_visible = [i for i in range(len(mask)) if mask[i] and 27 <= i <= 38]
    obs, _, _, _, info = env.step(reserve_visible[0])
    
    # After P0's move, it's P1's turn
    # Player 1 reserves blind card
    mask = info['action_mask']
    reserve_blind = [i for i in range(len(mask)) if mask[i] and 39 <= i <= 41]
    if len(reserve_blind) > 0:
        obs, _, _, _, info = env.step(reserve_blind[0])
        
        # After P1's move, it's P0's turn again
        # Player 0 reserves another visible card
        mask = info['action_mask'] 
        reserve_visible = [i for i in range(len(mask)) if mask[i] and 27 <= i <= 38]
        if len(reserve_visible) > 0:
            obs, _, _, _, info = env.step(reserve_visible[0])
            
            # After P0's second reserve move, it's P1's turn (observation from P1's perspective)
            sections = get_observation_sections(obs)
            
            # From P1's perspective: P0 (opponent) should have 2 reserved cards
            assert sections['opponent'][12] == 2, "Opponent (P0) should have 2 reserved cards"
            
            # Current player (P1) should have 1 reserved card  
            assert sections['current_player'][12] == 1, "Current player (P1) should have 1 reserved card"
            
            # P0's cards should be visible to P1 (both were reserved from board)
            opponent_card_1 = get_reserved_card_data(sections['opponent_reserved'], 0)
            opponent_card_2 = get_reserved_card_data(sections['opponent_reserved'], 1)
            assert opponent_card_1 is not None, "Should see first opponent card"
            assert opponent_card_2 is not None, "Should see second opponent card"
            assert opponent_card_1['revealed'] == 1, "First card should be revealed"
            assert opponent_card_2['revealed'] == 1, "Second card should be revealed"
            
            # P1's own card should be visible to themselves
            own_card = get_reserved_card_data(sections['own_reserved'], 0)
            assert own_card is not None, "Should have own reserved card"


def test_reserved_card_encoding_function():
    """Test the _encode_reserved_card helper function."""
    from splendor_gym.engine.state import Card
    
    # Test encoding a real card
    card = Card(id=1, tier=2, color='blue', points=1, cost={'white': 2, 'green': 1})
    
    # Test revealed card
    encoded_revealed = _encode_reserved_card(card, STANDARD_COLORS, revealed=True)
    assert len(encoded_revealed) == 14, "Should encode to 14 elements"
    assert encoded_revealed[0] == 1, "Present flag should be 1"
    assert encoded_revealed[1] == 2, "Tier should be 2"
    assert encoded_revealed[2] == 1, "Points should be 1"
    assert encoded_revealed[4] == 1, "Blue color should be set"
    assert encoded_revealed[8] == 2, "White cost should be 2"
    assert encoded_revealed[10] == 1, "Green cost should be 1"
    assert encoded_revealed[13] == 1, "Revealed flag should be 1"
    
    # Test hidden card (should be all zeros when passed as None to simulate hiding)
    encoded_hidden = _encode_reserved_card(None, STANDARD_COLORS, revealed=False)
    assert len(encoded_hidden) == 14, "Should encode to 14 elements"
    assert all(x == 0 for x in encoded_hidden), "Hidden card should be all zeros"


def test_buy_reserved_card_removes_from_observation():
    """Test that buying a reserved card removes it from observation."""
    env = make_env(seed=789)
    obs, info = env.reset(seed=789)
    
    # Player 0 reserves a visible card
    mask = info['action_mask']
    reserve_visible = [i for i in range(len(mask)) if mask[i] and 27 <= i <= 38]
    obs, _, _, _, info = env.step(reserve_visible[0])
    
    # After P0's reserve, it's P1's turn - check that P0 has 1 reserved card
    sections = get_observation_sections(obs)
    assert sections['opponent'][12] == 1, "Opponent (P0) should have 1 reserved card after reserving"
    
    # P1 takes some action to get back to P0
    mask = info['action_mask']
    legal_actions = [i for i in range(len(mask)) if mask[i]]
    obs, _, _, _, info = env.step(legal_actions[0])
    
    # Now it's P0's turn - they should see their own reserved card
    sections = get_observation_sections(obs)
    assert sections['current_player'][12] == 1, "Current player (P0) should have 1 reserved card"
    
    # Give P0 enough tokens to buy the card (simplified test)
    env.state.players[0].tokens[0] = 10  # Give lots of white tokens
    env.state.players[0].tokens[1] = 10  # Give lots of blue tokens
    env.state.players[0].tokens[2] = 10  # Give lots of green tokens
    env.state.players[0].tokens[3] = 10  # Give lots of red tokens
    env.state.players[0].tokens[4] = 10  # Give lots of black tokens
    
    # Regenerate action mask after modifying player state
    from splendor_gym.engine.rules import legal_moves
    mask = legal_moves(env.state)
    
    # Try to buy the reserved card (action 42 is buy reserved slot 0)
    if mask[42]:  # If can buy reserved card 0
        obs, _, _, _, info = env.step(42)
        # After buying, it's P1's turn - P0 should have 0 reserved cards
        sections = get_observation_sections(obs)
        assert sections['opponent'][12] == 0, "Opponent (P0) should have 0 reserved cards after buying"
    else:
        pytest.skip("Cannot buy reserved card in this test scenario")


def test_observation_consistency_across_perspectives():
    """Test that information visibility is consistent when perspectives switch."""
    env = make_env(seed=999)
    obs, info = env.reset(seed=999)
    
    # Player 0 reserves visible card  
    mask = info['action_mask']
    reserve_visible = [i for i in range(len(mask)) if mask[i] and 27 <= i <= 38]
    obs_p1_perspective, _, _, _, info = env.step(reserve_visible[0])
    
    # Player 1 reserves blind card
    mask = info['action_mask']
    reserve_blind = [i for i in range(len(mask)) if mask[i] and 39 <= i <= 41]
    if len(reserve_blind) > 0:
        obs_p0_perspective, _, _, _, info = env.step(reserve_blind[0])
        
        # From P0's perspective: should see P1's card count but not details (hidden)
        sections_p0 = get_observation_sections(obs_p0_perspective)
        assert sections_p0['opponent'][12] == 1, "P0 should see P1 has 1 reserved card"
        opponent_card_p0_view = sections_p0['opponent_reserved'][:14]
        assert np.all(opponent_card_p0_view == 0), "P0 should not see P1's hidden card details"
        
        # From P1's perspective (previous obs): should see P0's card details (revealed)
        sections_p1 = get_observation_sections(obs_p1_perspective)
        assert sections_p1['opponent'][12] == 1, "P1 should see P0 has 1 reserved card"
        opponent_card_p1_view = get_reserved_card_data(sections_p1['opponent_reserved'], 0)
        assert opponent_card_p1_view is not None, "P1 should see P0's card details"
        assert opponent_card_p1_view['revealed'] == 1, "P0's card should be marked as revealed"


def test_comprehensive_reserved_card_scenario():
    """Comprehensive test covering all aspects of reserved card observation."""
    env = make_env(seed=42)
    obs, info = env.reset(seed=42)
    
    # Verify initial state
    sections = get_observation_sections(obs)
    assert np.all(sections['own_reserved'] == 0), "Initially no reserved cards"
    assert np.all(sections['opponent_reserved'] == 0), "Initially no opponent reserved cards"
    
    # Player 0 reserves visible card
    mask = info['action_mask']
    reserve_visible = [i for i in range(len(mask)) if mask[i] and 27 <= i <= 38]
    obs, _, _, _, info = env.step(reserve_visible[0])
    
    # From P1's perspective: P0's visible card should be revealed
    sections = get_observation_sections(obs)
    assert sections['opponent'][12] == 1, "P0 should have 1 reserved card"
    
    opponent_card = get_reserved_card_data(sections['opponent_reserved'], 0)
    assert opponent_card is not None, "Should see P0's revealed card"
    assert opponent_card['revealed'] == 1, "P0's card should be marked as revealed"
    
    # Store P0's card info for later verification
    p0_card_tier = opponent_card['tier']
    p0_card_points = opponent_card['points']
    
    # Player 1 reserves blind card (if available)
    mask = info['action_mask']
    reserve_blind = [i for i in range(len(mask)) if mask[i] and 39 <= i <= 41]
    if reserve_blind:
        obs, _, _, _, info = env.step(reserve_blind[0])
        
        # From P0's perspective: P1's blind card should be hidden
        sections = get_observation_sections(obs)
        assert sections['opponent'][12] == 1, "P1 should have 1 reserved card"
        
        # P1's card should be completely hidden (all zeros)
        opponent_reserved_slot_0 = sections['opponent_reserved'][:14]
        assert np.all(opponent_reserved_slot_0 == 0), "P1's hidden card should be all zeros"
        
        # P0 should still see their own card
        assert sections['current_player'][12] == 1, "P0 should have 1 reserved card"
        own_card = get_reserved_card_data(sections['own_reserved'], 0)
        assert own_card is not None, "P0 should see their own card"
        assert own_card['revealed'] == 1, "Own cards always marked as revealed"
        assert own_card['tier'] == p0_card_tier, "Own card info should be consistent"
        assert own_card['points'] == p0_card_points, "Own card info should be consistent"
    
    print("✅ All reserved card observation behaviors working correctly!")


if __name__ == "__main__":
    pytest.main([__file__])
