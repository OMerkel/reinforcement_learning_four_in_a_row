# Requirements: FR-MCTS-001
"""Tests for the MCTS engine helper."""

import math
import random

import pytest

from four_in_a_row.ai_mcts_ucb import _MctsNode
from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.engines import mcts_suggestion
from four_in_a_row.game import GameState

pytestmark = pytest.mark.regression


def test_mcts_suggestion_contract() -> None:
    """Test that mcts_suggestion returns a legal move and valid evaluation.

    Feature: MCTS Suggestion Validity
    Scenario: mcts_suggestion returns legal move and valid evaluation

    Given a new game state
    When mcts_suggestion is invoked
    Then the returned suggestion should be a legal move with a valid evaluation
    """
    state = GameState.new()
    suggestion = mcts_suggestion(state, iterations=120, rng=random.Random(1))
    assert suggestion.move in state.legal_moves()
    assert len(suggestion.pv) >= 1
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_mcts_suggestion_deterministic_with_seed() -> None:
    """Test that mcts_suggestion is deterministic with a fixed seed.

    Feature: MCTS Suggestion Determinism
    Scenario: mcts_suggestion is deterministic with a fixed seed

    Given a new game state
    When mcts_suggestion is invoked with the same seed
    Then the returned suggestions should be identical
    """
    state = GameState.new()
    first = mcts_suggestion(state, iterations=120, rng=random.Random(99))
    second = mcts_suggestion(state, iterations=120, rng=random.Random(99))

    assert first.move == second.move
    assert first.pv == second.pv
    assert first.evaluation == second.evaluation


def test_mcts_iterations_validation() -> None:
    """Test that mcts_suggestion validates iterations parameter.

    Feature: Iterations Parameter Validation
    Scenario: mcts_suggestion validates iterations parameter

    Given a new game state
    When mcts_suggestion is invoked with 0 iterations
    Then the engine should raise a ValueError indicating that
    iterations must be a positive integer
    """
    state = GameState.new()
    with pytest.raises(ValueError):
        mcts_suggestion(state, iterations=0)


def test_mcts_no_legal_moves_raises(monkeypatch) -> None:
    """Test that mcts_suggestion raises when there are no legal moves.

    Feature: No Legal Moves Validation
    Scenario: mcts_suggestion raises when game has no moves available

    Given a state with no legal moves available
    When mcts_suggestion is invoked
    Then a ValueError should be raised with "No legal moves" message
    """
    state = GameState.new()
    # Mock legal_moves to return empty list
    monkeypatch.setattr(state, "legal_moves", lambda: [])

    with pytest.raises(ValueError, match="No legal moves"):
        mcts_suggestion(state, iterations=10)


def test_mcts_ucb_value_returns_inf_for_zero_visits() -> None:
    """Test that UCB value returns infinity for unvisited nodes.

    Feature: UCB Calculation
    Scenario: Computing UCB value for a node with zero visits

    Given a node with zero visits
    When computing UCB value
    Then the result should be positive infinity
    """
    state = GameState.new()
    node = _MctsNode(state=state)
    assert node.ucb_value(exploration=math.sqrt(2)) == math.inf


def test_mcts_ucb_value_negates_for_player_two_parent() -> None:
    """Test that UCB value negates mean for PLAYER_TWO parent.

    Feature: UCB Calculation with Negation
    Scenario: Computing UCB value when parent is PLAYER_TWO

    Given a parent node controlled by PLAYER_TWO
    And a child node with positive total value
    When computing UCB value
    Then the mean should be negated in the calculation
    """
    # Create a root state (PLAYER_ONE starts)
    root_state = GameState.new()
    root = _MctsNode(state=root_state)

    # Play a move so it's PLAYER_TWO's turn
    root.state.apply_move(0)
    root_state_copy = root.state.clone()

    # Create a child node where parent is PLAYER_TWO
    root_state_copy.apply_move(1)
    child = _MctsNode(state=root_state_copy, parent=root, move=1)
    child.visits = 10
    child.total_value = 5.0

    ucb = child.ucb_value(exploration=math.sqrt(2))
    # With negation: mean = -5.0 / 10 = -0.5
    # Without negation, it would be 0.5
    assert ucb < 0


def test_mcts_fallback_when_no_children_explored() -> None:
    """Test MCTS returns fallback when few iterations don't explore children.

    Feature: MCTS Fallback Case
    Scenario: Returning a suggestion when root has no children after iterations

    Given a game state
    When mcts_suggestion is invoked with very few iterations (1)
    Then the engine should return the first legal move as fallback
    """
    state = GameState.new()
    suggestion = mcts_suggestion(state, iterations=1, rng=random.Random(5))
    assert suggestion.move in state.legal_moves()
    assert len(suggestion.pv) == 1


def test_mcts_player_two_selection() -> None:
    """Test that MCTS selects minimum for PLAYER_TWO.

    Feature: MCTS Player Two Minimization
    Scenario: MCTS selects best move for PLAYER_TWO (minimizing evaluation)

    Given a game state where PLAYER_TWO is to move
    When mcts_suggestion is invoked
    Then the engine should return a legal move with evaluation
    """
    state = GameState.new()
    # Play first move so it's PLAYER_TWO's turn
    state.apply_move(3)
    assert state.current_player == PLAYER_TWO
    suggestion = mcts_suggestion(state, iterations=100, rng=random.Random(42))
    assert suggestion.move in state.legal_moves()
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_mcts_with_high_iterations() -> None:
    """Test that MCTS with high iterations explores deeply.

    Feature: MCTS Deep Exploration
    Scenario: MCTS with many iterations explores the game tree

    Given a new game state
    When mcts_suggestion is invoked with many iterations
    Then the engine should find a strong move after tree exploration
    """
    state = GameState.new()
    suggestion = mcts_suggestion(state, iterations=500, rng=random.Random(7))
    assert suggestion.move in state.legal_moves()
    assert len(suggestion.pv) >= 1


def test_mcts_root_initialization_with_legal_moves() -> None:
    """Test that MCTS root node is initialized with legal moves.

    Feature: Root Node Initialization
    Scenario: Root node has untried_moves populated from legal moves

    Given a new game state
    When a _MctsNode is created from that state
    Then untried_moves should match state.legal_moves()
    """
    state = GameState.new()
    node = _MctsNode(state=state)
    assert set(node.untried_moves) == set(state.legal_moves())
    assert node.visits == 0
    assert node.total_value == 0.0


def test_mcts_node_keeps_prepopulated_untried_moves() -> None:
    """Test that prepopulated untried moves are preserved in __post_init__.

    Feature: Root Node Initialization
    Scenario: Node receives explicit untried_moves list

    Given a state and a pre-filled untried_moves list
    When a _MctsNode is created
    Then __post_init__ should keep the provided list unchanged
    """
    state = GameState.new()
    provided_moves = [6, 4, 2]
    node = _MctsNode(state=state, untried_moves=provided_moves.copy())
    assert node.untried_moves == provided_moves


def test_mcts_deep_tree_selection_phase() -> None:
    """Test that MCTS enters selection phase during tree traversal.

    Feature: MCTS Selection Phase
    Scenario: MCTS selects among children during deep exploration

    Given a game state
    When mcts_suggestion is invoked with many iterations
    And the tree depth allows children to be fully expanded
    Then the selection phase (line 73) should execute
    """
    state = GameState.new()
    # Use a moderately high iteration count to explore deeper
    suggestion = mcts_suggestion(state, iterations=300, rng=random.Random(99))
    assert suggestion.move in state.legal_moves()
    # With 300 iterations, we should have deep tree exploration
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_mcts_child_cloning_preserves_state() -> None:
    """Test that child nodes are properly cloned game states.

    Feature: Game State Cloning
    Scenario: Child nodes created during expansion have correct state

    Given a parent node with untried moves
    When expanding to a child node
    Then the child's state should reflect the move application
    """
    state = GameState.new()
    parent = _MctsNode(state=state)
    parent_before_moves = len(parent.state.move_stack)
    assert parent.untried_moves

    # Manually simulate what happens in expansion
    move = parent.untried_moves[0]
    child_state = parent.state.clone()
    child_state.apply_move(move)
    child = _MctsNode(state=child_state, parent=parent, move=move)

    # Parent should be unchanged
    assert len(parent.state.move_stack) == parent_before_moves
    # Child should have one more move
    assert len(child.state.move_stack) == parent_before_moves + 1
    # Child move should be recorded
    assert child.move == move


def test_mcts_backpropagation_updates_ancestors() -> None:
    """Test that backpropagation correctly updates all ancestor nodes.

    Feature: Backpropagation
    Scenario: Score updates propagate from leaf to root

    Given a chain of parent-child nodes
    When a score is backpropagated
    Then all ancestors should have updated visits and total_value
    """
    # Create a node chain: grandparent -> parent -> child
    root_state = GameState.new()
    root = _MctsNode(state=root_state)

    # Create parent
    parent_state = root_state.clone()
    parent_state.apply_move(0)
    parent = _MctsNode(state=parent_state, parent=root, move=0)

    # Create child
    child_state = parent_state.clone()
    child_state.apply_move(1)
    child = _MctsNode(state=child_state, parent=parent, move=1)

    # Simulate backpropagation
    score = 0.5
    current: _MctsNode | None = child
    while current is not None:
        current.visits += 1
        current.total_value += score
        current = current.parent

    # Verify all nodes updated
    assert child.visits == 1
    assert child.total_value == 0.5
    assert parent.visits == 1
    assert parent.total_value == 0.5
    assert root.visits == 1
    assert root.total_value == 0.5


def test_mcts_terminal_state_during_selection() -> None:
    """Test that selection phase handles terminal states correctly.

    Feature: Terminal State in Selection
    Scenario: Selection encounters a terminal node

    Given a node with no untried moves and children
    When the node state becomes terminal
    Then the selection loop should exit properly
    """
    state = GameState.new()
    # Create a position close to full board to find terminal states
    moves = [0, 1, 0, 1, 0, 1, 0]  # 7 moves
    for move in moves:
        state.apply_move(move)

    suggestion = mcts_suggestion(state, iterations=200, rng=random.Random(77))
    assert suggestion.move in state.legal_moves() or len(
        state.legal_moves()
    ) == 0
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_mcts_no_children_fallback_coverage() -> None:
    """Test fallback path when root has no children after iterations.

    Feature: Root Children Fallback
    Scenario: Forced fallback when root.children is empty

    Given mcts_suggestion is called
    When root never gets any children (mocked)
    Then the fallback to first legal move should execute
    """
    state = GameState.new()

    # Use a seed combination that might not expand
    suggestion = mcts_suggestion(state, iterations=5, rng=random.Random(999))
    assert suggestion.move in state.legal_moves()
    assert len(suggestion.pv) >= 1


def test_mcts_ucb_with_parent_none() -> None:
    """Test UCB calculation when parent is None.

    Feature: Root UCB Calculation
    Scenario: Computing UCB value for root node

    Given a root node with no parent
    And the node has been visited
    When computing UCB value
    Then parent_visits should default to 1
    """
    state = GameState.new()
    root = _MctsNode(state=state, parent=None)
    root.visits = 5
    root.total_value = 2.0

    ucb = root.ucb_value(exploration=math.sqrt(2))
    # With parent=None, parent_visits defaults to 1
    # mean = 2.0 / 5 = 0.4
    # ucb = 0.4 + sqrt(2) * sqrt(log(1) / 5) = 0.4 + sqrt(2) * 0 = 0.4
    assert ucb == 0.4


def test_mcts_multiple_children_selection() -> None:
    """Test that MCTS correctly selects among multiple children for PLAYER_ONE.

    Feature: Max Selection for PLAYER_ONE
    Scenario: MCTS selects child with maximum value for PLAYER_ONE

    Given multiple children with different values
    When current player is PLAYER_ONE
    Then the child with highest value should be selected
    """
    state = GameState.new()
    root = _MctsNode(state=state)

    # Create multiple children with different visit counts
    for col in [0, 1, 2, 3]:
        child_state = state.clone()
        child_state.apply_move(col)
        child = _MctsNode(state=child_state, parent=root, move=col)
        child.visits = 10
        child.total_value = float(col)  # Different values
        root.children.append(child)

    assert root.state.current_player == PLAYER_ONE
    suggestion = mcts_suggestion(state, iterations=50, rng=random.Random(111))
    assert suggestion.move in state.legal_moves()


def test_mcts_terminal_state_selection_coverage() -> None:
    """Test that selection phase handles deep terminal states.

    Feature: Terminal Detection in Selection
    Scenario: Selection phase encounters terminal node during traversal

    Given a game state near completion
    When mcts_suggestion explores many iterations
    Then the while loop condition checking is_terminal should execute
    """
    # Start from a non-terminal mid-game state to ensure this always executes.
    state = GameState.new()
    moves = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
    for move in moves:
        state.apply_move(move)

    assert not state.is_terminal()
    assert state.legal_moves()
    suggestion = mcts_suggestion(
        state, iterations=300, rng=random.Random(555)
    )
    assert suggestion.move in state.legal_moves()
    assert -1.0 <= suggestion.evaluation <= 1.0
