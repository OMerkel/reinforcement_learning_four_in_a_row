# Requirements and Verification Register

## Conventions

Requirements are grouped by architectural ownership. FR is functional; NFR is
nonfunctional. IDs are stable. `implemented` means code and named tests exist,
not that the latest run passed. `planned` means not yet implemented.

Formal English Gherkin places `Feature` before `Scenario` and spells the
requested GIVEN/WHEN/THEN clauses `Given`/`When`/`Then`. Each block below is an
independent feature specification. These specifications link to pytest; a
Cucumber runner is not installed or claimed to execute them.

JSON records in HTML comments are the KPI source of truth. Test selectors use
JUnit dotted module/case names; `module.*` means the entire named suite. Every
matched parametrization must pass. Missing, skipped, or failed cases do not
verify a requirement. Consecutive requirement comments at the start of each
test-suite file list its IDs for reverse traceability and apply to the suite.

## Game Domain: game.py

### FR-GAME-001: Legal Game State
<!-- requirement: {"id":"FR-GAME-001","status":"implemented","tests":["tests.test_game.*"]} -->
Tests: [tests/test_game.py](../tests/test_game.py), suite `tests.test_game`.

```gherkin
Feature: Deterministic seven-column six-row game rules
  Scenario: Apply and undo moves and detect outcomes
    Given an empty board or a legal partially played position
    When players drop tokens and request undo or outcome detection
    Then gravity, turn order, and four-in-a-row outcomes are respected
    And out-of-range, full-column, and empty-undo operations are rejected
```

## History and Game Persistence: history.py, sgf.py

### FR-HISTORY-001: Reversible History
<!-- requirement: {"id":"FR-HISTORY-001","status":"implemented","tests":["tests.test_history_and_sgf.*"]} -->
Tests: [tests/test_history_and_sgf.py](../tests/test_history_and_sgf.py), suite
`tests.test_history_and_sgf`.

```gherkin
Feature: Consistent undo and replay
  Scenario: Traverse recorded history
    Given validated move records and a matching game state
    When a move is undone or replayed
    Then the board, player, and history stacks stay consistent
    And inconsistent records or empty operations are rejected
```

### FR-SGF-001: Validated Game Exchange
<!-- requirement: {"id":"FR-SGF-001","status":"implemented","tests":["tests.test_history_and_sgf.*"]} -->
Tests: [tests/test_history_and_sgf.py](../tests/test_history_and_sgf.py), suite
`tests.test_history_and_sgf`.

```gherkin
Feature: SGF persistence
  Scenario: Round-trip a game or reject an invalid game
    Given a recorded move sequence or imported SGF text
    When it is exported and imported
    Then board size, starting player, move order, and playable moves are validated
    And a valid round-trip preserves the game sequence
```

## Search Engines: engines.py and ai_* Modules

### FR-ENGINE-001: Stable Dispatch
<!-- requirement: {"id":"FR-ENGINE-001","status":"implemented","tests":["tests.test_engine_dispatch.*"]} -->
Tests: [tests/test_engine_dispatch.py](../tests/test_engine_dispatch.py), suite
`tests.test_engine_dispatch`.

```gherkin
Feature: Common engine suggestion contract
  Scenario: Dispatch a configured engine
    Given a valid position and a supported engine name or documented alias
    When a suggestion is requested
    Then the corresponding engine returns an EngineSuggestion
    And an unknown engine name is rejected
```

### FR-RANDOM-001: Random Baseline
<!-- requirement: {"id":"FR-RANDOM-001","status":"implemented","tests":["tests.test_engine_random.*"]} -->
Tests: [tests/test_engine_random.py](../tests/test_engine_random.py), suite
`tests.test_engine_random`.

```gherkin
Feature: Seedable random baseline
  Scenario: Select without strategic evaluation
    Given a legal position and a seeded random generator
    When the random engine chooses a move
    Then its move is legal and reproducible for identical RNG state
    And a board with no legal moves is rejected
```

### FR-MINIMAX-001: Minimax Baseline
<!-- requirement: {"id":"FR-MINIMAX-001","status":"implemented","tests":["tests.test_engine_minimax.*"]} -->
Tests: [tests/test_engine_minimax.py](../tests/test_engine_minimax.py), suite
`tests.test_engine_minimax`.

```gherkin
Feature: Depth-limited minimax evaluation
  Scenario: Evaluate both player perspectives
    Given a legal or terminal position and a valid search depth
    When minimax searches the position
    Then player one maximizes and player two minimizes the upstream evaluation
    And the principal variation starts with the suggested legal move
    And invalid depths and unavailable moves are rejected
```

### FR-NEGAMAX-001: Negamax Baseline
<!-- requirement: {"id":"FR-NEGAMAX-001","status":"implemented","tests":["tests.test_engine_negamax_split.*","tests.test_negamax_module.*"]} -->
Tests: [tests/test_engine_negamax_split.py](../tests/test_engine_negamax_split.py)
and [tests/test_negamax_module.py](../tests/test_negamax_module.py), named suites.

```gherkin
Feature: Perspective-correct negamax search
  Scenario: Compare equivalent search formulations
    Given curated positions and matching minimax and negamax depths
    When the engines search those positions
    Then their evaluations agree within the asserted test tolerances
    And negamax is repeatable and does not mutate the source board
```

### FR-ALPHABETA-001: Pruned Search Baseline
<!-- requirement: {"id":"FR-ALPHABETA-001","status":"implemented","tests":["tests.test_engine_alphabeta.*"]} -->
Tests: [tests/test_engine_alphabeta.py](../tests/test_engine_alphabeta.py), suite
`tests.test_engine_alphabeta`.

```gherkin
Feature: Alpha-beta search
  Scenario: Search with bound-based pruning
    Given a playable position and a positive depth
    When alpha-beta searches
    Then maximizer and minimizer respect search bounds
    And immediate winning moves and principal variations are returned correctly
    And invalid depths and unavailable moves are rejected
```

### FR-MCTS-001: Monte Carlo Baseline
<!-- requirement: {"id":"FR-MCTS-001","status":"implemented","tests":["tests.test_engine_mcts.*"]} -->
Tests: [tests/test_engine_mcts.py](../tests/test_engine_mcts.py), suite
`tests.test_engine_mcts`.

```gherkin
Feature: Seedable UCB Monte Carlo search
  Scenario: Select and back-propagate simulated outcomes
    Given a game position, positive iteration budget, and seeded RNG
    When MCTS expands, selects, and evaluates nodes
    Then suggestions satisfy the legal-move contract
    And UCB perspectives and ancestor statistics remain consistent
    And identical seeded runs are reproducible
```

## Original Presentation: cli.py, gui.py

### FR-CLI-001: Preserve the Original CLI
<!-- requirement: {"id":"FR-CLI-001","status":"implemented","tests":["tests.test_cli.*"]} -->
Tests: [tests/test_cli.py](../tests/test_cli.py), suite `tests.test_cli`.

```gherkin
Feature: Interactive command-driven play
  Scenario: Play and manage a terminal game
    Given a running CLI with a game and history
    When the user moves, suggests, plays an engine, saves, loads, undoes, or replays
    Then each documented command performs its validated operation
    And invalid commands report an error without corrupting the game
```

### FR-GUI-001: Preserve Configurable Graphical Play
<!-- requirement: {"id":"FR-GUI-001","status":"implemented","tests":["tests.gui.test_gui_ai_flow.*","tests.gui.test_gui_click_interaction.*","tests.gui.test_gui_config.*","tests.gui.test_gui_entrypoints.*","tests.gui.test_gui_post_game_prompt.*","tests.gui.test_gui_rendering.*"]} -->
Tests: named suites in [tests/gui](../tests/gui): `test_gui_ai_flow`,
`test_gui_click_interaction`, `test_gui_config`, `test_gui_entrypoints`,
`test_gui_post_game_prompt`, and `test_gui_rendering`.

```gherkin
Feature: Graphical human and search-engine players
  Scenario: Configure seats and play through the GUI
    Given independently configured human or supported upstream AI players
    When the GUI handles clicks, AI turns, rendering, and post-game commands
    Then moves update the shared game and terminal outcomes are displayed
    And new-game, save, and quit workflows remain available
```

## Learning: rl.py

### FR-RL-001: Zero Strategic Prior
<!-- requirement: {"id":"FR-RL-001","status":"implemented","tests":["tests.test_rl.test_cold_start_and_player_relative_encoding","tests.test_rl.test_learning_does_not_use_heuristics"]} -->
Tests: `test_cold_start_and_player_relative_encoding` and
`test_learning_does_not_use_heuristics` in [tests/test_rl.py](../tests/test_rl.py).

```gherkin
Feature: Rules-only cold-start learning
  Scenario: Start without a supplied strategy
    Given a new agent with an empty table
    When an unseen position is encoded or self-play begins
    Then unseen action values are zero and encoding is mover-relative
    And training does not use heuristic evaluation or search teachers
```

### FR-RL-002: Legal Epsilon-Greedy Actions
<!-- requirement: {"id":"FR-RL-002","status":"implemented","tests":["tests.test_rl.test_mask_and_random_ties","tests.test_rl.test_invalid_hyperparameters","tests.test_rl.test_invalid_actions_and_rewards"]} -->
Tests: `test_mask_and_random_ties`, `test_invalid_hyperparameters`, and
`test_invalid_actions_and_rewards` in [tests/test_rl.py](../tests/test_rl.py).

```gherkin
Feature: Masked exploration and exploitation
  Scenario: Select among available columns
    Given an active board, valid hyperparameters, and potentially full columns
    When the agent explores or chooses a maximum-valued action
    Then only legal columns are selected and tied best actions are randomly sampled
    And invalid parameters, rewards, and learning actions are rejected
```

### FR-RL-003: Correct Zero-Sum Targets
<!-- requirement: {"id":"FR-RL-003","status":"implemented","tests":["tests.test_rl.test_opponent_bootstrap_and_terminal_reward","tests.test_rl.test_draw_has_zero_bootstrap_and_is_counted"]} -->
Tests: `test_opponent_bootstrap_and_terminal_reward` and
`test_draw_has_zero_bootstrap_and_is_counted` in [tests/test_rl.py](../tests/test_rl.py).

```gherkin
Feature: Player-relative temporal-difference learning
  Scenario: Learn from a move and its resulting board
    Given a legal transition and the mover's reward
    When the Q update is computed
    Then a nonterminal target subtracts the discounted best legal opponent value
    And a winning terminal target is one and a drawn terminal target is zero
    And terminal targets have no bootstrap and terminal boards have no new action
```

### FR-RL-004: Self-Play Training
<!-- requirement: {"id":"FR-RL-004","status":"implemented","tests":["tests.test_rl.test_seeded_self_play_and_checkpoint_resume","tests.test_rl.test_learning_does_not_use_heuristics","tests.test_rl_cli.test_train_evaluate_play_e2e","tests.test_rl_cli.test_in_process_train_resume_and_module_entrypoint"]} -->
Tests: named self-play cases in [tests/test_rl.py](../tests/test_rl.py) and
`test_train_evaluate_play_e2e`, `test_in_process_train_resume_and_module_entrypoint`
in [tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Shared-agent self-play experience
  Scenario: Train complete games
    Given a seeded agent and a positive episode count
    When self-play training runs
    Then both seats use shared action values and starting tokens alternate
    And each transition is updated and outcome counts sum to completed episodes
```

### FR-RL-005: Versioned Resume
<!-- requirement: {"id":"FR-RL-005","status":"implemented","tests":["tests.test_rl.test_seeded_self_play_and_checkpoint_resume","tests.test_rl.test_reject_invalid_checkpoint","tests.test_rl.test_checkpoint_missing_fields","tests.test_rl_cli.test_train_evaluate_play_e2e"]} -->
Tests: named checkpoint cases in [tests/test_rl.py](../tests/test_rl.py) and
`test_train_evaluate_play_e2e` in [tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Checkpointed experiment continuation
  Scenario: Save, validate, and resume a local agent
    Given a trained agent and a checkpoint path
    When the checkpoint is saved and loaded
    Then version, algorithm, table, parameters, episodes, and RNG state persist
    And resumed learning equals uninterrupted continuation in the same environment
    And malformed or unsupported checkpoints are rejected
```

### NFR-RL-001: Reproducible Learning
<!-- requirement: {"id":"NFR-RL-001","status":"implemented","tests":["tests.test_rl.test_seeded_self_play_and_checkpoint_resume"]} -->
Test: `test_seeded_self_play_and_checkpoint_resume` in [tests/test_rl.py](../tests/test_rl.py).

```gherkin
Feature: Same-environment reproducibility
  Scenario: Repeat an experiment
    Given identical project and Python versions, parameters, and RNG state
    When the same number of self-play episodes completes
    Then statistics, action values, and final RNG state are equal
```

## Learning Presentation and Evaluation: rl_cli.py

### FR-PLAY-001: Independent Seats
<!-- requirement: {"id":"FR-PLAY-001","status":"implemented","tests":["tests.test_rl_cli.test_human_combinations","tests.test_rl_cli.test_different_seat_checkpoints_and_missing_checkpoint","tests.test_rl_cli.test_train_evaluate_play_e2e","tests.test_rl_cli.test_human_input_recovery_and_interrupt","tests.test_rl_cli.test_match_draw_rendering"]} -->
Tests: named human/play/e2e cases in [tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Independently configured human and AI seats
  Scenario: Play through the RL CLI
    Given either seat configured as human, learned agent, or supported search engine
    When a match starts with any required seat checkpoint
    Then both seats use the shared upstream board and may load different agents
    And invalid human input is recoverable and quit interrupts cleanly
```

### FR-EVAL-001: Frozen Balanced Evaluation
<!-- requirement: {"id":"FR-EVAL-001","status":"implemented","tests":["tests.test_rl_cli.test_evaluation_is_balanced_and_frozen","tests.test_rl_cli.test_evaluation_outcomes_and_in_memory_freeze","tests.test_rl_cli.test_search_opponents_are_available","tests.test_rl_cli.test_invalid_cli_arguments"]} -->
Tests: named evaluation cases in [tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Evidence-based policy evaluation
  Scenario: Measure a checkpoint without training
    Given a frozen agent, fixed opponent settings, seed, and positive even game count
    When evaluation runs equally often from both seats
    Then wins, losses, draws, score, and a win-rate interval are reported
    And the original table, episodes, RNG state, and checkpoint remain unchanged
    And the CLI report identifies the checkpoint digest and opponent settings
```

## Environment and Quality: pyproject.toml, scripts/quality.py

### NFR-ENV-001: uv and .venv
<!-- requirement: {"id":"NFR-ENV-001","status":"implemented","tests":["tests.test_rl_cli.test_train_evaluate_play_e2e"]} -->
Test: `test_train_evaluate_play_e2e` in [tests/test_rl_cli.py](../tests/test_rl_cli.py).
Setup gate: `uv sync --locked --extra rl --dev` using `uv.lock`.

```gherkin
Feature: Isolated locked Python development
  Scenario: Execute workflows from the project environment
    Given dependencies resolved by uv into the root .venv and a uv.lock file
    When train, resume, evaluation, and play launch as subprocesses
    Then each command uses the same .venv interpreter as the test runner
```

### NFR-QA-001: Separate Coverage Gates
<!-- requirement: {"id":"NFR-QA-001","status":"implemented","tests":["tests.test_quality.test_independent_coverage_metrics","tests.test_quality.test_quality_cli_success_failure_and_missing_input"]} -->
Tests: named coverage-gate cases in [tests/test_quality.py](../tests/test_quality.py).
Acceptance gate: full suite followed by `uv run python -m scripts.quality`.

```gherkin
Feature: TDD with independently measured coverage
  Scenario: Assess a full unit and end-to-end test run
    Given coverage JSON for production package code and the quality tool
    When statements, branches, and function entries are evaluated
    Then each percentage must be strictly greater than 95
    And exactly 95 percent fails rather than passing through a combined average
```

Function coverage means entry into the first executable statement of each
named function or method, including nested definitions, using AST and coverage
lines. Lambdas, generated dataclass methods, module-level code, and explicitly
excluded lines are outside this **function-entry** metric; measurable statements
and branches still contribute to the other metrics. This is not full path
coverage. Test code is not included in production coverage.

### NFR-TRACE-001: Bidirectional Evidence
<!-- requirement: {"id":"NFR-TRACE-001","status":"implemented","tests":["tests.test_quality.test_requirements_use_passed_tests_and_suite_headers","tests.test_quality.test_missing_links_and_unknown_reverse_ids","tests.test_quality.test_reject_invalid_register"]} -->
Tests: named traceability cases in [tests/test_quality.py](../tests/test_quality.py).

```gherkin
Feature: Test-result-backed requirement KPI
  Scenario: Reconcile register, test headers, and test results
    Given unique requirement records and current full-run JUnit results
    When coverage and reverse traceability are computed
    Then only requirements whose entire named evidence passed are verified
    And missing tests, missing header IDs, and unknown reverse IDs are reported
    And planned requirements remain in the overall coverage denominator
    And all implemented requirements must be verified to pass the quality gate
```

## Neural Learning and Planned Strong-Play Assessment

Neural mechanics and bounded snapshots are implemented. Historical opponents,
Strength certification, historical opponents, and resource budgets remain
planned; PUCT policy/value mechanics and budgeted search are now implemented.

### FR-NEURAL-001: Generalizing Q Function
<!-- requirement: {"id":"FR-NEURAL-001","status":"implemented","tests":["tests.test_rl_nn.*","tests.test_rl_cli.test_nn_cli_e2e","tests.test_rl_cli.test_neural_evaluation_preserves_weights_and_replay"]} -->
Tests: [tests/test_rl_nn.py](../tests/test_rl_nn.py), including
`test_neural_zero_prior_and_encoding`, `test_double_dqn_opponent_target`,
`test_neural_legal_action_mask`, `test_replay_reflection_and_target_sync`, and
`test_tiny_terminal_dataset_can_be_fit`; neural e2e and frozen evaluation tests
in [tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Neural action-value learning
  Scenario: Generalize beyond memorized positions
    Given a random network and self-play experience without teacher labels
    When masked opponent-relative Double DQN minibatch updates run
    Then replay and frozen target predictions train the online network
    And illegal actions never enter selection or target maxima
    And reflection reverses columns and actions consistently
    And a tiny terminal dataset can be fitted without heuristic labels
```

### FR-NN-SAVE-001: Bounded Neural Checkpoints
<!-- requirement: {"id":"FR-NN-SAVE-001","status":"implemented","tests":["tests.test_rl_nn.test_neural_checkpoint_resume_and_fixed_model_size","tests.test_rl_nn.test_reject_malformed_neural_payload","tests.test_rl_cli.test_nn_cli_e2e","tests.test_rl_cli.test_neural_cli_default_export_and_guards"]} -->
Tests: named resume, malformed-payload, export, and CLI guard tests in
[tests/test_rl_nn.py](../tests/test_rl_nn.py) and
[tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Bounded model storage and resumable neural experiments
  Scenario: Persist training or export a frozen playing model
    Given a fixed-size neural network and a replay buffer with a finite capacity
    When a training snapshot or inference-only model is saved
    Then a snapshot preserves online and target weights, optimizer, replay, RNG and counters
    And replay never exceeds its configured capacity
    And inference export excludes replay, target and optimizer state
    And inference-only files cannot resume training
    And malformed snapshots are rejected with restricted tensor loading
    And existing checkpoints are not replaced without resume or explicit overwrite
```

Within the tested CPU/Python/PyTorch environment, save/load continuation produces
identical parameters and replay. Cross-version or cross-hardware identity is not
promised. The default 24967-parameter model export is below 150000 bytes; full
training snapshots are larger but replay growth stops at the configured cap.

### FR-OUTCOME-001: Completed-Game Credit Assignment
<!-- requirement: {"id":"FR-OUTCOME-001","status":"implemented","tests":["tests.test_rl_nn.test_episode_outcomes_teach_both_players","tests.test_rl_nn.test_discounted_episode_outcomes_and_legacy_mode","tests.test_rl_nn.test_outcome_resume_and_partial_episode_guard","tests.test_rl_nn.test_version_one_snapshot_preserves_td_only_training","tests.test_rl_nn.test_invalid_outcome_settings_and_returns","tests.test_rl_nn.test_outcome_collection_is_bounded_and_draw_returns_are_zero","tests.test_rl_nn.test_full_episode_fits_winning_and_losing_moves","tests.test_rl_cli.test_tactical_benchmark_cli_and_outcome_resume","tests.test_rl_cli.test_invalid_strength_workflow_arguments"]} -->
Tests: named outcome, loss-fitting, bounded-episode, and legacy-resume cases in
[tests/test_rl_nn.py](../tests/test_rl_nn.py) and workflow cases in
[tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Direct learning from complete game outcomes
  Scenario: Assign mover-relative returns after self-play finishes
    Given a neural agent and a completed legal self-play game
    When the outcome is attached to its collected actions
    Then winner actions have positive discounted returns
    And loser actions have negative discounted returns and draws have zero
    And reflected returns enter bounded replay without search teacher labels
    And the configured weight mixes outcome and masked Double DQN targets
    And checkpoints preserve labels and exact same-environment continuation
    And old snapshots retain TD-only behavior unless explicitly upgraded
```

### FR-TACTIC-001: Frozen Tactical Diagnostics
<!-- requirement: {"id":"FR-TACTIC-001","status":"implemented","tests":["tests.test_rl_benchmark.*","tests.test_rl_cli.test_tactical_benchmark_cli_and_outcome_resume","tests.test_rl_cli.test_invalid_strength_workflow_arguments"]} -->
Tests: [tests/test_rl_benchmark.py](../tests/test_rl_benchmark.py) and named
benchmark/resume workflow cases in [tests/test_rl_cli.py](../tests/test_rl_cli.py).

```gherkin
Feature: Evaluation-only tactical policy comparison
  Scenario: Score frozen checkpoints over mirrored and color-swapped cases
    Given rule-verified win, block, safety, and double-threat positions
    When checkpoints and untrained policies are evaluated over specified seeds
    Then category accuracy, failed cases, selected columns and hashes are reported
    And policy parameters, replay, RNG, counters and source files stay unchanged
    And no diagnostic case or expected move is supplied to the trainer
```

### FR-POOL-001: Historical Opponents
<!-- requirement: {"id":"FR-POOL-001","status":"implemented","tests":["tests.test_rl_puct.test_historical_pool_freezes_and_caps_archived_weights","tests.test_rl_puct.test_historical_pool_resume_is_exact","tests.test_rl_puct.test_disabled_pool_and_invalid_history_settings","tests.test_rl_puct.test_pool_cli_and_resume_controls","tests.test_rl_puct.test_version_one_puct_snapshot_keeps_original_self_play","tests.test_rl_puct.test_corrupted_historical_pool_is_rejected","tests.test_rl_puct.test_history_training_does_not_read_diagnostic_labels","tests.test_rl_puct.test_wrong_pool_controls_and_opening_prefixes_are_rejected"]} -->
Tests: named frozen-history, capacity, resume, CLI, corruption and isolation
cases in [tests/test_rl_puct.py](../tests/test_rl_puct.py).

```gherkin
Feature: Historical-opponent self-play
  Scenario: Train against frozen policies
    Given a capped history pool and declared archival intervals
    When opponents are sampled during a completed training game
    Then opponents remain frozen during games
    And learner seats alternate and only learner decisions generate targets
    And archival weights, sampling RNG and counters survive exact resume
    And old snapshots keep self-play unless pool settings are explicitly enabled
    And archived snapshots are not automatically promoted as playing champions
```

The initial policy and scheduled completed-game snapshots enter a bounded
FIFO history. This is training diversity, not a strength-promotion gate.
Candidate promotion remains a separate FR-STRENGTH-001 responsibility.

### FR-FORK-DEFENSE-001: Opponent Double-Threat Avoidance
<!-- requirement: {"id":"FR-FORK-DEFENSE-001","status":"implemented","tests":["tests.test_rl_benchmark.*"]} -->
Tests: [tests/test_rl_benchmark.py](../tests/test_rl_benchmark.py), including
`test_fork_defense_rejects_moves_that_allow_double_threats` and
`test_old_tactical_subset_remains_comparable`.

```gherkin
Feature: Evaluation-only opponent fork defense
  Scenario: Avoid giving the opponent two immediate winning columns
    Given a legal pinned position with a solvable fork threat
    When legal moves and opponent replies are checked using the rules
    Then acceptable moves prevent unanswerable opponent double threats
    And immediate winning counter-replies are not labelled as losses
    And mirrored and color-swapped variants have consistent labels
    And the original tactical subset remains separately comparable
    And none of these expected actions enter the training replay
```

### NFR-EVAL-DIVERSITY-001: Paired Legal Opening Evaluation
<!-- requirement: {"id":"NFR-EVAL-DIVERSITY-001","status":"implemented","tests":["tests.test_rl_puct.test_opening_evaluation_is_paired_and_frozen","tests.test_rl_puct.test_wrong_pool_controls_and_opening_prefixes_are_rejected"]} -->
Tests: named paired-opening and prefix-validation cases in
[tests/test_rl_puct.py](../tests/test_rl_puct.py).

```gherkin
Feature: Reproducible broader evaluation positions
  Scenario: Play both seats from seeded legal prefixes
    Given a frozen agent, evaluation seed and a prefix length from zero to six
    When balanced-seat evaluation generates opening positions
    Then each opening is used for both learner token seats
    And reports record the exact zero-based opening moves
    And source weights, replay, RNG and checkpoints remain unchanged
    And zero-prefix evaluation preserves the previous workflow
```

### FR-PUCT-001: Alternative Policy-Value Learning
<!-- requirement: {"id":"FR-PUCT-001","status":"implemented","tests":["tests.test_rl_puct.*"]} -->
Tests: [tests/test_rl_puct.py](../tests/test_rl_puct.py), including policy masks,
selection-score arithmetic, terminal backup, visit targets, self-play,
two-head fitting, and budgeted CLI workflows.

```gherkin
Feature: AlphaZero-style policy-value self-play
  Scenario: Generate search-improved training targets
    Given an untrained policy-value network and game rules
    When PUCT self-play produces visits and terminal outcomes
    Then masked normalized visits train the policy head
    And mover-relative final outcomes train value without heuristic labels
    And terminal leaves use exact rules and backup negates each player switch
    And evaluation separates trained search, untrained search and unaided policy
```

### FR-PUCT-SAVE-001: Policy-Value Checkpoint Compatibility
<!-- requirement: {"id":"FR-PUCT-SAVE-001","status":"implemented","tests":["tests.test_rl_puct.test_self_play_labels_and_exact_resume","tests.test_rl_puct.test_reject_invalid_puct_checkpoint","tests.test_rl_puct.test_puct_cli_train_evaluate_export_and_play","tests.test_rl_puct.test_puct_cli_rejects_wrong_controls_and_corruption"]} -->
Tests: named resume, corruption, backend-control and export cases in
[tests/test_rl_puct.py](../tests/test_rl_puct.py).

```gherkin
Feature: Separate resumable policy-value checkpoints
  Scenario: Persist a PUCT agent without reinterpreting DQN weights
    Given a seeded policy-value learner and capped replay
    When its snapshot or compact model is saved and loaded
    Then policy and value heads retain their predictions
    And full snapshots retain optimizer, replay, counters and RNG state
    And same-environment continuation is exact
    And inference exports cannot resume and malformed payloads are rejected
    And existing DQN and JSON snapshots retain their original backend
```

### FR-STRENGTH-001: Honest Strong-Play Assessment
<!-- requirement: {"id":"FR-STRENGTH-001","status":"planned","tests":[]} -->
Planned test: `test_held_out_solver_position_accuracy` and multi-seed benchmarks.

```gherkin
Feature: Predeclared strength assessment
  Scenario: Compare against independent strong references
    Given held-out solver positions, fixed opponents, and declared budgets and targets
    When multi-seed balanced-seat evaluation completes
    Then mistake rates, tactical accuracy, and opponent results are published
    And master-strength claims require the declared criteria to be met
```

Numerical mastery criteria must be agreed before this gate is implemented.
Connect Four is solved: a second player cannot force a draw against a perfect
first player. No guaranteed overall win rate or training budget is promised.

### NFR-PERF-001: Declared Resource Budget
<!-- requirement: {"id":"NFR-PERF-001","status":"planned","tests":[]} -->
Planned tests: `test_inference_latency_budget`, `test_training_memory_budget`.

```gherkin
Feature: Bounded experimental resource use
  Scenario: Benchmark on declared hardware
    Given a recorded CPU or GPU and fixed inference budget
    When inference and a 10000-episode training run are measured
    Then provisional p95 move latency is at most two seconds
    And peak process memory is at most four GiB
    And hardware, throughput, and checkpoint size accompany the result
```

These provisional budgets require target-hardware confirmation; they are not
current performance claims.

## KPI Workflow

```powershell
uv sync --locked --extra rl --dev
uv run --extra rl pytest
uv run python -m scripts.quality
uv run zensical build
```

The full suite writes `coverage.json` and `test-results.xml`; the quality command
writes `quality-report.json`. It fails if any code metric is at or below 95,
an implemented requirement lacks passing evidence, or a trace link is inconsistent.
Use artifacts from the same run and revision: the tool cannot authenticate stale
artifacts supplied by a caller. Documentation builds are a separate required gate.

Overall coverage is verified requirements divided by all registered requirements.
Implemented coverage uses only implemented requirements as its denominator.
Neither measures playing strength. A baseline can pass its delivery gate while
overall scope coverage remains below 100 percent because stronger self-play
and strength certification are planned.

Before implementing new behavior, add a scenario and a failing named test and
put its ID in the suite header. Implement the smallest slice and run the focused
test. Update status when code exists; the current test report determines
verification. Update architecture and learning rationale when contracts change.
