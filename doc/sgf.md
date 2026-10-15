# SGF - The Smart Game Format

This page documents how the project stores and restores games using SGF.

For architectural context and module boundaries, see
[software_architecture.md](software_architecture.md).

## Purpose in this project

SGF is used as a compact, text-based persistence format for Four in a Row game
history.

In this codebase it supports two practical goals:

- saving a played game to disk
- loading a saved move sequence back into the program

The implementation lives in `four_in_a_row/sgf.py`.

## What is stored

The SGF support in this project stores the game as:

- root metadata describing the game and board
- a starting player indicator
- a linear sequence of move nodes

This is intentionally focused on the needs of the current application. It is not
trying to model every SGF feature or every possible variation tree.

## Export format

`export_sgf(move_columns, starting_player=PLAYER_ONE)` serializes a move list to
SGF text.

### Root properties currently written

The exporter writes a root node containing these properties:

- `GM[41]`
- `FF[4]`
- `CA[UTF-8]`
- `AP[four-in-a-row:1.0]`
- `SZ[7:6]`
- `PL[B]` or `PL[W]`

The generated content is wrapped in a single SGF game tree and ends with a final
newline.

### Move encoding

Each move is written as one SGF node:

- `;B[x]` for Player 1
- `;W[x]` for Player 2

The project stores only the **column**, not the row. This matches the game rules,
because the row is determined automatically by gravity when the move is replayed.

### Column mapping

Columns are encoded as letters:

- `a` -> column `0`
- `b` -> column `1`
- ...
- `g` -> column `6`

The helper `_column_to_sgf()` validates that exported columns stay within the
legal board width.

## Import behavior

`import_sgf(content)` parses SGF text and returns:

- the starting player
- the ordered list of move columns

The parser is intentionally strict in a few important places so that invalid or
incompatible data fails early.

### Board size validation

The importer requires `SZ[7:6]` to be present.

If not, it raises `SgfError` with an unsupported board size message.

This guarantees that imported files match the fixed board dimensions used by the
game implementation.

### Starting player handling

The parser looks for `PL[B]` or `PL[W]`.

- if present, it sets the starting player accordingly
- if missing, it defaults to Player 1

### Move parsing

Move nodes are extracted with a regex matching entries of the form:

- `;B[a]`
- `;W[d]`

For each parsed node, the importer:

1. checks that the move belongs to the player whose turn is expected
2. converts the SGF coordinate into a 0-based column index
3. appends the column to the move list
4. flips the expected player for the next move

## Validation rules

The SGF layer raises `SgfError`, a specialized `ValueError`, for malformed or
unsupported input.

Current validation includes:

- invalid export starting player
- out-of-range export column
- unsupported board size
- invalid move order
- out-of-range imported column
- replay-based legality validation of the full move sequence

### Invalid move order

If the SGF content starts from `PL[B]`, the first move must be by `B`.

Likewise, the importer expects strict alternation between players. If a move node
breaks that order, import fails immediately.

### Replay-based legality validation

After parsing the move list, the importer rebuilds the game by replaying every
move on a fresh `GameState`.

This catches SGF files that are syntactically valid but impossible to play on a
real board. For example, import now fails if the move sequence:

- overfills a column
- uses an otherwise illegal move while reconstructing the position

In those cases, `import_sgf()` raises `SgfError` instead of deferring the
failure until a later load or rebuild step.

## Example structure

A short game might look conceptually like this:

```text
(;GM[41]FF[4]CA[UTF-8]AP[four-in-a-row:1.0]SZ[7:6]PL[B];B[a];W[b];B[c])
```

This means:

- Four in a Row board size `7x6`
- Player 1 starts
- moves are columns `0`, `1`, `2`

## Why rows are not stored

Connect Four style games do not need explicit row coordinates when the move
history is replayed in order.

Given the column sequence and the starting player:

- move order is deterministic
- gravity determines the row automatically
- the board can be reconstructed exactly

This keeps the exported data simple while still allowing faithful replay.

## Relationship to the history system

The SGF module stores and restores the move sequence, while the history layer is
responsible for in-memory undo and replay behavior during a running session.

Together they provide:

- short-term interaction via undo and replay
- long-term persistence via save and load

## Test coverage

The tests currently verify that:

- SGF export and import round-trip correctly
- invalid board sizes are rejected
- invalid player order is rejected
- invalid export arguments are rejected
- out-of-range imported columns are rejected
- unplayable parsed move sequences are rejected during replay validation

These tests help ensure the format remains stable as the project evolves.

## Current scope and limitations

The implementation is intentionally small and focused.

It currently supports:

- a single linear move sequence
- one fixed board size: `7x6`
- column-based move encoding only

It does not currently attempt to support:

- SGF variations and branches
- comments, annotations or analysis markup
- alternate board sizes
- richer metadata beyond the written root properties

That narrow scope is appropriate for the current CLI game and keeps the parser
easy to reason about.
