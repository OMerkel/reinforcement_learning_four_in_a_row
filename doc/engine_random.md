# Random engine

`random_suggestion()` is the simplest strategy:

1. collect legal moves
2. choose one using a supplied or internal pseudo-random generator
3. apply the move on a cloned state
4. evaluate the resulting position

Characteristics:

- very fast
- useful as a baseline
- deterministic in tests when a seeded `random.Random` instance is passed in
- returns a one-move PV: the selected move only

This engine is helpful for smoke testing, UI validation and low-cost examples.
