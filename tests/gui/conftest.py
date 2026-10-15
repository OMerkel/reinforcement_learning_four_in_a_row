"""Shared fixtures for GUI-focused test modules."""
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

from collections.abc import Callable

import matplotlib.pyplot as plt
import pytest

from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.gui import PLAYER_KIND_HUMAN, FourInARowGui, PlayerConfig

plt.switch_backend("Agg")


@pytest.fixture
def input_from() -> Callable[[list[str]], Callable[[str], str]]:
    """Build an input function from a predefined sequence of answers."""

    def _input_from(values: list[str]) -> Callable[[str], str]:
        iterator = iter(values)

        def _inner(_prompt: str) -> str:
            return next(iterator)

        return _inner

    return _input_from


@pytest.fixture
def make_app(
    input_from: Callable[[list[str]], Callable[[str], str]],
) -> Callable[
    [dict[int, PlayerConfig] | None, list[str] | None],
    tuple[FourInARowGui, list[str]],
]:
    """Create a GUI app and capture output lines for assertions."""

    def _make_app(
        players: dict[int, PlayerConfig] | None = None,
        inputs: list[str] | None = None,
    ) -> tuple[FourInARowGui, list[str]]:
        outputs: list[str] = []
        if players is None:
            players = {
                PLAYER_ONE: PlayerConfig(kind=PLAYER_KIND_HUMAN),
                PLAYER_TWO: PlayerConfig(kind=PLAYER_KIND_HUMAN),
            }
        app = FourInARowGui(
            players=players,
            input_fn=input_from([] if inputs is None else inputs),
            output_fn=outputs.append,
        )
        return app, outputs

    return _make_app
