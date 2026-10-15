"""Shared EngineSuggestion type used by engines and AI modules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineSuggestion:
    """Structured suggestion from an engine."""

    engine: str
    move: int
    pv: list[int]
    evaluation: float
