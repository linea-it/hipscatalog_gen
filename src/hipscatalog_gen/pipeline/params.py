"""Normalized parameter containers used by selection modes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MagGlobalParams:
    """Normalized parameters for mag_global selection."""

    mag_min: float
    mag_max: float
    sentinel: float | None = None


@dataclass(frozen=True)
class ScoreGlobalParams:
    """Normalized parameters for score_global selection."""

    score_min: float
    score_max: float
    sentinel: float | None = None


@dataclass(frozen=True)
class ScoreDensityHybridParams:
    """Normalized parameters for score_density_hybrid selection."""

    score_min: float
    score_max: float
    sentinel: float | None = None
