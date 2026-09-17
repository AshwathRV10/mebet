"""Sampling windows and time-decay weighting.

The brief is explicit that "last 5 matches" is not automatically the right
sample. So the feature layer computes several windows in parallel - 5, 10,
20, season-to-date, and a time-decayed all-history figure - and exposes all
of them. Which one carries weight is then a modelling decision measured by
backtesting, not an assumption baked into the features.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

#: Named windows in matches. ``None`` means "everything available".
WINDOWS: dict[str, Optional[int]] = {
    "last5": 5,
    "last10": 10,
    "last20": 20,
    "all": None,
}


def half_life_weight(age_days: float, half_life_days: float) -> float:
    """Exponential decay: a match ``half_life_days`` old counts half as much.

    This is the Dixon-Coles style down-weighting of old results. It lets recent
    form matter without letting it erase long-run quality, because old matches
    keep a non-zero weight rather than dropping out of the sample entirely.
    """
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, max(age_days, 0.0) / half_life_days)


def decay_weights(dates: Sequence[dt.date], as_of: dt.date, half_life_days: float) -> list[float]:
    return [half_life_weight((as_of - d).days, half_life_days) for d in dates]


@dataclass
class WeightedStat:
    """A statistic summarised over a sample, with the sample size kept.

    Sample size travels with the value because a 1.9 goals-per-game average
    over 3 matches and over 40 matches are not the same claim, and the
    confidence and data-quality layers need to tell them apart.
    """

    value: Optional[float]
    n: int
    effective_n: float = 0.0
    window: str = ""

    @property
    def is_usable(self) -> bool:
        return self.value is not None and self.n > 0

    def as_dict(self) -> dict:
        return {"value": self.value, "n": self.n, "effective_n": round(self.effective_n, 2),
                "window": self.window}


def weighted_mean(
    values: Sequence[Optional[float]],
    weights: Optional[Sequence[float]] = None,
    *,
    window: str = "",
) -> WeightedStat:
    """Mean over observed values only; missing entries are excluded, not zeroed."""
    pairs = [
        (v, (weights[i] if weights is not None else 1.0))
        for i, v in enumerate(values)
        if v is not None
    ]
    if not pairs:
        return WeightedStat(None, 0, 0.0, window)
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return WeightedStat(None, len(pairs), 0.0, window)
    mean = sum(v * w for v, w in pairs) / total_w
    return WeightedStat(mean, len(pairs), total_w, window)


def rate(numerators: Sequence[Optional[float]], *, window: str = "") -> WeightedStat:
    """Proportion of observed entries that are truthy (e.g. clean sheets)."""
    observed = [v for v in numerators if v is not None]
    if not observed:
        return WeightedStat(None, 0, 0.0, window)
    return WeightedStat(sum(1 for v in observed if v) / len(observed), len(observed),
                        float(len(observed)), window)
