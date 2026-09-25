"""Model contract.

A model is anything that, given data available before a cutoff, produces a
probability distribution over match outcomes. Models differ in what they can
produce: Dixon-Coles yields a full scoreline matrix, Elo only a 1X2 split.
Rather than forcing every model to fake the richer output, the contract lets
each declare what it provides and the engine combines what is available.

Every model must also be able to say "not enough data". A model that always
answers is a model that will sometimes answer from noise.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..db.index import MatchIndex
from ..db.repository import AsOfRepository


@dataclass
class TrainingContext:
    """What a model is allowed to learn from."""

    repo: AsOfRepository
    competition_ids: Sequence[int]
    as_of: dt.datetime
    half_life_days: float = 270.0
    min_matches: int = 60
    seasons_back: int = 5
    #: Shared in-memory index. Models that need features at many historical
    #: cutoffs pass it to the repositories they create, which is the
    #: difference between a fit taking seconds and taking minutes.
    index: Optional[MatchIndex] = None

    def training_matches(self):
        since = dt.date(self.as_of.year - self.seasons_back, 7, 1)
        return self.repo.completed_matches(competition_ids=self.competition_ids, since=since)


@dataclass
class PredictionRequest:
    home_team_id: int
    away_team_id: int
    competition_id: int
    kickoff: dt.datetime
    neutral_venue: bool = False
    features: Optional[object] = None      # MatchFeatures, when built
    adjustments: dict = field(default_factory=dict)


@dataclass
class ModelPrediction:
    model_key: str
    sufficient_data: bool = True
    note: str = ""
    #: home/draw/away probabilities; always present when sufficient_data.
    outcome_probs: Optional[dict[str, float]] = None
    #: Joint distribution over (home goals, away goals). Only goal models.
    score_matrix: Optional[np.ndarray] = None
    expected_home_goals: Optional[float] = None
    expected_away_goals: Optional[float] = None
    #: Count markets (corners, cards): per-side expectations and the
    #: distribution over the match total, from index 0 upwards.
    expected_home_count: Optional[float] = None
    expected_away_count: Optional[float] = None
    total_distribution: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.outcome_probs:
            total = sum(self.outcome_probs.values())
            if not 0.98 <= total <= 1.02:
                raise ValueError(f"{self.model_key}: outcome probabilities sum to {total:.4f}")


class PredictionModel(ABC):
    key: str = ""
    name: str = ""
    #: What this model can speak to.
    produces: tuple[str, ...] = ("1x2",)
    #: Whether it yields a full scoreline distribution.
    yields_score_matrix: bool = False
    sport: str = "football"

    def __init__(self, **params) -> None:
        self.params = params
        self.fitted = False
        self.fit_diagnostics: dict = {}
        self.training_cutoff: Optional[dt.datetime] = None
        self.n_training_matches = 0

    @abstractmethod
    def fit(self, ctx: TrainingContext) -> "PredictionModel":
        ...

    @abstractmethod
    def predict(self, request: PredictionRequest) -> ModelPrediction:
        ...

    def describe(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "produces": list(self.produces),
            "params": {k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool))},
            "training_cutoff": self.training_cutoff.isoformat() if self.training_cutoff else None,
            "n_training_matches": self.n_training_matches,
            "diagnostics": self.fit_diagnostics,
        }

    def _insufficient(self, note: str) -> ModelPrediction:
        return ModelPrediction(model_key=self.key, sufficient_data=False, note=note)


def outcome_probs_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    """Collapse a scoreline matrix into home/draw/away."""
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    total = home + draw + away
    if total <= 0:
        raise ValueError("score matrix has no mass")
    return {"home": home / total, "draw": draw / total, "away": away / total}
