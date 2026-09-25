"""Model registry - the seam for adding new models.

A new model is added by registering it here. The backtesting engine will then
evaluate it alongside the others automatically, and the ensemble will consider
it. Nothing else needs to change.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from ..logging_setup import get_logger
from .base import PredictionModel
from .counts import CardsModel, CornersModel
from .dixon_coles import DixonColesModel
from .elo import EloModel
from .ensemble import EnsembleModel
from .logistic import LogisticOutcomeModel

log = get_logger("models.registry")

#: Identifies the modelling configuration. Ensemble weights measured by a
#: backtest are only reused by the same version: weights learned for one set of
#: models say nothing reliable about a different set.
MODEL_VERSION = "2026.09-shot-ratings"

#: Ensemble weights used when no backtest of the current model version has
#: been run. Fitted by log-loss minimisation on walk-forward predictions for
#: 2021-22 and 2022-23 across all five leagues (3,583 matches) - seasons that
#: precede the window used to report accuracy. Reproduce with
#: ``scripts/accuracy_study/final_test.py``.
DEFAULT_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "dixon_coles": 0.3893,
    "elo": 0.3961,
    "logistic": 0.2147,
}

#: Models that produce match-outcome probabilities and can be ensembled.
OUTCOME_MODELS: dict[str, Callable[[], PredictionModel]] = {
    "dixon_coles": DixonColesModel,
    "elo": EloModel,
    "logistic": LogisticOutcomeModel,
}

#: Models for auxiliary count markets, evaluated on their own terms.
COUNT_MODELS: dict[str, Callable[[], PredictionModel]] = {
    "corners": CornersModel,
    "cards": CardsModel,
}


def build_outcome_models(keys: Optional[Iterable[str]] = None) -> list[PredictionModel]:
    keys = list(keys) if keys is not None else list(OUTCOME_MODELS)
    out = []
    for key in keys:
        factory = OUTCOME_MODELS.get(key)
        if factory is None:
            log.warning("unknown outcome model: %s", key)
            continue
        out.append(factory())
    return out


def build_count_models(keys: Optional[Iterable[str]] = None) -> list[PredictionModel]:
    keys = list(keys) if keys is not None else list(COUNT_MODELS)
    return [COUNT_MODELS[k]() for k in keys if k in COUNT_MODELS]


def build_ensemble(weights: Optional[dict[str, float]] = None,
                   keys: Optional[Iterable[str]] = None) -> EnsembleModel:
    """Ensemble with measured weights when given, else the validated defaults."""
    if weights:
        return EnsembleModel(build_outcome_models(keys), weights=weights,
                             weight_source="measured by backtest")
    return EnsembleModel(build_outcome_models(keys), weights=dict(DEFAULT_ENSEMBLE_WEIGHTS),
                         weight_source="validated default")


def all_model_keys() -> list[str]:
    return list(OUTCOME_MODELS) + list(COUNT_MODELS) + ["ensemble"]
