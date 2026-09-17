"""Ensemble over the outcome models.

Weights are *measured*, not chosen. The backtesting engine produces
out-of-sample probabilities for every component model; ``fit_weights`` then
finds the mixture over those models that minimises log loss on that history,
subject to the weights being non-negative and summing to one. If no backtest
has been run, the ensemble falls back to equal weights and says so, rather
than pretending to an optimisation it has not performed.

Combining scorelines: only some models produce a full scoreline matrix. The
ensemble takes the best available matrix and rescales its home/draw/away
regions to match the ensemble's 1X2 probabilities. The result stays a proper
distribution and stays consistent with the headline numbers, so the "most
likely score" can never contradict the "most likely result".
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from ..logging_setup import get_logger
from .base import (
    ModelPrediction,
    PredictionModel,
    PredictionRequest,
    TrainingContext,
    outcome_probs_from_matrix,
)

log = get_logger("models.ensemble")

OUTCOMES = ("home", "draw", "away")
EPS = 1e-9


def fit_weights(
    model_probabilities: dict[str, np.ndarray],
    outcomes: np.ndarray,
    *,
    l2: float = 0.01,
) -> tuple[dict[str, float], dict]:
    """Find simplex weights minimising log loss over stored backtest results.

    ``model_probabilities`` maps model key -> array of shape (n, 3) aligned
    with ``outcomes``, an integer array of realised outcome indices.
    """
    keys = sorted(model_probabilities)
    if not keys:
        return {}, {"error": "no model probabilities supplied"}
    stack = np.stack([model_probabilities[k] for k in keys])  # (m, n, 3)
    n = stack.shape[1]
    if n < 30:
        equal = {k: 1.0 / len(keys) for k in keys}
        return equal, {"method": "equal", "reason": f"only {n} scored matches"}

    rows = np.arange(n)

    def loss(raw):
        # Softmax keeps the weights on the simplex without constrained solvers.
        w = np.exp(raw - raw.max())
        w = w / w.sum()
        mixed = np.tensordot(w, stack, axes=(0, 0))       # (n, 3)
        picked = np.clip(mixed[rows, outcomes], EPS, 1.0)
        return float(-np.mean(np.log(picked)) + l2 * np.sum(w ** 2))

    result = minimize(loss, np.zeros(len(keys)), method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
    raw = result.x
    w = np.exp(raw - raw.max())
    w = w / w.sum()
    weights = {k: float(round(v, 4)) for k, v in zip(keys, w)}

    # Report what the optimisation actually bought over equal weighting.
    def logloss_for(weight_vector):
        mixed = np.tensordot(weight_vector, stack, axes=(0, 0))
        picked = np.clip(mixed[rows, outcomes], EPS, 1.0)
        return float(-np.mean(np.log(picked)))

    equal_vec = np.full(len(keys), 1.0 / len(keys))
    diagnostics = {
        "method": "log-loss minimisation",
        "scored_matches": int(n),
        "log_loss_weighted": round(logloss_for(w), 5),
        "log_loss_equal": round(logloss_for(equal_vec), 5),
        "converged": bool(result.success),
        "per_model_log_loss": {
            k: round(
                float(-np.mean(np.log(np.clip(model_probabilities[k][rows, outcomes], EPS, 1.0)))), 5
            )
            for k in keys
        },
    }
    return weights, diagnostics


def rescale_matrix_to_outcomes(matrix: np.ndarray, target: dict[str, float]) -> np.ndarray:
    """Rescale a scoreline matrix so its 1X2 margins equal ``target``."""
    current = outcome_probs_from_matrix(matrix)
    out = matrix.astype(float).copy()
    n = out.shape[0]
    idx = np.indices(out.shape)
    home_mask = idx[0] > idx[1]
    draw_mask = idx[0] == idx[1]
    away_mask = idx[0] < idx[1]
    for mask, key in ((home_mask, "home"), (draw_mask, "draw"), (away_mask, "away")):
        if current[key] > EPS:
            out[mask] *= target[key] / current[key]
    total = out.sum()
    return out / total if total > 0 else out


class EnsembleModel(PredictionModel):
    key = "ensemble"
    name = "Performance-weighted ensemble"
    produces = ("1x2", "goals", "correct_score", "btts", "clean_sheet")
    yields_score_matrix = True

    def __init__(self, models: Sequence[PredictionModel],
                 weights: Optional[dict[str, float]] = None) -> None:
        super().__init__()
        self.models = list(models)
        self.weights = dict(weights or {})
        self.weight_source = "equal" if not weights else "measured"

    def fit(self, ctx: TrainingContext) -> "EnsembleModel":
        for model in self.models:
            model.fit(ctx)
        self.fitted = any(m.fitted for m in self.models)
        self.training_cutoff = ctx.as_of
        self.n_training_matches = max((m.n_training_matches for m in self.models), default=0)
        self.fit_diagnostics = {
            "components": {m.key: m.fit_diagnostics for m in self.models},
            "weight_source": self.weight_source,
            "weights": self.weights,
        }
        return self

    def set_weights(self, weights: dict[str, float], source: str = "measured") -> None:
        self.weights = dict(weights)
        self.weight_source = source

    def _effective_weights(self, usable: Sequence[str]) -> dict[str, float]:
        if not usable:
            return {}
        if self.weights:
            selected = {k: self.weights.get(k, 0.0) for k in usable}
            total = sum(selected.values())
            if total > 0:
                return {k: v / total for k, v in selected.items()}
            log.debug("stored weights gave zero mass to the usable models; using equal weights")
        return {k: 1.0 / len(usable) for k in usable}

    def predict(self, request: PredictionRequest) -> ModelPrediction:
        component_predictions: dict[str, ModelPrediction] = {}
        notes: dict[str, str] = {}
        for model in self.models:
            pred = model.predict(request)
            if pred.sufficient_data and pred.outcome_probs:
                component_predictions[model.key] = pred
            else:
                notes[model.key] = pred.note or "no output"

        if not component_predictions:
            detail = "; ".join(f"{k}: {v}" for k, v in notes.items())
            return self._insufficient(f"no component model could produce a prediction ({detail})")

        weights = self._effective_weights(list(component_predictions))
        combined = {
            outcome: sum(
                weights[key] * component_predictions[key].outcome_probs[outcome]
                for key in component_predictions
            )
            for outcome in OUTCOMES
        }
        total = sum(combined.values())
        combined = {k: v / total for k, v in combined.items()}

        # Prefer the scoreline matrix from the most heavily weighted model that
        # produces one, then reconcile it with the ensemble's 1X2 split.
        matrix = None
        source_key = None
        for key in sorted(component_predictions, key=lambda k: weights.get(k, 0), reverse=True):
            if component_predictions[key].score_matrix is not None:
                matrix = component_predictions[key].score_matrix
                source_key = key
                break

        expected_home = expected_away = None
        if matrix is not None:
            matrix = rescale_matrix_to_outcomes(matrix, combined)
            goals = np.arange(matrix.shape[0])
            expected_home = float((matrix.sum(axis=1) * goals).sum())
            expected_away = float((matrix.sum(axis=0) * goals).sum())

        return ModelPrediction(
            model_key=self.key,
            sufficient_data=True,
            outcome_probs=combined,
            score_matrix=matrix,
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
            diagnostics={
                "weights": {k: round(v, 4) for k, v in weights.items()},
                "weight_source": self.weight_source,
                "components": {
                    k: {kk: round(vv, 4) for kk, vv in p.outcome_probs.items()}
                    for k, p in component_predictions.items()
                },
                "score_matrix_from": source_key,
                "unavailable_components": notes,
            },
        )
