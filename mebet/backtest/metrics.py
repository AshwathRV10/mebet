"""Scoring rules and calibration.

Accuracy alone is a poor test of a probabilistic model: predicting the home
team every time scores respectably in football and tells you nothing. The
metrics that matter are the proper scoring rules - log loss and Brier - plus
calibration, which asks the question that actually matters for this project:
when the model says 60%, does it happen 60% of the time?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

EPS = 1e-15
OUTCOME_ORDER = ("home", "draw", "away")


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log probability assigned to what actually happened."""
    if len(outcomes) == 0:
        return float("nan")
    picked = np.clip(probabilities[np.arange(len(outcomes)), outcomes], EPS, 1.0)
    return float(-np.mean(np.log(picked)))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Multi-class Brier: mean squared error against the one-hot truth."""
    if len(outcomes) == 0:
        return float("nan")
    truth = np.zeros_like(probabilities)
    truth[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def accuracy(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    if len(outcomes) == 0:
        return float("nan")
    return float(np.mean(np.argmax(probabilities, axis=1) == outcomes))


def binary_log_loss(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1 - EPS)
    y = np.asarray(outcomes, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_brier(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
    p, a = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(p - a))) if len(a) else float("nan")


def rmse(predicted: Sequence[float], actual: Sequence[float]) -> float:
    p, a = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((p - a) ** 2))) if len(a) else float("nan")


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    def as_dict(self) -> dict:
        return {
            "range": [round(self.lower, 2), round(self.upper, 2)],
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 4),
            "observed": round(self.observed_rate, 4),
            "gap": round(self.observed_rate - self.mean_predicted, 4),
        }


def calibration(probabilities: Sequence[float], outcomes: Sequence[int],
                bins: int = 10) -> list[CalibrationBin]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    out: list[CalibrationBin] = []
    if len(y) == 0:
        return out
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(CalibrationBin(float(lo), float(hi), n,
                                  float(p[mask].mean()), float(y[mask].mean())))
    return out


def expected_calibration_error(probabilities: Sequence[float],
                               outcomes: Sequence[int], bins: int = 10) -> float:
    """Average |predicted - observed| weighted by bin population.

    Zero means the stated probabilities match reality; large values mean the
    model is over- or under-confident regardless of its accuracy.
    """
    binned = calibration(probabilities, outcomes, bins)
    total = sum(b.count for b in binned)
    if not total:
        return float("nan")
    return float(
        sum(b.count * abs(b.observed_rate - b.mean_predicted) for b in binned) / total
    )


@dataclass
class MarketScore:
    market: str
    model_key: str
    n: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    calibration: list[CalibrationBin] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "market": self.market,
            "model": self.model_key,
            "n": self.n,
            "metrics": {k: (round(v, 5) if v == v else None) for k, v in self.metrics.items()},
            "calibration": [b.as_dict() for b in self.calibration],
        }


def score_outcome_market(probabilities: np.ndarray, outcomes: np.ndarray,
                         model_key: str, market: str = "1x2") -> MarketScore:
    score = MarketScore(market=market, model_key=model_key, n=len(outcomes))
    if len(outcomes) == 0:
        return score
    score.metrics = {
        "accuracy": accuracy(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
        "brier": brier_score(probabilities, outcomes),
    }
    # Calibration measured on the home-win probability, the market's main axis.
    home_probs = probabilities[:, 0]
    home_actual = (outcomes == 0).astype(int)
    score.calibration = calibration(home_probs, home_actual)
    score.metrics["calibration_error"] = expected_calibration_error(home_probs, home_actual)
    return score


def score_binary_market(probabilities: Sequence[float], outcomes: Sequence[int],
                        model_key: str, market: str) -> MarketScore:
    score = MarketScore(market=market, model_key=model_key, n=len(outcomes))
    if not len(outcomes):
        return score
    score.metrics = {
        "log_loss": binary_log_loss(probabilities, outcomes),
        "brier": binary_brier(probabilities, outcomes),
        "accuracy": float(np.mean((np.asarray(probabilities) >= 0.5).astype(int)
                                  == np.asarray(outcomes))),
        "calibration_error": expected_calibration_error(probabilities, outcomes),
        "base_rate": float(np.mean(outcomes)),
    }
    score.calibration = calibration(probabilities, outcomes)
    return score


def score_numeric_market(predicted: Sequence[float], actual: Sequence[float],
                         model_key: str, market: str) -> MarketScore:
    score = MarketScore(market=market, model_key=model_key, n=len(actual))
    if not len(actual):
        return score
    score.metrics = {
        "mae": mae(predicted, actual),
        "rmse": rmse(predicted, actual),
        "mean_predicted": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
        "bias": float(np.mean(np.asarray(predicted) - np.asarray(actual))),
    }
    return score


def baseline_scores(outcomes: np.ndarray) -> dict[str, float]:
    """Reference points a model must beat to be worth anything.

    ``always_home`` is the naive rule; ``base_rates`` predicts the historical
    frequency of each outcome every time. A model that cannot beat base rates
    has learned nothing.
    """
    if len(outcomes) == 0:
        return {}
    n = len(outcomes)
    rates = np.array([(outcomes == i).mean() for i in range(3)])
    constant = np.tile(rates, (n, 1))
    always_home = np.tile(np.array([1 - 2 * EPS, EPS, EPS]), (n, 1))
    return {
        "base_rate_log_loss": log_loss(constant, outcomes),
        "base_rate_brier": brier_score(constant, outcomes),
        "base_rate_accuracy": float(rates.max()),
        "always_home_accuracy": accuracy(always_home, outcomes),
        "home_rate": float(rates[0]),
        "draw_rate": float(rates[1]),
        "away_rate": float(rates[2]),
    }
