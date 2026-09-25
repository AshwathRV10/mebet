"""Fast walk-forward evaluation for comparing model configurations.

``BacktestRunner`` is the full, persisted backtest. This module is the lighter
instrument used to *choose* between configurations, and it is built around
the two ways model tuning usually fools itself:

*   **Tuning on the reported data.** Pick the best of fifty settings on the
    same matches you then report, and the reported number is optimistic by
    construction. So evaluation takes an explicit window, and the workflow is:
    tune on a validation window, then score the finalists once on a later,
    untouched test window.
*   **Mistaking noise for improvement.** A log-loss difference of 0.003 over a
    few hundred matches is well inside chance. ``paired_comparison`` scores two
    configurations on the identical set of matches and reports a bootstrap
    interval for the difference, resampling whole weeks rather than single
    matches because fixtures in the same round share conditions.

Every fit happens at a checkpoint no later than the matches it predicts, via
the same ``AsOfRepository`` cutoff rules as everything else.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from sqlalchemy.orm import Session

from ..db.index import MatchIndex
from ..db.models import Competition
from ..db.repository import AsOfRepository
from ..models.base import PredictionModel, PredictionRequest, TrainingContext

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
EPS = 1e-12


@dataclass
class MatchScore:
    match_id: int
    competition: str
    date: dt.date
    outcome: int
    probs: tuple[float, float, float]
    over25: Optional[float] = None
    btts: Optional[float] = None
    total_goals: int = 0
    both_scored: bool = False

    @property
    def log_loss(self) -> float:
        return -float(np.log(max(self.probs[self.outcome], EPS)))

    @property
    def brier(self) -> float:
        truth = np.zeros(3)
        truth[self.outcome] = 1.0
        return float(np.sum((np.array(self.probs) - truth) ** 2))


@dataclass
class EvaluationResult:
    label: str
    scores: dict[int, MatchScore] = field(default_factory=dict)
    declined: int = 0
    fits: int = 0

    def subset(self, match_ids: Iterable[int]) -> list[MatchScore]:
        return [self.scores[m] for m in match_ids if m in self.scores]

    def summary(self, match_ids: Optional[Iterable[int]] = None) -> dict:
        rows = self.subset(match_ids) if match_ids is not None else list(self.scores.values())
        if not rows:
            return {"n": 0}
        ll = np.array([r.log_loss for r in rows])
        acc = np.mean([int(np.argmax(r.probs)) == r.outcome for r in rows])
        out = {
            "n": len(rows),
            "log_loss": float(ll.mean()),
            "brier": float(np.mean([r.brier for r in rows])),
            "accuracy": float(acc),
        }
        ou = [r for r in rows if r.over25 is not None]
        if ou:
            p = np.clip([r.over25 for r in ou], EPS, 1 - EPS)
            y = np.array([r.total_goals > 2.5 for r in ou], dtype=float)
            out["ou25_log_loss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return out


def walk_forward(
    session: Session,
    index: MatchIndex,
    competition_key: str,
    start: dt.date,
    end: dt.date,
    model_factory: Callable[[], PredictionModel],
    *,
    refit_days: int = 7,
    label: str = "",
    needs_features: bool = False,
) -> EvaluationResult:
    """Predict every played match in [start, end] from data before it."""
    comp = session.query(Competition).filter_by(key=competition_key).one()
    matches = sorted(
        (m for m in index.by_competition.get(comp.id, ())
         if start <= m.kickoff_date <= end and m.result is not None),
        key=lambda m: (m.kickoff_date, m.id),
    )
    result = EvaluationResult(label=label)
    model: Optional[PredictionModel] = None
    checkpoint: Optional[dt.date] = None

    for m in matches:
        if checkpoint is None or (m.kickoff_date - checkpoint).days >= refit_days:
            checkpoint = m.kickoff_date
            cutoff = dt.datetime.combine(checkpoint, dt.time(0, 0))
            repo = AsOfRepository(session, cutoff, index=index)
            model = model_factory()
            model.fit(TrainingContext(repo=repo, competition_ids=[comp.id], as_of=cutoff,
                                      index=index))
            result.fits += 1

        kickoff = m.kickoff_utc or dt.datetime.combine(m.kickoff_date, dt.time(0, 0))
        features = None
        if needs_features:
            from ..features.football import build_match_features

            features = build_match_features(
                AsOfRepository(session, kickoff, index=index),
                m.home_team_id, m.away_team_id, comp.id,
            )
        pred = model.predict(PredictionRequest(
            home_team_id=m.home_team_id, away_team_id=m.away_team_id,
            competition_id=comp.id, kickoff=kickoff, features=features,
        ))
        if not pred.sufficient_data or not pred.outcome_probs:
            result.declined += 1
            continue

        over25 = btts = None
        if pred.score_matrix is not None:
            mat = pred.score_matrix
            idx = np.add.outer(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
            over25 = float(mat[idx >= 3].sum())
            btts = float(mat[1:, 1:].sum())
        p = pred.outcome_probs
        result.scores[m.id] = MatchScore(
            match_id=m.id, competition=competition_key, date=m.kickoff_date,
            outcome=OUTCOME_INDEX[m.result],
            probs=(p["home"], p["draw"], p["away"]),
            over25=over25, btts=btts,
            total_goals=m.ft_home_goals + m.ft_away_goals,
            both_scored=m.ft_home_goals > 0 and m.ft_away_goals > 0,
        )
    return result


def paired_comparison(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    metric: str = "log_loss",
    n_boot: int = 4000,
    seed: int = 7,
) -> dict:
    """Candidate minus baseline on shared matches, with a week-block bootstrap.

    Negative means the candidate is better (lower loss).
    """
    shared = sorted(set(baseline.scores) & set(candidate.scores))
    if not shared:
        return {"n": 0}

    def per_match(result: EvaluationResult) -> np.ndarray:
        rows = result.subset(shared)
        if metric == "log_loss":
            return np.array([r.log_loss for r in rows])
        if metric == "brier":
            return np.array([r.brier for r in rows])
        if metric == "ou25_log_loss":
            p = np.clip([r.over25 for r in rows], EPS, 1 - EPS)
            y = np.array([r.total_goals > 2.5 for r in rows], dtype=float)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        raise ValueError(metric)

    diff = per_match(candidate) - per_match(baseline)
    weeks = np.array([
        f"{baseline.scores[m].competition}:{baseline.scores[m].date.isocalendar()[:2]}"
        for m in shared
    ])
    groups = {w: np.where(weeks == w)[0] for w in np.unique(weeks)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        boots[b] = diff[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(shared),
        "baseline": float(per_match(baseline).mean()),
        "candidate": float(per_match(candidate).mean()),
        "diff": float(diff.mean()),
        "ci95": (float(lo), float(hi)),
        "p_better": float(np.mean(boots < 0)),
    }


def merge(results: Sequence[EvaluationResult], label: str = "") -> EvaluationResult:
    """Combine per-competition results into one pooled result."""
    out = EvaluationResult(label=label or (results[0].label if results else ""))
    for r in results:
        out.scores.update(r.scores)
        out.declined += r.declined
        out.fits += r.fits
    return out
