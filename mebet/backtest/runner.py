"""Walk-forward backtesting.

The procedure follows the brief exactly: take a historical match whose result
is known, pretend it has not happened, build the prediction from information
that existed beforehand, then compare against what occurred.

Two separate cutoffs keep it honest:

*   **Model fitting** happens at periodic checkpoints. A model used to predict
    a match is fitted on data up to the checkpoint at or before that match -
    never later. Refitting before every single match would be the theoretical
    ideal but is computationally wasteful; refitting periodically is
    conservative in the right direction, because the model is slightly
    *stale* rather than slightly informed.
*   **Feature building** happens at each match's own kickoff, so form and
    standings are exactly what was visible that day.

Every read in both paths goes through ``AsOfRepository``, which enforces the
cutoff in the query and verifies the rows it returns.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.index import MatchIndex
from ..db.models import BacktestMetric, BacktestPrediction, BacktestRun, Competition, Match
from ..db.repository import AsOfRepository
from ..logging_setup import get_logger
from ..models.base import PredictionRequest, TrainingContext
from ..models.ensemble import EnsembleModel, fit_weights
from ..models.registry import MODEL_VERSION, build_count_models, build_outcome_models
from ..features.football import build_match_features
from . import metrics as M

log = get_logger("backtest.runner")

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


@dataclass
class BacktestConfig:
    competition_key: str
    from_date: dt.date
    to_date: dt.date
    refit_days: int = 30
    seasons_back: int = 5
    half_life_days: float = 270.0
    model_keys: Optional[Sequence[str]] = None
    include_counts: bool = True
    include_ensemble: bool = True
    #: Goal lines scored as binary markets.
    goal_lines: tuple[float, ...] = (1.5, 2.5, 3.5)
    corner_lines: tuple[float, ...] = (9.5, 10.5)
    card_lines: tuple[float, ...] = (3.5, 4.5)

    def label(self) -> str:
        return (f"{self.competition_key} {self.from_date}..{self.to_date} "
                f"refit={self.refit_days}d")


@dataclass
class BacktestResult:
    config: BacktestConfig
    n_matches: int = 0
    scores: list[M.MarketScore] = field(default_factory=list)
    baselines: dict = field(default_factory=dict)
    ensemble_weights: dict = field(default_factory=dict)
    ensemble_diagnostics: dict = field(default_factory=dict)
    #: Weights fitted on an earlier slice and scored on a later one, so the
    #: benefit of weighting is measured out of sample rather than asserted.
    ensemble_validation: dict = field(default_factory=dict)
    run_id: Optional[int] = None
    skipped: dict = field(default_factory=dict)

    def by_model(self, market: str = "1x2") -> dict[str, dict]:
        return {
            s.model_key: s.metrics for s in self.scores if s.market == market
        }

    def best_model(self, market: str = "1x2", metric: str = "log_loss") -> Optional[str]:
        candidates = [
            s for s in self.scores
            if s.market == market and metric in s.metrics and s.metrics[metric] == s.metrics[metric]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.metrics[metric]).model_key

    def as_dict(self) -> dict:
        return {
            "config": {
                "competition": self.config.competition_key,
                "from": self.config.from_date.isoformat(),
                "to": self.config.to_date.isoformat(),
                "refit_days": self.config.refit_days,
                "seasons_back": self.config.seasons_back,
            },
            "n_matches": self.n_matches,
            "baselines": {k: round(v, 5) for k, v in self.baselines.items()},
            "scores": [s.as_dict() for s in self.scores],
            "ensemble_weights": self.ensemble_weights,
            "ensemble_diagnostics": self.ensemble_diagnostics,
            "ensemble_validation": self.ensemble_validation,
            "skipped": self.skipped,
            "run_id": self.run_id,
        }


class BacktestRunner:
    def __init__(self, session: Session, sport: str = "football",
                 index: Optional[MatchIndex] = None) -> None:
        self.session = session
        self.sport = sport
        self.index = index or MatchIndex(session, sport=sport)

    def run(self, config: BacktestConfig, *, persist: bool = True,
            progress: bool = True) -> BacktestResult:
        competition = self.session.execute(
            select(Competition).where(Competition.sport == self.sport,
                                      Competition.key == config.competition_key)
        ).scalars().first()
        if competition is None:
            raise ValueError(f"unknown competition {config.competition_key}")

        test_matches = list(self.session.execute(
            select(Match).where(
                Match.sport == self.sport,
                Match.competition_id == competition.id,
                Match.status == "played",
                Match.kickoff_date >= config.from_date,
                Match.kickoff_date <= config.to_date,
                Match.ft_home_goals.is_not(None),
            ).order_by(Match.kickoff_date, Match.id)
        ).scalars())

        result = BacktestResult(config=config)
        if not test_matches:
            log.warning("no matches to score for %s", config.label())
            return result

        # Accumulators, keyed by model then market.
        outcome_probs: dict[str, list[list[float]]] = {}
        outcome_truth: list[int] = []
        goal_totals_pred: dict[str, list[float]] = {}
        goal_totals_actual: list[float] = []
        binary: dict[tuple[str, str], tuple[list[float], list[int]]] = {}
        numeric: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
        stored_rows: list[BacktestPrediction] = []
        skipped: dict[str, int] = {}

        models = build_outcome_models(config.model_keys)
        count_models = build_count_models() if config.include_counts else []
        ensemble = EnsembleModel(models) if config.include_ensemble else None

        checkpoint: Optional[dt.date] = None
        scored = 0

        for match in test_matches:
            kickoff = match.kickoff_utc or dt.datetime.combine(match.kickoff_date, dt.time(0, 0))

            # Refit when the checkpoint has expired. Fitting uses the
            # checkpoint date, which is always at or before this match.
            if checkpoint is None or (match.kickoff_date - checkpoint).days >= config.refit_days:
                checkpoint = match.kickoff_date
                fit_cutoff = dt.datetime.combine(checkpoint, dt.time(0, 0))
                fit_repo = AsOfRepository(self.session, fit_cutoff, sport=self.sport,
                                          index=self.index)
                ctx = TrainingContext(
                    repo=fit_repo, competition_ids=[competition.id], as_of=fit_cutoff,
                    half_life_days=config.half_life_days, seasons_back=config.seasons_back,
                    index=self.index,
                )
                for model in models:
                    model.fit(ctx)
                for model in count_models:
                    model.fit(ctx)
                if progress:
                    log.info("refit at %s (%d/%d matches scored)",
                             checkpoint, scored, len(test_matches))

            # Features at this match's own kickoff.
            repo = AsOfRepository(self.session, kickoff, sport=self.sport, index=self.index)
            try:
                features = build_match_features(repo, match.home_team_id, match.away_team_id,
                                                competition.id)
            except Exception as exc:  # noqa: BLE001
                skipped["feature_error"] = skipped.get("feature_error", 0) + 1
                log.debug("features failed for match %s: %s", match.id, exc)
                continue

            request = PredictionRequest(
                home_team_id=match.home_team_id,
                away_team_id=match.away_team_id,
                competition_id=competition.id,
                kickoff=kickoff,
                neutral_venue=match.neutral_venue,
                features=features,
                adjustments={"referee": match.referee or ""},
            )

            truth = OUTCOME_INDEX.get(match.result)
            if truth is None:
                skipped["no_result"] = skipped.get("no_result", 0) + 1
                continue

            predictions = {}
            for model in models:
                pred = model.predict(request)
                if pred.sufficient_data and pred.outcome_probs:
                    predictions[model.key] = pred
                else:
                    skipped[f"{model.key}_insufficient"] = (
                        skipped.get(f"{model.key}_insufficient", 0) + 1
                    )

            if not predictions:
                skipped["no_model_output"] = skipped.get("no_model_output", 0) + 1
                continue

            if ensemble is not None:
                ens = ensemble.predict(request)
                if ens.sufficient_data and ens.outcome_probs:
                    predictions["ensemble"] = ens

            # Only score matches where every model produced output, so the
            # comparison between models is like for like.
            expected_keys = {m.key for m in models}
            if not expected_keys.issubset(predictions.keys()):
                skipped["incomplete_model_set"] = skipped.get("incomplete_model_set", 0) + 1
                continue

            outcome_truth.append(truth)
            actual_total = match.ft_home_goals + match.ft_away_goals
            goal_totals_actual.append(float(actual_total))
            scored += 1

            for key, pred in predictions.items():
                probs = [pred.outcome_probs[o] for o in M.OUTCOME_ORDER]
                outcome_probs.setdefault(key, []).append(probs)
                if persist:
                    stored_rows.append(BacktestPrediction(
                        match_id=match.id, model_key=key, market="1x2",
                        selection=M.OUTCOME_ORDER[int(np.argmax(probs))],
                        probability=float(max(probs)),
                        actual_outcome=float(truth),
                        correct=bool(int(np.argmax(probs)) == truth),
                    ))

                if pred.score_matrix is not None:
                    matrix = pred.score_matrix
                    totals = np.zeros(matrix.shape[0] + matrix.shape[1] - 1)
                    for h in range(matrix.shape[0]):
                        for a in range(matrix.shape[1]):
                            totals[h + a] += matrix[h, a]
                    expected_total = float(
                        (pred.expected_home_goals or 0) + (pred.expected_away_goals or 0)
                    )
                    goal_totals_pred.setdefault(key, []).append(expected_total)

                    for line in config.goal_lines:
                        over_prob = float(totals[int(np.ceil(line)):].sum())
                        bucket = binary.setdefault((key, f"over_under_{line}"), ([], []))
                        bucket[0].append(over_prob)
                        bucket[1].append(int(actual_total > line))

                    btts_prob = float(matrix[1:, 1:].sum())
                    bucket = binary.setdefault((key, "btts"), ([], []))
                    bucket[0].append(btts_prob)
                    bucket[1].append(int(match.ft_home_goals > 0 and match.ft_away_goals > 0))

            # Count markets, scored against what the match actually recorded.
            home_line, away_line = repo.stat_pair(self.index.by_id.get(match.id) or match)
            for model in count_models:
                pred = model.predict(request)
                if not pred.sufficient_data or pred.total_distribution is None:
                    continue
                if model.key == "corners":
                    if home_line is None or away_line is None:
                        continue
                    if home_line.corners is None or away_line.corners is None:
                        continue
                    actual = float(home_line.corners + away_line.corners)
                    lines = config.corner_lines
                else:
                    if home_line is None or away_line is None:
                        continue
                    if home_line.yellow_cards is None or away_line.yellow_cards is None:
                        continue
                    actual = float(
                        (home_line.yellow_cards or 0) + (away_line.yellow_cards or 0)
                        + 2 * ((home_line.red_cards or 0) + (away_line.red_cards or 0))
                    )
                    lines = config.card_lines

                predicted_total = (pred.expected_home_count or 0) + (pred.expected_away_count or 0)
                bucket = numeric.setdefault((model.key, f"expected_{model.key}"), ([], []))
                bucket[0].append(float(predicted_total))
                bucket[1].append(actual)

                dist = pred.total_distribution
                for line in lines:
                    threshold = int(np.ceil(line))
                    over_prob = float(dist[threshold:].sum()) if threshold < len(dist) else 0.0
                    b = binary.setdefault((model.key, f"{model.key}_over_under_{line}"), ([], []))
                    b[0].append(over_prob)
                    b[1].append(int(actual > line))

        result.n_matches = scored
        result.skipped = skipped
        if scored == 0:
            log.warning("backtest scored no matches: %s", skipped)
            return result

        truth_array = np.array(outcome_truth)
        result.baselines = M.baseline_scores(truth_array)

        for key, rows in outcome_probs.items():
            probs = np.array(rows)
            result.scores.append(M.score_outcome_market(probs, truth_array, key))

        for key, preds in goal_totals_pred.items():
            result.scores.append(M.score_numeric_market(
                preds, goal_totals_actual[: len(preds)], key, "expected_goals"
            ))

        for (key, market), (probs, actuals) in binary.items():
            result.scores.append(M.score_binary_market(probs, actuals, key, market))

        for (key, market), (preds, actuals) in numeric.items():
            result.scores.append(M.score_numeric_market(preds, actuals, key, market))

        # Ensemble weights derived from these measured results.
        component_arrays = {
            k: np.array(v) for k, v in outcome_probs.items() if k != "ensemble"
        }
        if len(component_arrays) >= 2:
            weights, diagnostics = fit_weights(component_arrays, truth_array)
            result.ensemble_weights = weights
            # These weights are fitted on the whole scored sample, so their
            # in-sample log loss is optimistic by construction. Say so, and
            # measure the real benefit on a held-out slice below.
            diagnostics["in_sample"] = True
            result.ensemble_diagnostics = diagnostics
            result.ensemble_validation = self._validate_weights(
                component_arrays, truth_array
            )

        if persist:
            result.run_id = self._persist(config, result, stored_rows)
        return result

    @staticmethod
    def _validate_weights(component_arrays: dict, truth: np.ndarray,
                          split: float = 0.6) -> dict:
        """Fit ensemble weights on an earlier slice, score them on a later one.

        Matches are already in chronological order, so this is a genuine
        forward test: it answers whether weighting the models by measured
        performance would have helped on matches the weighting never saw.
        """
        n = len(truth)
        cut = int(n * split)
        if cut < 60 or n - cut < 60:
            return {"note": f"too few matches ({n}) to validate weights out of sample"}

        train = {k: v[:cut] for k, v in component_arrays.items()}
        test = {k: v[cut:] for k, v in component_arrays.items()}
        truth_test = truth[cut:]

        weights, _ = fit_weights(train, truth[:cut])
        keys = sorted(component_arrays)
        stacked = np.stack([test[k] for k in keys])

        def mixed_log_loss(weight_map: dict) -> float:
            vector = np.array([weight_map.get(k, 0.0) for k in keys], dtype=float)
            total = vector.sum()
            if total <= 0:
                return float("nan")
            vector = vector / total
            probs = np.tensordot(vector, stacked, axes=(0, 0))
            return M.log_loss(probs, truth_test)

        equal = {k: 1.0 / len(keys) for k in keys}
        singles = {
            k: M.log_loss(test[k], truth_test) for k in keys
        }
        best_single = min(singles, key=singles.get)
        return {
            "fitted_on": cut,
            "scored_on": int(n - cut),
            "weights_from_training_slice": weights,
            "log_loss_weighted_out_of_sample": round(mixed_log_loss(weights), 5),
            "log_loss_equal_out_of_sample": round(mixed_log_loss(equal), 5),
            "log_loss_per_model_out_of_sample": {k: round(v, 5) for k, v in singles.items()},
            "best_single_model": best_single,
            "weighting_beat_equal": bool(mixed_log_loss(weights) < mixed_log_loss(equal)),
            "weighting_beat_best_single": bool(
                mixed_log_loss(weights) < singles[best_single]
            ),
        }

    def _persist(self, config: BacktestConfig, result: BacktestResult,
                 rows: list[BacktestPrediction]) -> int:
        run = BacktestRun(
            sport=self.sport,
            label=config.label(),
            spec={
                "competition": config.competition_key,
                "model_version": MODEL_VERSION,
                "refit_days": config.refit_days,
                "seasons_back": config.seasons_back,
                "goal_lines": list(config.goal_lines),
                "ensemble_weights": result.ensemble_weights,
                "ensemble_validation": result.ensemble_validation,
            },
            from_date=config.from_date,
            to_date=config.to_date,
            n_matches=result.n_matches,
            notes=(
                f"baselines: {result.baselines}; "
                f"ensemble weights: {result.ensemble_weights}"
            )[:2000],
        )
        self.session.add(run)
        self.session.flush()

        for score in result.scores:
            for metric, value in score.metrics.items():
                if value != value:      # NaN
                    continue
                self.session.add(BacktestMetric(
                    run_id=run.id, model_key=score.model_key, market=score.market,
                    metric=metric, value=float(value), n=score.n,
                    detail={"calibration": [b.as_dict() for b in score.calibration]}
                    if metric == "calibration_error" else {},
                ))
        for row in rows:
            row.run_id = run.id
            self.session.add(row)
        self.session.flush()
        return run.id


def latest_ensemble_weights(session: Session, competition_key: str) -> Optional[dict[str, float]]:
    """Weights from the most recent backtest of this competition *and* model version.

    A backtest of an earlier model version measured different models, so its
    weights are not reused; the caller then falls back to the validated
    defaults shipped with the current version.
    """
    runs = session.execute(
        select(BacktestRun)
        .where(BacktestRun.spec["competition"].as_string() == competition_key)
        .order_by(BacktestRun.created_at.desc())
    ).scalars()
    for run in runs:
        spec = run.spec or {}
        if spec.get("model_version") != MODEL_VERSION:
            continue
        weights = spec.get("ensemble_weights") or {}
        if len(weights) >= 2:
            return {k: float(v) for k, v in weights.items()}
    return None
