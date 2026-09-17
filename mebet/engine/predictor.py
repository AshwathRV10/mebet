"""The prediction engine.

Orchestrates the full sequence the user asked for:

    collect -> validate -> analyse -> run models -> generate predictions

and returns one object carrying the predictions, the explanations, the data
quality assessment and the provenance of everything used. Predictions are
versioned: recalculating creates a new version and records what changed, so
the history of what was believed and when is preserved.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import __version__
from ..config import get_settings
from ..db.models import (
    Match,
    PlayerMatchStats,
    Prediction,
    PredictionFactor,
    PredictionTarget,
    Team,
)
from ..db.index import MatchIndex
from ..db.repository import AsOfRepository
from ..features.football import build_match_features
from ..logging_setup import get_logger
from ..models.base import PredictionRequest, TrainingContext
from ..models.players import PlayerModel
from ..models.registry import build_count_models, build_ensemble, build_outcome_models
from ..quality import (
    DataQualityReport,
    QualityTier,
    check_conflicts,
    check_freshness,
    check_lineups,
    check_sample_sizes,
    check_statistic_coverage,
    check_suspicious_values,
    finalise,
)
from . import targets as T
from .explain import ExplanationEngine

log = get_logger("engine.predictor")

#: Fitted models, reused between requests. Fitting is the expensive step and it
#: depends only on (competition, cutoff date, the data in hand) - so the cache
#: key carries a fingerprint of the data and any ingestion invalidates it.
_MODEL_CACHE: dict[tuple, object] = {}
_MODEL_CACHE_LIMIT = 24


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


@dataclass
class PredictionResult:
    match_id: int
    home_team: str
    away_team: str
    competition: str
    kickoff: Optional[dt.datetime]
    generated_at: dt.datetime
    as_of: dt.datetime
    version: int = 1
    outcome: dict[str, float] = field(default_factory=dict)
    most_likely_score: Optional[str] = None
    targets: list[T.Target] = field(default_factory=list)
    factors: list = field(default_factory=list)
    quality: Optional[DataQualityReport] = None
    model_weights: dict = field(default_factory=dict)
    model_diagnostics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    features_snapshot: dict = field(default_factory=dict)
    player_projections: list = field(default_factory=list)
    change_summary: str = ""
    warnings: list[str] = field(default_factory=list)

    def targets_by_market(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for t in self.targets:
            out.setdefault(t.market, []).append(t.as_dict())
        return out

    def as_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "competition": self.competition,
            "kickoff": self.kickoff.isoformat() if self.kickoff else None,
            "generated_at": self.generated_at.isoformat(),
            "as_of": self.as_of.isoformat(),
            "version": self.version,
            "outcome": {k: round(v, 4) for k, v in self.outcome.items()},
            "most_likely_score": self.most_likely_score,
            "markets": self.targets_by_market(),
            "explanation": [f.as_dict() for f in self.factors],
            "data_quality": self.quality.as_dict() if self.quality else None,
            "model_weights": self.model_weights,
            "model_diagnostics": self.model_diagnostics,
            "provenance": self.provenance,
            "features": self.features_snapshot,
            "players": [p.as_dict() for p in self.player_projections],
            "change_summary": self.change_summary,
            "warnings": self.warnings,
        }


class PredictionEngine:
    def __init__(self, session: Session, sport: str = "football",
                 index: Optional[MatchIndex] = None) -> None:
        self.session = session
        self.sport = sport
        self.settings = get_settings()
        # Built lazily and reused across calls: analysing several matches, or
        # refitting at many cutoffs, should load the history once.
        self._index = index

    def index(self) -> MatchIndex:
        if self._index is None:
            self._index = MatchIndex(self.session, sport=self.sport)
        return self._index

    # -- main entry point --------------------------------------------------
    def predict_match(
        self,
        match: Match,
        *,
        as_of: Optional[dt.datetime] = None,
        ensemble_weights: Optional[dict[str, float]] = None,
        provenance: Optional[dict] = None,
        include_players: bool = True,
        persist: bool = True,
    ) -> PredictionResult:
        as_of = as_of or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        if as_of.tzinfo is not None:
            as_of = as_of.astimezone(dt.timezone.utc).replace(tzinfo=None)

        repo = AsOfRepository(self.session, as_of, sport=self.sport, index=self.index())
        home = self.session.get(Team, match.home_team_id)
        away = self.session.get(Team, match.away_team_id)
        competition = repo.competition(match.competition_id)

        result = PredictionResult(
            match_id=match.id,
            home_team=home.canonical_name if home else "?",
            away_team=away.canonical_name if away else "?",
            competition=competition.name if competition else "",
            kickoff=match.kickoff_utc,
            generated_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            as_of=as_of,
            provenance=provenance or {},
        )

        # ---- analyse -----------------------------------------------------
        features = build_match_features(repo, match.home_team_id, match.away_team_id,
                                        match.competition_id)
        result.features_snapshot = features.as_dict()

        # ---- validate ----------------------------------------------------
        quality = self._assess_quality(repo, match, features, provenance or {})
        result.quality = quality
        confidence = quality.confidence_multiplier

        if quality.tier == QualityTier.INSUFFICIENT:
            result.warnings.append(
                "Insufficient reliable data to produce a prediction for this match."
            )
            if persist:
                self._persist(result, match)
            return result

        # ---- fit models ---------------------------------------------------
        ctx = TrainingContext(
            repo=repo,
            competition_ids=[match.competition_id],
            as_of=as_of,
            half_life_days=self.settings.time_decay_half_life_days,
            index=self.index(),
        )
        ensemble = self._fitted_ensemble(ctx, match.competition_id, as_of, ensemble_weights)
        result.model_weights = ensemble.weights or {}

        request = PredictionRequest(
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            competition_id=match.competition_id,
            kickoff=match.kickoff_utc or dt.datetime.combine(match.kickoff_date, dt.time(15, 0)),
            neutral_venue=match.neutral_venue,
            features=features,
            adjustments={"referee": match.referee or ""},
        )

        main = ensemble.predict(request)
        component_diagnostics = {
            m.key: m.predict(request).diagnostics for m in ensemble.models if m.fitted
        }
        result.model_diagnostics = {
            "ensemble": main.diagnostics,
            "components": component_diagnostics,
            "fit": {m.key: m.fit_diagnostics for m in ensemble.models},
        }

        if not main.sufficient_data or not main.outcome_probs:
            result.warnings.append(
                f"Insufficient reliable data to produce a match-result prediction: {main.note}"
            )
        else:
            main.validate()
            result.outcome = main.outcome_probs
            result.targets.extend(T.outcome_targets(
                main.outcome_probs, ensemble.key, confidence,
                result.home_team, result.away_team,
            ))
            if main.score_matrix is not None:
                result.targets.extend(
                    T.scoreline_targets(main.score_matrix, ensemble.key, confidence)
                )
                result.targets.extend(T.goals_targets(
                    main.score_matrix, ensemble.key, confidence,
                    result.home_team, result.away_team,
                ))
                result.targets.extend(T.first_half_targets(
                    main.score_matrix, features.home.first_half_share,
                    features.away.first_half_share, ensemble.key, confidence,
                ))
                top = max(
                    (
                        (main.score_matrix[h, a], h, a)
                        for h in range(main.score_matrix.shape[0])
                        for a in range(main.score_matrix.shape[1])
                    ),
                    default=None,
                )
                if top:
                    result.most_likely_score = f"{top[1]}-{top[2]}"

        # ---- count markets --------------------------------------------------
        for model in self._fitted_count_models(ctx, match.competition_id, as_of):
            prediction = model.predict(request)
            lines = T.CORNER_LINES if model.key == "corners" else T.CARD_LINES
            result.targets.extend(T.count_targets(
                prediction, model.key, lines, confidence,
                result.home_team, result.away_team,
            ))
            result.model_diagnostics.setdefault("counts", {})[model.key] = (
                prediction.diagnostics if prediction.sufficient_data
                else {"unavailable": prediction.note}
            )

        # ---- player markets --------------------------------------------------
        availability = self._availability_map(repo, match)
        if include_players:
            player_model = PlayerModel()
            projections = []
            for team_id, expected in (
                (match.home_team_id, main.expected_home_goals),
                (match.away_team_id, main.expected_away_goals),
            ):
                projections.extend(player_model.project_team(
                    repo, team_id, team_expected_goals=expected,
                    availability=availability.get(team_id, {}),
                ))
            result.player_projections = projections
            result.targets.extend(T.player_targets(projections, confidence))

        # ---- explain ----------------------------------------------------------
        explainer = ExplanationEngine(features)
        result.factors = explainer.build(
            availability=self._availability_for_explanation(availability, match),
            model_diagnostics={
                **component_diagnostics.get("dixon_coles", {}),
                **component_diagnostics.get("elo", {}),
            },
        )

        if persist:
            self._persist(result, match)
        return result

    def _fitted_count_models(self, ctx: TrainingContext, competition_id: int,
                             as_of: dt.datetime):
        key = ("counts", competition_id, as_of.date().isoformat(),
               self._data_fingerprint(competition_id))
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        models = build_count_models()
        for model in models:
            model.fit(ctx)
        if len(_MODEL_CACHE) >= _MODEL_CACHE_LIMIT:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = models
        return models

    def _data_fingerprint(self, competition_id: int) -> tuple:
        """Cheap signature of the stored data for this competition."""
        index = self.index()
        matches = index.by_competition.get(competition_id, ())
        latest = matches[0].kickoff_date.isoformat() if matches else "none"
        return (len(matches), latest)

    def _fitted_ensemble(self, ctx: TrainingContext, competition_id: int,
                         as_of: dt.datetime, weights: Optional[dict]):
        key = (
            competition_id, as_of.date().isoformat(), self._data_fingerprint(competition_id),
            tuple(sorted((weights or {}).items())),
        )
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            log.debug("reusing fitted models for %s", key[:2])
            return cached
        ensemble = build_ensemble(weights=weights)
        ensemble.fit(ctx)
        if len(_MODEL_CACHE) >= _MODEL_CACHE_LIMIT:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = ensemble
        return ensemble

    # -- quality ------------------------------------------------------------
    def _assess_quality(self, repo: AsOfRepository, match: Match, features,
                        provenance: dict) -> DataQualityReport:
        report = DataQualityReport()
        report.sources_used = provenance.get("sources_used", [])
        report.sources_failed = provenance.get("failed", [])
        if not report.sources_used:
            report.sources_used = self._sources_behind(match)

        home_rows = repo.team_history(match.home_team_id, limit=40)
        away_rows = repo.team_history(match.away_team_id, limit=40)
        check_sample_sizes(report, len(home_rows), len(away_rows),
                           features.h2h.get("matches", 0))
        coverage_home = check_statistic_coverage(report, home_rows, label="home")
        coverage_away = check_statistic_coverage(report, away_rows, label="away")
        report.stats_coverage = {"home": coverage_home, "away": coverage_away}
        check_suspicious_values(report, home_rows + away_rows)

        kickoff = match.kickoff_utc or dt.datetime.combine(match.kickoff_date, dt.time(15, 0))
        check_freshness(report, home_rows + away_rows, repo.as_of, kickoff=kickoff)
        check_lineups(report, repo.lineups(match.id), kickoff, repo.as_of)
        check_conflicts(report, provenance.get("conflicts", []))

        availability = repo.availability(match.home_team_id) + repo.availability(match.away_team_id)
        report.availability_data = bool(availability)
        if not availability:
            report.add("no_team_news", "warning",
                       "no injury or suspension information was retrieved for either squad")

        player_rows = repo.team_player_history(
            match.home_team_id, since=dt.date(repo.as_of.year - 1, 7, 1)
        )
        report.player_data = bool(player_rows)
        if not player_rows:
            report.add("no_player_data", "info",
                       "no player-level statistics are available for this competition")

        report.weather = repo.conditions(match.id) is not None
        return finalise(report)

    def _sources_behind(self, match: Match) -> list[str]:
        """Which providers supplied the evidence this prediction rests on.

        An upcoming fixture has no statistics of its own, so reporting only
        its own source would understate what was used. The sources that matter
        are the ones behind the two teams' match histories.
        """
        from ..db.models import DataSource, TeamMatchStats

        team_ids = [match.home_team_id, match.away_team_id]
        rows = self.session.execute(
            select(DataSource.key)
            .join(TeamMatchStats, TeamMatchStats.source_id == DataSource.id)
            .where(TeamMatchStats.team_id.in_(team_ids))
            .distinct()
        ).scalars()
        keys = set(rows)

        player_sources = self.session.execute(
            select(DataSource.key)
            .join(PlayerMatchStats, PlayerMatchStats.source_id == DataSource.id)
            .where(PlayerMatchStats.team_id.in_(team_ids))
            .distinct()
        ).scalars()
        keys.update(player_sources)

        if match.source_id:
            src = self.session.get(DataSource, match.source_id)
            if src:
                keys.add(src.key)
        return sorted(keys)

    # -- availability --------------------------------------------------------
    def _availability_map(self, repo: AsOfRepository, match: Match) -> dict[int, dict[int, dict]]:
        out: dict[int, dict[int, dict]] = {}
        for team_id in (match.home_team_id, match.away_team_id):
            entries = {}
            for row in repo.availability(team_id):
                entries[row.player_id] = {
                    "status": row.status,
                    "reason": row.reason,
                    "chance_of_playing": row.chance_of_playing,
                    "observed_at": row.observed_at.isoformat() if row.observed_at else None,
                }
            out[team_id] = entries
        return out

    def _availability_for_explanation(self, availability: dict, match: Match) -> dict:
        from ..db.models import Player

        out = {"home": [], "away": []}
        for side, team_id in (("home", match.home_team_id), ("away", match.away_team_id)):
            for player_id, info in (availability.get(team_id) or {}).items():
                if info.get("status") not in {"injured", "suspended", "unavailable", "doubtful"}:
                    continue
                player = self.session.get(Player, player_id)
                out[side].append({
                    "player": player.full_name if player else f"player:{player_id}",
                    "status": info.get("status"),
                    "reason": info.get("reason", ""),
                })
        return out

    # -- persistence ---------------------------------------------------------
    def _persist(self, result: PredictionResult, match: Match) -> Prediction:
        previous = self.session.execute(
            select(Prediction)
            .where(Prediction.match_id == match.id, Prediction.superseded.is_(False))
            .order_by(Prediction.version.desc())
        ).scalars().first()

        version = 1
        if previous is not None:
            previous.superseded = True
            version = previous.version + 1
            result.change_summary = self._describe_change(previous, result)
        result.version = version

        row = Prediction(
            match_id=match.id,
            version=version,
            created_at=result.generated_at,
            as_of=result.as_of,
            engine_version=__version__,
            primary_model="ensemble",
            data_quality_tier=result.quality.tier.value if result.quality else "UNKNOWN",
            data_quality=result.quality.as_dict() if result.quality else {},
            model_weights=result.model_weights,
            change_summary=result.change_summary,
        )
        self.session.add(row)
        self.session.flush()

        for t in result.targets:
            self.session.add(PredictionTarget(
                prediction_id=row.id, market=t.market, selection=t.selection,
                probability=t.probability, expected_value=t.expected_value,
                interval_low=t.interval_low, interval_high=t.interval_high,
                model_key=t.model_key, confidence=t.confidence,
                sufficient_data=t.sufficient_data, note=t.note,
                subject_type=t.subject_type, subject_id=t.subject_id,
                subject_name=t.subject_name,
            ))
        for f in result.factors:
            self.session.add(PredictionFactor(
                prediction_id=row.id, market=f.market, favours=f.favours,
                category=f.category, statement=f.statement, impact=f.impact,
                evidence=f.evidence,
            ))
        self.session.flush()
        return row

    def _describe_change(self, previous: Prediction, result: PredictionResult) -> str:
        """Say what moved between versions, so a refresh is informative."""
        old = {
            t.selection: t.probability
            for t in previous.targets if t.market == "1x2" and t.probability is not None
        }
        if not old or not result.outcome:
            return "recalculated with the latest available data"
        deltas = []
        for key in ("home", "draw", "away"):
            if key in old and key in result.outcome:
                change = result.outcome[key] - old[key]
                if abs(change) >= 0.01:
                    deltas.append(f"{key} {change:+.1%}")
        quality_changed = ""
        if result.quality and previous.data_quality_tier != result.quality.tier.value:
            quality_changed = (
                f"; data quality {previous.data_quality_tier} -> {result.quality.tier.value}"
            )
        if not deltas:
            return f"no material change in match-result probabilities{quality_changed}"
        return "changes: " + ", ".join(deltas) + quality_changed
