"""Application service: the "Analyze Match" and "Refresh & Recalculate" flows.

Shared by the HTTP API and the command line so both behave identically.

The sequence for an analysis is the one the brief describes:

    collect -> validate -> analyse -> run models -> generate predictions

with the detail that collection is *scoped*: only the competition and teams
involved are fetched, because the application exists to answer questions
about specific matches rather than to mirror the world's football data.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Sequence

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.index import MatchIndex
from ..db.models import Competition, Match, Prediction, Season, Team
from ..logging_setup import get_logger
from ..normalize.teams import TeamResolver
from ..pipeline import AcquisitionPipeline, PipelineReport, season_label, seasons_back
from ..sources.registry import status_report
from ..backtest.runner import latest_ensemble_weights
from .predictor import PredictionEngine, PredictionResult, clear_model_cache

log = get_logger("engine.service")

#: How many seasons of history to pull when a competition is first analysed.
DEFAULT_HISTORY_SEASONS = 8


@dataclass
class AnalysisRequest:
    competition_key: str
    home_team: str
    away_team: str
    kickoff: Optional[dt.datetime] = None
    kickoff_date: Optional[dt.date] = None
    sport: str = "football"
    refresh: bool = True
    history_seasons: int = DEFAULT_HISTORY_SEASONS
    include_players: bool = True
    as_of: Optional[dt.datetime] = None

    def resolved_date(self) -> dt.date:
        if self.kickoff is not None:
            return self.kickoff.date()
        if self.kickoff_date is not None:
            return self.kickoff_date
        raise ValueError("a kickoff date or datetime is required")


@dataclass
class AnalysisResponse:
    ok: bool
    prediction: Optional[PredictionResult] = None
    pipeline: Optional[PipelineReport] = None
    message: str = ""
    match_id: Optional[int] = None
    created_fixture: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "message": self.message,
            "match_id": self.match_id,
            "created_fixture": self.created_fixture,
            "warnings": self.warnings,
            "acquisition": self.pipeline.as_dict() if self.pipeline else None,
            "prediction": self.prediction.as_dict() if self.prediction else None,
        }


class AnalysisService:
    def __init__(self, session: Session, sport: str = "football") -> None:
        self.session = session
        self.sport = sport
        self.settings = get_settings()
        self._index: Optional[MatchIndex] = None

    def invalidate_index(self) -> None:
        """Called after ingestion so the next prediction sees the new rows."""
        self._index = None
        clear_model_cache()

    def _engine(self) -> PredictionEngine:
        if self._index is None:
            self._index = MatchIndex(self.session, sport=self.sport)
        return PredictionEngine(self.session, sport=self.sport, index=self._index)

    # -- reference data ----------------------------------------------------
    def competitions(self) -> list[dict]:
        rows = self.session.execute(
            select(Competition).where(Competition.sport == self.sport)
            .order_by(Competition.country, Competition.tier)
        ).scalars()
        out = []
        for c in rows:
            match_count = self.session.execute(
                select(Match.id).where(Match.competition_id == c.id).limit(10000)
            ).scalars().all()
            out.append({
                "key": c.key, "name": c.name, "country": c.country,
                "tier": c.tier, "format": c.format, "matches": len(match_count),
            })
        return out

    def known_competition_keys(self) -> list[str]:
        """Every competition any registered source can serve, plus stored ones."""
        from ..sources.datahub_footballdata import COMPETITIONS as DATAHUB
        from ..sources.footballdata_couk import COMPETITIONS as FDCOUK
        from ..sources.openfootball import COMPETITIONS as OPENFOOTBALL

        keys = set(DATAHUB) | set(FDCOUK) | set(OPENFOOTBALL)
        keys.update(c["key"] for c in self.competitions())
        return sorted(keys)

    def teams(self, competition_key: Optional[str] = None) -> list[dict]:
        if competition_key:
            comp = self._competition(competition_key)
            if comp is None:
                return []
            recent = dt.date.today() - dt.timedelta(days=550)
            rows = self.session.execute(
                select(Match.home_team_id, Match.away_team_id)
                .where(Match.competition_id == comp.id, Match.kickoff_date >= recent)
            ).all()
            ids = {r[0] for r in rows} | {r[1] for r in rows}
            teams = [self.session.get(Team, i) for i in ids]
        else:
            teams = list(self.session.execute(
                select(Team).where(Team.sport == self.sport)
            ).scalars())
        return sorted(
            [{"id": t.id, "name": t.canonical_name, "country": t.country}
             for t in teams if t is not None],
            key=lambda t: t["name"],
        )

    def source_status(self) -> list[dict]:
        out = []
        for status in status_report():
            out.append({
                "key": status.key, "available": status.available, "detail": status.detail,
                "checked_at": status.checked_at.isoformat(),
            })
        return out

    def _competition(self, key: str) -> Optional[Competition]:
        return self.session.execute(
            select(Competition).where(Competition.sport == self.sport, Competition.key == key)
        ).scalars().first()

    # -- acquisition -------------------------------------------------------
    def refresh_competition(self, competition_key: str, *, seasons: int,
                            include_players: bool = True) -> PipelineReport:
        pipeline = AcquisitionPipeline(self.session, sport=self.sport)
        current = season_label(dt.date.today())
        wanted = seasons_back(current, seasons)
        report = pipeline.load_history(competition_key, wanted)
        pipeline.load_fixtures(competition_key, report)
        if include_players:
            # Player statistics are heavy; the current and previous season are
            # what the player models actually use.
            pipeline.load_player_stats(competition_key, wanted[-2:], report)
        pipeline.load_availability(competition_key, report)
        self.invalidate_index()
        return report

    # -- match resolution --------------------------------------------------
    def find_or_create_match(self, request: AnalysisRequest) -> tuple[Optional[Match], bool, str]:
        comp = self._competition(request.competition_key)
        if comp is None:
            return None, False, (
                f"competition {request.competition_key} is not in the database; "
                "run a refresh for it first"
            )
        resolver = TeamResolver(self.session, sport=self.sport)
        home = resolver.resolve(request.home_team, country=comp.country, create=False)
        away = resolver.resolve(request.away_team, country=comp.country, create=False)
        missing = [
            name for name, team in ((request.home_team, home), (request.away_team, away))
            if team is None
        ]
        if missing:
            return None, False, (
                f"no team on record matching {', '.join(repr(m) for m in missing)} in "
                f"{comp.name}. Check the spelling, or refresh the competition so the "
                "team is loaded."
            )
        if home.id == away.id:
            return None, False, "the two teams resolve to the same club"

        target = request.resolved_date()
        window = [target + dt.timedelta(days=d) for d in (-2, -1, 0, 1, 2)]
        match = self.session.execute(
            select(Match).where(
                Match.competition_id == comp.id,
                Match.home_team_id == home.id,
                Match.away_team_id == away.id,
                Match.kickoff_date.in_(window),
            ).order_by(Match.kickoff_date)
        ).scalars().first()
        if match is not None:
            return match, False, ""

        # No fixture on record. Create a placeholder so the match can be
        # analysed - it holds no results, only the identity of the fixture.
        season = self._season_for(comp, target)
        match = Match(
            sport=self.sport, competition_id=comp.id, season_id=season.id if season else None,
            kickoff_date=target,
            kickoff_utc=request.kickoff.replace(tzinfo=None) if request.kickoff else None,
            kickoff_is_exact=request.kickoff is not None,
            home_team_id=home.id, away_team_id=away.id, status="scheduled",
            external_ids={"created_by": "user_request"},
        )
        self.session.add(match)
        self.session.flush()
        return match, True, (
            "this fixture was not found in any source, so it has been recorded as a "
            "user-entered fixture; the prediction still uses only real historical data"
        )

    def _season_for(self, comp: Competition, date: dt.date) -> Optional[Season]:
        label = season_label(date)
        season = self.session.execute(
            select(Season).where(Season.competition_id == comp.id, Season.label == label)
        ).scalars().first()
        if season is None:
            season = Season(competition_id=comp.id, label=label)
            self.session.add(season)
            self.session.flush()
        return season

    # -- main flows ---------------------------------------------------------
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        response = AnalysisResponse(ok=False)
        report: Optional[PipelineReport] = None

        if request.refresh:
            try:
                report = self.refresh_competition(
                    request.competition_key, seasons=request.history_seasons,
                    include_players=request.include_players,
                )
                response.pipeline = report
                if not report.succeeded:
                    response.message = (
                        "No data source could be reached for this competition. "
                        "The prediction below, if any, relies entirely on previously "
                        "stored data."
                    )
                    response.warnings.append(response.message)
            except Exception as exc:  # noqa: BLE001
                log.exception("refresh failed")
                response.warnings.append(f"data refresh failed: {exc}")

        match, created, message = self.find_or_create_match(request)
        if match is None:
            response.message = message
            return response
        if message:
            response.warnings.append(message)
        response.created_fixture = created
        response.match_id = match.id

        weights = latest_ensemble_weights(self.session, request.competition_key)
        provenance = report.as_dict() if report else {}
        engine = self._engine()
        prediction = engine.predict_match(
            match,
            as_of=request.as_of,
            ensemble_weights=weights,
            provenance=provenance,
            include_players=request.include_players,
        )
        response.prediction = prediction
        response.ok = True
        response.message = message or "analysis complete"
        response.warnings.extend(prediction.warnings)
        return response

    def refresh_and_recalculate(self, match_id: int, *,
                                include_players: bool = True) -> AnalysisResponse:
        """The "Refresh Data & Recalculate" action.

        Pulls the newest data for the competition, then produces a *new
        version* of the prediction with a summary of what changed.
        """
        match = self.session.get(Match, match_id)
        if match is None:
            return AnalysisResponse(ok=False, message=f"no match with id {match_id}")
        comp = self.session.get(Competition, match.competition_id)
        request = AnalysisRequest(
            competition_key=comp.key,
            home_team=self.session.get(Team, match.home_team_id).canonical_name,
            away_team=self.session.get(Team, match.away_team_id).canonical_name,
            kickoff=match.kickoff_utc,
            kickoff_date=match.kickoff_date,
            include_players=include_players,
        )
        return self.analyze(request)

    # -- stored predictions --------------------------------------------------
    def prediction_history(self, match_id: int) -> list[dict]:
        rows = self.session.execute(
            select(Prediction).where(Prediction.match_id == match_id)
            .order_by(Prediction.version.desc())
        ).scalars()
        out = []
        for p in rows:
            outcome = {
                t.selection: t.probability for t in p.targets
                if t.market == "1x2" and t.probability is not None
            }
            out.append({
                "version": p.version,
                "created_at": p.created_at.isoformat(),
                "as_of": p.as_of.isoformat(),
                "data_quality": p.data_quality_tier,
                "outcome": {k: round(v, 4) for k, v in outcome.items()},
                "change_summary": p.change_summary,
                "superseded": p.superseded,
                "model_weights": p.model_weights,
            })
        return out

    def upcoming_fixtures(self, competition_key: str, limit: int = 40) -> list[dict]:
        comp = self._competition(competition_key)
        if comp is None:
            return []
        today = dt.date.today()
        rows = self.session.execute(
            select(Match).where(
                Match.competition_id == comp.id,
                Match.kickoff_date >= today,
                Match.status != "played",
            ).order_by(Match.kickoff_date).limit(limit)
        ).scalars()
        out = []
        for m in rows:
            home, away = self.session.get(Team, m.home_team_id), self.session.get(Team, m.away_team_id)
            out.append({
                "match_id": m.id,
                "kickoff": m.kickoff_utc.isoformat() if m.kickoff_utc else None,
                "date": m.kickoff_date.isoformat(),
                "home": home.canonical_name if home else "?",
                "away": away.canonical_name if away else "?",
                "matchday": m.matchday,
            })
        return out
