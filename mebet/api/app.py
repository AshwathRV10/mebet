"""HTTP API and static frontend host.

Runs locally by default (127.0.0.1) but binds from configuration, so the same
build serves a LAN or a private server without code changes. API keys live in
the environment and are never sent to the browser: the sources endpoint
reports names and availability only.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..backtest import BacktestConfig, BacktestRunner
from ..config import get_settings
from ..db.models import BacktestMetric, BacktestRun
from ..db.session import init_db, session_scope
from ..engine.service import AnalysisRequest, AnalysisService
from ..logging_setup import get_logger, setup_logging

log = get_logger("api")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class AnalyzeBody(BaseModel):
    sport: str = "football"
    competition: str = Field(..., description="Competition key, e.g. ENG.1")
    home_team: str
    away_team: str
    kickoff: Optional[dt.datetime] = None
    date: Optional[dt.date] = None
    refresh: bool = True
    history_seasons: int = 8
    include_players: bool = True

    def to_request(self) -> AnalysisRequest:
        if self.kickoff is None and self.date is None:
            raise HTTPException(400, "either 'kickoff' or 'date' is required")
        return AnalysisRequest(
            competition_key=self.competition,
            home_team=self.home_team,
            away_team=self.away_team,
            kickoff=self.kickoff,
            kickoff_date=self.date,
            sport=self.sport,
            refresh=self.refresh,
            history_seasons=self.history_seasons,
            include_players=self.include_players,
        )


class BacktestBody(BaseModel):
    competition: str
    from_date: dt.date
    to_date: dt.date
    refit_days: int = 30
    seasons_back: int = 5


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()
    init_db()

    app = FastAPI(
        title="mebet",
        version=__version__,
        description="Local-first, evidence-based sports match prediction.",
    )
    # Permissive by default because the service is expected to run on the
    # user's own machine; restrict via a reverse proxy when deployed further.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "database": settings.resolved_database_url().split("://")[0],
            "offline_mode": settings.offline,
            "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

    @app.get("/api/sources")
    def sources():
        with session_scope() as s:
            return {"sources": AnalysisService(s).source_status()}

    @app.get("/api/competitions")
    def competitions():
        with session_scope() as s:
            service = AnalysisService(s)
            return {
                "stored": service.competitions(),
                "available": service.known_competition_keys(),
            }

    @app.get("/api/teams")
    def teams(competition: Optional[str] = None):
        with session_scope() as s:
            return {"teams": AnalysisService(s).teams(competition)}

    @app.get("/api/fixtures")
    def fixtures(competition: str, limit: int = 40):
        with session_scope() as s:
            return {"fixtures": AnalysisService(s).upcoming_fixtures(competition, limit)}

    @app.post("/api/analyze")
    def analyze(body: AnalyzeBody):
        request = body.to_request()
        with session_scope() as s:
            response = AnalysisService(s, sport=body.sport).analyze(request)
            if not response.ok and not response.prediction:
                raise HTTPException(422, response.message)
            return response.as_dict()

    @app.post("/api/matches/{match_id}/refresh")
    def refresh(match_id: int, include_players: bool = True):
        with session_scope() as s:
            response = AnalysisService(s).refresh_and_recalculate(
                match_id, include_players=include_players
            )
            if not response.ok:
                raise HTTPException(422, response.message)
            return response.as_dict()

    @app.get("/api/matches/{match_id}/predictions")
    def predictions(match_id: int):
        with session_scope() as s:
            return {"versions": AnalysisService(s).prediction_history(match_id)}

    @app.get("/api/backtests")
    def list_backtests(limit: int = 20):
        with session_scope() as s:
            runs = s.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit).all()
            out = []
            for run in runs:
                metrics = s.query(BacktestMetric).filter(BacktestMetric.run_id == run.id).all()
                grouped: dict = {}
                for m in metrics:
                    grouped.setdefault(m.market, {}).setdefault(m.model_key, {})[m.metric] = round(
                        m.value, 5
                    )
                out.append({
                    "id": run.id,
                    "label": run.label,
                    "created_at": run.created_at.isoformat(),
                    "from": run.from_date.isoformat() if run.from_date else None,
                    "to": run.to_date.isoformat() if run.to_date else None,
                    "n_matches": run.n_matches,
                    "spec": run.spec,
                    "metrics": grouped,
                })
            return {"runs": out}

    @app.post("/api/backtests")
    def run_backtest(body: BacktestBody):
        with session_scope() as s:
            config = BacktestConfig(
                competition_key=body.competition, from_date=body.from_date,
                to_date=body.to_date, refit_days=body.refit_days,
                seasons_back=body.seasons_back,
            )
            result = BacktestRunner(s).run(config)
            return result.as_dict()

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(WEB_DIR / "index.html"))

    return app


app = create_app()
