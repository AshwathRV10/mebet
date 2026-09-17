"""Acquisition pipeline: ask sources, ingest what came back, report the rest.

The pipeline never papers over a failed source. If football-data.co.uk is
unreachable and the datahub mirror answers, the result says exactly that, and
the data-quality layer downgrades accordingly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from sqlalchemy.orm import Session

from .logging_setup import get_logger
from .normalize.ingest import Ingestor, IngestionReport
from .sources.base import Capability, SourceAdapter, SourceResponse
from .sources.registry import build_sources, sources_for

log = get_logger("pipeline")


def _safe_call(adapter: SourceAdapter, method: str, **kwargs) -> SourceResponse:
    """Call an adapter without letting a bug in it abort the whole run.

    Third-party feeds change shape without notice. A parser that trips over an
    unexpected payload should cost us that one source, not the ingestion.
    """
    try:
        return getattr(adapter, method)(**kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberate isolation boundary
        log.exception("adapter %s.%s raised", adapter.key, method)
        return SourceResponse.failure(adapter.key, f"adapter error: {type(exc).__name__}: {exc}")


def season_label(date: dt.date) -> str:
    """European football season containing ``date`` (August-May)."""
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def seasons_back(from_season: str, count: int) -> list[str]:
    start = int(from_season.split("-")[0])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start - count + 1, start + 1)]


@dataclass
class SourceOutcome:
    source_key: str
    capability: str
    ok: bool
    records: int = 0
    written: int = 0
    updated: int = 0
    error: str = ""
    urls: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    retrieved_at: Optional[dt.datetime] = None


@dataclass
class PipelineReport:
    outcomes: list[SourceOutcome] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def add(self, outcome: SourceOutcome) -> None:
        self.outcomes.append(outcome)

    @property
    def succeeded(self) -> list[SourceOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[SourceOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def sources_used(self) -> list[str]:
        return sorted({o.source_key for o in self.succeeded})

    def as_dict(self) -> dict:
        return {
            "sources_used": self.sources_used(),
            "succeeded": [
                {
                    "source": o.source_key, "capability": o.capability, "records": o.records,
                    "written": o.written, "updated": o.updated, "urls": o.urls,
                    "detail": o.detail,
                    "retrieved_at": o.retrieved_at.isoformat() if o.retrieved_at else None,
                }
                for o in self.succeeded
            ],
            "failed": [
                {"source": o.source_key, "capability": o.capability, "error": o.error}
                for o in self.failed
            ],
            "conflicts": self.conflicts,
        }


class AcquisitionPipeline:
    def __init__(self, session: Session, sport: str = "football",
                 adapters: Optional[Sequence[SourceAdapter]] = None,
                 commit_each: bool = True) -> None:
        self.session = session
        self.sport = sport
        self.ingestor = Ingestor(session, sport=sport)
        self._adapters = list(adapters) if adapters is not None else None
        # A full historical load runs for minutes and writes hundreds of
        # thousands of rows. Committing after each unit of work makes the
        # results durable as they arrive, keeps the write-ahead log from
        # growing without bound, and lets an interrupted load resume.
        self.commit_each = commit_each

    def _checkpoint(self) -> None:
        if self.commit_each:
            self.session.commit()

    def _sources(self, capability: Capability) -> list[SourceAdapter]:
        if self._adapters is not None:
            return sorted(
                [a for a in self._adapters if a.supports(capability, self.sport)],
                key=lambda a: a.reliability, reverse=True,
            )
        return sources_for(capability, self.sport)

    # -- historical results & stats --------------------------------------
    def load_history(self, competition_key: str, seasons: Iterable[str],
                     report: Optional[PipelineReport] = None) -> PipelineReport:
        report = report or PipelineReport()
        # Stats-bearing sources first, then results-only ones, deduplicated
        # by key (each call constructs fresh adapter objects, so identity
        # comparison would not deduplicate).
        adapters: list[SourceAdapter] = []
        seen: set[str] = set()
        for adapter in self._sources(Capability.MATCH_STATS) + self._sources(Capability.MATCH_RESULTS):
            if adapter.key in seen:
                continue
            seen.add(adapter.key)
            adapters.append(adapter)
        if not adapters:
            report.add(SourceOutcome("(none)", "match_results", False,
                                     error="no source registered for match results"))
            return report

        for season in seasons:
            got_stats = False
            for adapter in adapters:
                # Once a stats-bearing source has covered a season, a
                # results-only source adds nothing but risk of conflict.
                if got_stats and not adapter.supports(Capability.MATCH_STATS, self.sport):
                    continue
                started = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                resp = _safe_call(adapter, "fetch_matches", competition_key=competition_key,
                                  season=season, sport=self.sport)
                if not resp.ok:
                    report.add(SourceOutcome(adapter.key, f"match_results:{season}", False,
                                             error=resp.error, urls=resp.urls))
                    continue
                ing = self.ingestor.ingest_matches(resp.records, source=adapter,
                                                   scope=f"{competition_key}/{season}")
                self.ingestor.log_run(ing, started)
                self._checkpoint()
                report.conflicts.extend(ing.conflicts)
                report.add(SourceOutcome(
                    adapter.key, f"match_results:{season}", True, records=len(resp.records),
                    written=ing.written, updated=ing.updated, urls=resp.urls,
                    detail=resp.detail, retrieved_at=resp.retrieved_at,
                ))
                if adapter.supports(Capability.MATCH_STATS, self.sport) and resp.records:
                    got_stats = True
        return report

    # -- fixtures ---------------------------------------------------------
    def load_fixtures(self, competition_key: str,
                      report: Optional[PipelineReport] = None) -> PipelineReport:
        report = report or PipelineReport()
        for adapter in self._sources(Capability.FIXTURES):
            started = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            resp = _safe_call(adapter, "fetch_fixtures", competition_key=competition_key,
                              sport=self.sport)
            if not resp.ok:
                report.add(SourceOutcome(adapter.key, "fixtures", False, error=resp.error,
                                         urls=resp.urls))
                continue
            ing = self.ingestor.ingest_matches(resp.records, source=adapter,
                                               scope=f"fixtures/{competition_key}")
            self.ingestor.log_run(ing, started)
            self._checkpoint()
            report.conflicts.extend(ing.conflicts)
            report.add(SourceOutcome(adapter.key, "fixtures", True, records=len(resp.records),
                                     written=ing.written, updated=ing.updated, urls=resp.urls,
                                     detail=resp.detail, retrieved_at=resp.retrieved_at))
        return report

    # -- players ----------------------------------------------------------
    def load_player_stats(self, competition_key: str, seasons: Iterable[str],
                          report: Optional[PipelineReport] = None) -> PipelineReport:
        report = report or PipelineReport()
        adapters = self._sources(Capability.PLAYER_STATS)
        if not adapters:
            report.add(SourceOutcome("(none)", "player_stats", False,
                                     error="no source registered for player statistics"))
            return report
        for season in seasons:
            for adapter in adapters:
                started = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                resp = _safe_call(adapter, "fetch_player_stats", competition_key=competition_key,
                                  season=season, sport=self.sport)
                if not resp.ok:
                    report.add(SourceOutcome(adapter.key, f"player_stats:{season}", False,
                                             error=resp.error, urls=resp.urls))
                    continue
                ing = self.ingestor.ingest_player_stats(resp.records, source=adapter,
                                                        scope=f"{competition_key}/{season}")
                self.ingestor.log_run(ing, started)
                self._checkpoint()
                report.add(SourceOutcome(
                    adapter.key, f"player_stats:{season}", True, records=len(resp.records),
                    written=ing.written, updated=ing.updated, urls=resp.urls,
                    detail=resp.detail, retrieved_at=resp.retrieved_at,
                ))
                break  # one player-stats source per season is sufficient
        return report

    # -- availability ------------------------------------------------------
    def load_availability(self, competition_key: str,
                          report: Optional[PipelineReport] = None) -> PipelineReport:
        report = report or PipelineReport()
        adapters = self._sources(Capability.AVAILABILITY)
        if not adapters:
            report.add(SourceOutcome("(none)", "availability", False,
                                     error="no source registered for injury/availability data"))
            return report
        for adapter in adapters:
            started = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            resp = _safe_call(adapter, "fetch_availability", competition_key=competition_key,
                              sport=self.sport)
            if not resp.ok:
                report.add(SourceOutcome(adapter.key, "availability", False, error=resp.error,
                                         urls=resp.urls))
                continue
            ing = self.ingestor.ingest_availability(resp.records, source=adapter,
                                                    scope=competition_key)
            self.ingestor.log_run(ing, started)
            self._checkpoint()
            report.add(SourceOutcome(adapter.key, "availability", True, records=len(resp.records),
                                     written=ing.written, updated=ing.updated, urls=resp.urls,
                                     detail=resp.detail, retrieved_at=resp.retrieved_at))
        return report

    # -- weather -----------------------------------------------------------
    def load_weather(self, match, latitude: float, longitude: float,
                     report: Optional[PipelineReport] = None) -> PipelineReport:
        report = report or PipelineReport()
        when = match.kickoff_utc or dt.datetime.combine(match.kickoff_date, dt.time(15, 0))
        for adapter in self._sources(Capability.WEATHER):
            resp = _safe_call(adapter, "fetch_weather", latitude=latitude, longitude=longitude,
                              when=when)
            if not resp.ok:
                report.add(SourceOutcome(adapter.key, "weather", False, error=resp.error))
                continue
            ing = self.ingestor.ingest_conditions(resp.records[0], match=match, source=adapter)
            report.add(SourceOutcome(adapter.key, "weather", True, records=1,
                                     written=ing.written, updated=ing.updated, urls=resp.urls))
            break
        return report
