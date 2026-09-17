"""Ingestion: canonical records -> database rows.

Idempotent by design. Re-running ingestion for a season updates what changed
and leaves everything else alone, so "Refresh Data & Recalculate" is cheap
and safe.

When two sources disagree about the same fact, neither value is silently
overwritten: the more reliable source wins for the match row, both stat rows
are retained under their own ``source_id``, and the disagreement is counted
so the data-quality layer can report it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import (
    Competition,
    DataSource,
    IngestionRun,
    Lineup,
    Match,
    MatchConditions,
    Player,
    PlayerAvailability,
    PlayerMatchStats,
    Season,
    Team,
    TeamMatchStats,
)
from ..logging_setup import get_logger
from ..sources.base import SourceAdapter
from .records import (
    AvailabilityRecord,
    ConditionsRecord,
    LineupRecord,
    MatchRecord,
    PlayerMatchRecord,
)
from .teams import TeamResolver

log = get_logger("normalize.ingest")


@dataclass
class IngestionReport:
    source_key: str
    scope: str = ""
    ok: bool = True
    written: int = 0
    updated: int = 0
    skipped: int = 0
    conflicts: list[str] = field(default_factory=list)
    message: str = ""
    urls: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.source_key}/{self.scope}: {self.written} new, {self.updated} updated, "
            f"{self.skipped} skipped, {len(self.conflicts)} conflicts"
        )


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _naive(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(dt.timezone.utc).replace(tzinfo=None)


class Ingestor:
    def __init__(self, session: Session, sport: str = "football") -> None:
        self.session = session
        self.sport = sport
        self.resolver = TeamResolver(session, sport=sport)

    # -- registry rows ---------------------------------------------------
    def source_row(self, adapter_or_key) -> DataSource:
        if isinstance(adapter_or_key, SourceAdapter):
            key = adapter_or_key.key
            meta = adapter_or_key.describe()
        else:
            key = str(adapter_or_key)
            meta = {}
        row = self.session.execute(
            select(DataSource).where(DataSource.key == key)
        ).scalars().first()
        if row is None:
            row = DataSource(
                key=key,
                name=meta.get("name", key),
                homepage=meta.get("homepage", ""),
                license=meta.get("license", ""),
                reliability=float(meta.get("reliability", 0.5)),
            )
            self.session.add(row)
            self.session.flush()
        elif meta:
            row.name = meta.get("name", row.name)
            row.reliability = float(meta.get("reliability", row.reliability))
        return row

    def competition_row(self, key: str, name: str, country: str = "",
                        fmt: str = "league", tier: Optional[int] = None) -> Competition:
        row = self.session.execute(
            select(Competition).where(Competition.sport == self.sport, Competition.key == key)
        ).scalars().first()
        if row is None:
            row = Competition(sport=self.sport, key=key, name=name, country=country,
                              format=fmt, tier=tier)
            self.session.add(row)
            self.session.flush()
        return row

    def season_row(self, competition: Competition, label: str) -> Season:
        row = self.session.execute(
            select(Season).where(Season.competition_id == competition.id, Season.label == label)
        ).scalars().first()
        if row is None:
            row = Season(competition_id=competition.id, label=label)
            self.session.add(row)
            self.session.flush()
        return row

    # -- matches ---------------------------------------------------------
    def ingest_matches(self, records: Sequence[MatchRecord], *, source,
                       scope: str = "") -> IngestionReport:
        report = IngestionReport(source_key=getattr(source, "key", str(source)), scope=scope)
        if not records:
            report.ok = False
            report.message = "no records supplied"
            return report

        src = self.source_row(source)
        for rec in records:
            comp = self.competition_row(
                rec.competition_key, rec.competition_name,
                country=str(rec.external_ids.get("country", "")),
            )
            season = self.season_row(comp, rec.season_label)
            home = self.resolver.resolve(rec.home_team, country=comp.country)
            away = self.resolver.resolve(rec.away_team, country=comp.country)
            if home is None or away is None or home.id == away.id:
                report.skipped += 1
                continue

            existing = self.session.execute(
                select(Match).where(
                    Match.sport == self.sport,
                    Match.competition_id == comp.id,
                    Match.kickoff_date == rec.kickoff_date,
                    Match.home_team_id == home.id,
                    Match.away_team_id == away.id,
                )
            ).scalars().first()

            if existing is None:
                # Same fixture may be listed on an adjacent date by another
                # source (timezone rounding); match on teams + season window.
                existing = self._find_nearby(comp.id, season.id, home.id, away.id, rec.kickoff_date)

            if existing is None:
                match = Match(
                    sport=self.sport,
                    competition_id=comp.id,
                    season_id=season.id,
                    kickoff_date=rec.kickoff_date,
                    kickoff_utc=_naive(rec.kickoff_utc),
                    kickoff_is_exact=rec.kickoff_is_exact,
                    home_team_id=home.id,
                    away_team_id=away.id,
                    status=rec.status,
                    stage=rec.stage,
                    matchday=rec.matchday,
                    neutral_venue=rec.neutral_venue,
                    ft_home_goals=rec.ft_home_goals,
                    ft_away_goals=rec.ft_away_goals,
                    ht_home_goals=rec.ht_home_goals,
                    ht_away_goals=rec.ht_away_goals,
                    referee=rec.referee,
                    external_ids=dict(rec.external_ids or {}),
                    source_id=src.id,
                    retrieved_at=_naive(rec.retrieved_at) or _utcnow(),
                )
                self.session.add(match)
                self.session.flush()
                report.written += 1
            else:
                match = existing
                if self._update_match(match, rec, src, report):
                    report.updated += 1

            self._write_team_stats(match, rec, home, away, src)

        self.session.flush()
        return report

    def _find_nearby(self, comp_id: int, season_id: int, home_id: int, away_id: int,
                     date: dt.date) -> Optional[Match]:
        window = [date - dt.timedelta(days=1), date, date + dt.timedelta(days=1)]
        return self.session.execute(
            select(Match).where(
                Match.competition_id == comp_id,
                Match.season_id == season_id,
                Match.home_team_id == home_id,
                Match.away_team_id == away_id,
                Match.kickoff_date.in_(window),
            )
        ).scalars().first()

    def _update_match(self, match: Match, rec: MatchRecord, src: DataSource,
                      report: IngestionReport) -> bool:
        changed = False
        incumbent = self.session.get(DataSource, match.source_id) if match.source_id else None
        incumbent_reliability = incumbent.reliability if incumbent else 0.0

        # Detect genuine disagreement on a recorded result.
        if (
            match.ft_home_goals is not None
            and rec.ft_home_goals is not None
            and (match.ft_home_goals, match.ft_away_goals) != (rec.ft_home_goals, rec.ft_away_goals)
        ):
            msg = (
                f"score disagreement for match {match.id} "
                f"({match.kickoff_date}): stored {match.ft_home_goals}-{match.ft_away_goals} "
                f"from {incumbent.key if incumbent else '?'} vs {rec.ft_home_goals}-{rec.ft_away_goals} "
                f"from {src.key}"
            )
            report.conflicts.append(msg)
            log.warning(msg)
            if src.reliability <= incumbent_reliability:
                return False  # keep the more reliable value

        # Fill gaps; overwrite only from a source at least as reliable.
        may_overwrite = src.reliability >= incumbent_reliability
        for attr, value in (
            ("ft_home_goals", rec.ft_home_goals),
            ("ft_away_goals", rec.ft_away_goals),
            ("ht_home_goals", rec.ht_home_goals),
            ("ht_away_goals", rec.ht_away_goals),
            ("referee", rec.referee or None),
            ("matchday", rec.matchday),
        ):
            if value is None or value == "":
                continue
            current = getattr(match, attr)
            if current is None or current == "" or (may_overwrite and current != value):
                setattr(match, attr, value)
                changed = True

        # A precise kickoff time beats a date-only record.
        if rec.kickoff_is_exact and not match.kickoff_is_exact and rec.kickoff_utc:
            match.kickoff_utc = _naive(rec.kickoff_utc)
            match.kickoff_is_exact = True
            match.kickoff_date = rec.kickoff_date
            changed = True

        if rec.status == "played" and match.status != "played":
            match.status = "played"
            changed = True

        if rec.external_ids:
            merged = dict(match.external_ids or {})
            merged.update(rec.external_ids)
            if merged != (match.external_ids or {}):
                match.external_ids = merged
                changed = True

        if changed:
            match.retrieved_at = _naive(rec.retrieved_at) or _utcnow()
            if may_overwrite:
                match.source_id = src.id
        return changed

    def _write_team_stats(self, match: Match, rec: MatchRecord, home: Team, away: Team,
                          src: DataSource) -> None:
        pairs = (
            (home, rec.home_stats, True, rec.ft_home_goals, rec.ft_away_goals, rec.ht_home_goals),
            (away, rec.away_stats, False, rec.ft_away_goals, rec.ft_home_goals, rec.ht_away_goals),
        )
        for team, line, is_home, gf, ga, htg in pairs:
            if line.is_empty() and gf is None:
                continue
            row = self.session.execute(
                select(TeamMatchStats).where(
                    TeamMatchStats.match_id == match.id,
                    TeamMatchStats.team_id == team.id,
                    TeamMatchStats.source_id == src.id,
                )
            ).scalars().first()
            if row is None:
                row = TeamMatchStats(match_id=match.id, team_id=team.id, is_home=is_home,
                                     source_id=src.id)
                self.session.add(row)
            row.goals = gf
            row.goals_conceded = ga
            row.ht_goals = htg
            row.shots = line.shots
            row.shots_on_target = line.shots_on_target
            row.corners = line.corners
            row.fouls = line.fouls
            row.yellow_cards = line.yellow_cards
            row.red_cards = line.red_cards
            row.possession = line.possession
            row.xg = line.xg
            row.xga = line.xga
            row.retrieved_at = _naive(rec.retrieved_at) or _utcnow()

    # -- players ---------------------------------------------------------
    def player_row(self, name: str, team: Optional[Team], position: str = "",
                   external_ids: Optional[dict] = None,
                   src: Optional[DataSource] = None) -> Optional[Player]:
        name = (name or "").strip()
        if not name:
            return None
        stmt = select(Player).where(Player.sport == self.sport, Player.full_name == name)
        if team is not None:
            stmt = stmt.where(Player.team_id == team.id)
        row = self.session.execute(stmt).scalars().first()
        if row is None:
            # A transferred player exists under another club; move rather than
            # duplicate, so their history stays in one place.
            row = self.session.execute(
                select(Player).where(Player.sport == self.sport, Player.full_name == name)
            ).scalars().first()
            if row is not None and team is not None:
                row.team_id = team.id
        if row is None:
            row = Player(
                sport=self.sport, full_name=name,
                team_id=team.id if team else None,
                position=position,
                external_ids=dict(external_ids or {}),
                source_id=src.id if src else None,
                retrieved_at=_utcnow(),
            )
            self.session.add(row)
            self.session.flush()
        else:
            if position and not row.position:
                row.position = position
            if external_ids:
                merged = dict(row.external_ids or {})
                merged.update(external_ids)
                row.external_ids = merged
        return row

    def ingest_player_stats(self, records: Sequence[PlayerMatchRecord], *, source,
                            scope: str = "") -> IngestionReport:
        report = IngestionReport(source_key=getattr(source, "key", str(source)), scope=scope)
        if not records:
            report.ok = False
            report.message = "no records supplied"
            return report
        src = self.source_row(source)

        # Cache match lookups: many player rows share one match.
        match_cache: dict[tuple[int, int, dt.date], Optional[int]] = {}

        for rec in records:
            team = self.resolver.resolve(rec.team_name) if rec.team_name else None
            player = self.player_row(rec.player_name, team, rec.position,
                                     rec.external_ids, src)
            if player is None:
                report.skipped += 1
                continue

            match_id = None
            if team is not None and rec.opponent_name:
                opponent = self.resolver.resolve(rec.opponent_name)
                if opponent is not None:
                    home_id = team.id if rec.was_home else opponent.id
                    away_id = opponent.id if rec.was_home else team.id
                    ck = (home_id, away_id, rec.match_date)
                    if ck not in match_cache:
                        m = self.session.execute(
                            select(Match).where(
                                Match.home_team_id == home_id,
                                Match.away_team_id == away_id,
                                Match.kickoff_date.in_([
                                    rec.match_date - dt.timedelta(days=1),
                                    rec.match_date,
                                    rec.match_date + dt.timedelta(days=1),
                                ]),
                            )
                        ).scalars().first()
                        match_cache[ck] = m.id if m else None
                    match_id = match_cache[ck]

            row = self.session.execute(
                select(PlayerMatchStats).where(
                    PlayerMatchStats.player_id == player.id,
                    PlayerMatchStats.match_date == rec.match_date,
                    PlayerMatchStats.source_id == src.id,
                )
            ).scalars().first()
            if row is None:
                row = PlayerMatchStats(player_id=player.id, match_date=rec.match_date,
                                       source_id=src.id)
                self.session.add(row)
                report.written += 1
            else:
                report.updated += 1

            row.match_id = match_id
            row.team_id = team.id if team else None
            row.minutes = rec.minutes
            row.started = rec.started
            row.goals = rec.goals
            row.assists = rec.assists
            row.shots = rec.shots
            row.shots_on_target = rec.shots_on_target
            row.key_passes = rec.key_passes
            row.xg = rec.xg
            row.xa = rec.xa
            row.xgc = rec.xgc
            row.tackles = rec.tackles
            row.saves = rec.saves
            row.yellow_cards = rec.yellow_cards
            row.red_cards = rec.red_cards
            row.extra = dict(rec.extra or {})
            row.retrieved_at = _naive(rec.retrieved_at) or _utcnow()

        self.session.flush()
        return report

    def ingest_availability(self, records: Sequence[AvailabilityRecord], *, source,
                            scope: str = "") -> IngestionReport:
        report = IngestionReport(source_key=getattr(source, "key", str(source)), scope=scope)
        if not records:
            report.ok = False
            report.message = "no records supplied"
            return report
        src = self.source_row(source)
        for rec in records:
            team = self.resolver.resolve(rec.team_name) if rec.team_name else None
            player = self.player_row(rec.player_name, team, rec.position, rec.external_ids, src)
            if player is None:
                report.skipped += 1
                continue
            observed = _naive(rec.observed_at) or _utcnow()
            existing = self.session.execute(
                select(PlayerAvailability).where(
                    PlayerAvailability.player_id == player.id,
                    PlayerAvailability.observed_at == observed,
                    PlayerAvailability.source_id == src.id,
                )
            ).scalars().first()
            if existing is not None:
                existing.status = rec.status
                existing.reason = rec.reason
                existing.chance_of_playing = rec.chance_of_playing
                report.updated += 1
                continue
            self.session.add(
                PlayerAvailability(
                    player_id=player.id,
                    team_id=team.id if team else None,
                    observed_at=observed,
                    status=rec.status,
                    reason=rec.reason,
                    chance_of_playing=rec.chance_of_playing,
                    expected_return=rec.expected_return,
                    source_id=src.id,
                    retrieved_at=_naive(rec.retrieved_at) or _utcnow(),
                )
            )
            report.written += 1
        self.session.flush()
        return report

    def ingest_lineups(self, records: Sequence[LineupRecord], *, match: Match,
                       source) -> IngestionReport:
        report = IngestionReport(source_key=getattr(source, "key", str(source)), scope="lineups")
        src = self.source_row(source)
        for rec in records:
            team = self.resolver.resolve(rec.team_name)
            if team is None:
                report.skipped += 1
                continue
            player = self.player_row(rec.player_name, team, src=src)
            if player is None:
                report.skipped += 1
                continue
            existing = self.session.execute(
                select(Lineup).where(
                    Lineup.match_id == match.id, Lineup.team_id == team.id,
                    Lineup.player_id == player.id, Lineup.kind == rec.kind,
                    Lineup.source_id == src.id,
                )
            ).scalars().first()
            if existing is not None:
                existing.is_starter = rec.is_starter
                report.updated += 1
                continue
            self.session.add(
                Lineup(match_id=match.id, team_id=team.id, player_id=player.id, kind=rec.kind,
                       is_starter=rec.is_starter, confidence=rec.confidence, source_id=src.id,
                       retrieved_at=_naive(rec.retrieved_at) or _utcnow())
            )
            report.written += 1
        self.session.flush()
        return report

    def ingest_conditions(self, record: ConditionsRecord, *, match: Match,
                          source) -> IngestionReport:
        report = IngestionReport(source_key=getattr(source, "key", str(source)), scope="weather")
        src = self.source_row(source)
        row = self.session.execute(
            select(MatchConditions).where(
                MatchConditions.match_id == match.id, MatchConditions.source_id == src.id
            )
        ).scalars().first()
        if row is None:
            row = MatchConditions(match_id=match.id, source_id=src.id)
            self.session.add(row)
            report.written += 1
        else:
            report.updated += 1
        row.temperature_c = record.temperature_c
        row.wind_kph = record.wind_kph
        row.precipitation_mm = record.precipitation_mm
        row.humidity_pct = record.humidity_pct
        row.description = record.description
        row.surface = record.surface
        row.raw = dict(record.raw or {})
        row.retrieved_at = _naive(record.retrieved_at) or _utcnow()
        self.session.flush()
        return report

    def log_run(self, report: IngestionReport, started: dt.datetime) -> None:
        self.session.add(
            IngestionRun(
                started_at=started,
                finished_at=_utcnow(),
                source_key=report.source_key,
                scope=report.scope,
                ok=report.ok,
                records_written=report.written,
                records_skipped=report.skipped,
                conflicts=len(report.conflicts),
                message=report.message or "; ".join(report.conflicts[:5]),
            )
        )
