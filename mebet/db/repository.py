"""Read access to stored observations, with as-of cutoffs enforced in SQL.

Every read used for feature building or model fitting goes through
``AsOfRepository``. The cutoff is a constructor argument, not an optional
keyword, so there is no code path that reads "all data" by accident.

Two distinct cutoffs matter and both are applied:

*   **Event time** - a match may only inform a prediction if the match had
    *finished* before the cutoff. A fixture that kicked off 20 minutes before
    the cutoff has no known result yet, so it is excluded.
*   **Observation time** - time-varying records (injuries, lineups, weather
    forecasts) may only be used if they were observed at or before the
    cutoff. Today's injury list must not inform a prediction dated 2023.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from ..logging_setup import get_logger
from .index import IndexedMatch, MatchIndex
from .models import (
    Competition,
    Lineup,
    Match,
    MatchConditions,
    Player,
    PlayerAvailability,
    PlayerMatchStats,
    Team,
    TeamMatchStats,
)

log = get_logger("db.repository")

#: A match is treated as concluded this long after kickoff. Used so that an
#: in-progress fixture is never mistaken for a finished one.
MATCH_DURATION = dt.timedelta(hours=2, minutes=30)


class LeakageError(RuntimeError):
    """Raised when data at or after the cutoff reaches the feature layer."""


def _as_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _naive_utc(value: dt.datetime) -> dt.datetime:
    """SQLite stores naive datetimes; normalise consistently to naive UTC."""
    return _as_utc(value).replace(tzinfo=None)


@dataclass(frozen=True)
class TeamMatchRow:
    """A single team's view of one completed match."""

    match_id: int
    date: dt.date
    kickoff: Optional[dt.datetime]
    competition_id: int
    team_id: int
    opponent_id: int
    is_home: bool
    goals_for: Optional[int]
    goals_against: Optional[int]
    ht_goals_for: Optional[int]
    ht_goals_against: Optional[int]
    shots: Optional[int]
    shots_on_target: Optional[int]
    shots_against: Optional[int]
    sot_against: Optional[int]
    corners: Optional[int]
    corners_against: Optional[int]
    fouls: Optional[int]
    fouls_against: Optional[int]
    yellows: Optional[int]
    yellows_against: Optional[int]
    reds: Optional[int]
    reds_against: Optional[int]
    xg: Optional[float]
    xga: Optional[float]
    source_ids: tuple[int, ...] = ()

    @property
    def points(self) -> Optional[int]:
        if self.goals_for is None or self.goals_against is None:
            return None
        if self.goals_for > self.goals_against:
            return 3
        if self.goals_for == self.goals_against:
            return 1
        return 0

    @property
    def outcome(self) -> Optional[str]:
        if self.goals_for is None or self.goals_against is None:
            return None
        if self.goals_for > self.goals_against:
            return "W"
        if self.goals_for == self.goals_against:
            return "D"
        return "L"


class AsOfRepository:
    """All reads are bounded by ``as_of``.

    ``as_of`` is an aware or naive UTC datetime; naive values are treated as
    UTC. Construction with no cutoff is intentionally impossible.
    """

    def __init__(self, session: Session, as_of: dt.datetime, *, sport: str = "football",
                 strict: bool = True, index: Optional[MatchIndex] = None) -> None:
        if as_of is None:  # pragma: no cover - defensive
            raise ValueError("as_of is required; unbounded reads are not permitted")
        self.session = session
        self.as_of = _naive_utc(as_of)
        self.sport = sport
        self.strict = strict
        # When an index is supplied, match reads are served from memory. The
        # cutoff rule and the verification step are identical either way.
        self.index = index
        self._row_cache: dict[tuple, list[TeamMatchRow]] = {}

    def concluded_before_cutoff(self, match) -> bool:
        """The in-memory equivalent of ``_completed_before_cutoff``."""
        if match.status != "played":
            return False
        if match.kickoff_is_exact and match.kickoff_utc is not None:
            return match.kickoff_utc <= self.as_of - MATCH_DURATION
        return match.kickoff_date < self.as_of.date()

    # -- internal helpers ------------------------------------------------
    def _completed_before_cutoff(self):
        """SQL predicate: the match had finished before the cutoff."""
        exact = and_(
            Match.kickoff_is_exact.is_(True),
            Match.kickoff_utc.is_not(None),
            Match.kickoff_utc <= self.as_of - MATCH_DURATION,
        )
        # Date-only rows: require the calendar date to be strictly earlier
        # than the cutoff date, which cannot overlap the cutoff instant.
        dateonly = and_(
            or_(Match.kickoff_is_exact.is_(False), Match.kickoff_utc.is_(None)),
            Match.kickoff_date < self.as_of.date(),
        )
        return and_(Match.status == "played", or_(exact, dateonly))

    def _verify(self, matches: Sequence[Match]) -> None:
        """Belt-and-braces check that the SQL predicate did its job."""
        if not self.strict:
            return
        limit_date = self.as_of.date()
        for m in matches:
            if m.kickoff_is_exact and m.kickoff_utc is not None:
                if m.kickoff_utc > self.as_of - MATCH_DURATION:
                    raise LeakageError(
                        f"match {m.id} (kickoff {m.kickoff_utc}) is not concluded before cutoff {self.as_of}"
                    )
            elif m.kickoff_date >= limit_date:
                raise LeakageError(
                    f"match {m.id} (date {m.kickoff_date}) is not before cutoff date {limit_date}"
                )

    # -- matches ---------------------------------------------------------
    def completed_matches(
        self,
        *,
        competition_ids: Optional[Iterable[int]] = None,
        since: Optional[dt.date] = None,
        team_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Match]:
        if self.index is not None:
            comps = set(competition_ids) if competition_ids is not None else None
            if team_id is not None:
                rows = self.index.team_matches(
                    team_id, concluded_before=self.concluded_before_cutoff,
                    competition_ids=comps, since=since, limit=limit,
                )
            else:
                rows = self.index.all_matches(
                    concluded_before=self.concluded_before_cutoff,
                    competition_ids=comps, since=since,
                )
                if limit is not None:
                    rows = rows[:limit]
            self._verify(rows)
            return rows

        stmt = select(Match).where(Match.sport == self.sport, self._completed_before_cutoff())
        if competition_ids is not None:
            ids = list(competition_ids)
            if not ids:
                return []
            stmt = stmt.where(Match.competition_id.in_(ids))
        if since is not None:
            stmt = stmt.where(Match.kickoff_date >= since)
        if team_id is not None:
            stmt = stmt.where(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
        stmt = stmt.order_by(Match.kickoff_date.desc(), Match.id.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = list(self.session.execute(stmt).scalars())
        self._verify(rows)
        return rows

    def team_history(
        self,
        team_id: int,
        *,
        limit: Optional[int] = None,
        competition_ids: Optional[Iterable[int]] = None,
        since: Optional[dt.date] = None,
        venue: Optional[str] = None,  # "home" | "away" | None
    ) -> list[TeamMatchRow]:
        """Completed matches for one team, newest first, as team-oriented rows."""
        cache_key = (team_id, limit, tuple(competition_ids) if competition_ids else None,
                     since, venue)
        cached = self._row_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.index is not None:
            comps = set(competition_ids) if competition_ids is not None else None
            matches = self.index.team_matches(
                team_id, concluded_before=self.concluded_before_cutoff,
                competition_ids=comps, since=since, venue=venue, limit=limit,
            )
            self._verify(matches)
            rows = [self._indexed_to_row(m, team_id) for m in matches]
            self._row_cache[cache_key] = rows
            return rows

        stmt = select(Match).where(Match.sport == self.sport, self._completed_before_cutoff())
        if venue == "home":
            stmt = stmt.where(Match.home_team_id == team_id)
        elif venue == "away":
            stmt = stmt.where(Match.away_team_id == team_id)
        else:
            stmt = stmt.where(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
        if competition_ids is not None:
            ids = list(competition_ids)
            if not ids:
                return []
            stmt = stmt.where(Match.competition_id.in_(ids))
        if since is not None:
            stmt = stmt.where(Match.kickoff_date >= since)
        stmt = stmt.order_by(Match.kickoff_date.desc(), Match.id.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        matches = list(self.session.execute(stmt).scalars())
        self._verify(matches)
        rows = [self._to_team_row(m, team_id) for m in matches]
        self._row_cache[cache_key] = rows
        return rows

    def _indexed_to_row(self, m: IndexedMatch, team_id: int) -> TeamMatchRow:
        is_home = m.home_team_id == team_id
        own = m.home_stats if is_home else m.away_stats
        opp = m.away_stats if is_home else m.home_stats
        return TeamMatchRow(
            match_id=m.id, date=m.kickoff_date, kickoff=m.kickoff_utc,
            competition_id=m.competition_id, team_id=team_id,
            opponent_id=m.away_team_id if is_home else m.home_team_id,
            is_home=is_home,
            goals_for=m.ft_home_goals if is_home else m.ft_away_goals,
            goals_against=m.ft_away_goals if is_home else m.ft_home_goals,
            ht_goals_for=m.ht_home_goals if is_home else m.ht_away_goals,
            ht_goals_against=m.ht_away_goals if is_home else m.ht_home_goals,
            shots=own.shots, shots_on_target=own.shots_on_target,
            shots_against=opp.shots, sot_against=opp.shots_on_target,
            corners=own.corners, corners_against=opp.corners,
            fouls=own.fouls, fouls_against=opp.fouls,
            yellows=own.yellow_cards, yellows_against=opp.yellow_cards,
            reds=own.red_cards, reds_against=opp.red_cards,
            xg=own.xg, xga=own.xga,
            source_ids=tuple(sorted({s for s in (own.source_id, opp.source_id) if s})),
        )

    def _stats_for(self, match_id: int) -> dict[int, TeamMatchStats]:
        rows = self.session.execute(
            select(TeamMatchStats).where(TeamMatchStats.match_id == match_id)
        ).scalars()
        # If several sources describe the same team/match, prefer the most
        # reliable-by-recency row; conflicts are surfaced by the quality layer.
        best: dict[int, TeamMatchStats] = {}
        for r in rows:
            cur = best.get(r.team_id)
            if cur is None or (r.retrieved_at or dt.datetime.min) > (cur.retrieved_at or dt.datetime.min):
                best[r.team_id] = r
        return best

    def _to_team_row(self, m: Match, team_id: int) -> TeamMatchRow:
        is_home = m.home_team_id == team_id
        opponent_id = m.away_team_id if is_home else m.home_team_id
        stats = self._stats_for(m.id)
        own = stats.get(team_id)
        opp = stats.get(opponent_id)

        gf = m.ft_home_goals if is_home else m.ft_away_goals
        ga = m.ft_away_goals if is_home else m.ft_home_goals
        htf = m.ht_home_goals if is_home else m.ht_away_goals
        hta = m.ht_away_goals if is_home else m.ht_home_goals

        src = tuple(sorted({s.source_id for s in (own, opp) if s is not None}))
        return TeamMatchRow(
            match_id=m.id,
            date=m.kickoff_date,
            kickoff=m.kickoff_utc,
            competition_id=m.competition_id,
            team_id=team_id,
            opponent_id=opponent_id,
            is_home=is_home,
            goals_for=gf,
            goals_against=ga,
            ht_goals_for=htf,
            ht_goals_against=hta,
            shots=own.shots if own else None,
            shots_on_target=own.shots_on_target if own else None,
            shots_against=opp.shots if opp else None,
            sot_against=opp.shots_on_target if opp else None,
            corners=own.corners if own else None,
            corners_against=opp.corners if opp else None,
            fouls=own.fouls if own else None,
            fouls_against=opp.fouls if opp else None,
            yellows=own.yellow_cards if own else None,
            yellows_against=opp.yellow_cards if opp else None,
            reds=own.red_cards if own else None,
            reds_against=opp.red_cards if opp else None,
            xg=own.xg if own else None,
            xga=own.xga if own else None,
            source_ids=src,
        )

    def stat_series(
        self,
        *,
        competition_ids: Optional[Iterable[int]] = None,
        since: Optional[dt.date] = None,
    ) -> list[dict]:
        """Bulk per-team, per-match statistics in one query.

        ``team_history`` is convenient but issues a query per match, which is
        far too slow for models that need every team's corner or card record
        over several seasons. This returns the same underlying facts in one
        pass, still bounded by the cutoff.
        """
        if self.index is not None:
            comps = set(competition_ids) if competition_ids is not None else None
            out: list[dict] = []
            for m in self.index.all_matches(
                concluded_before=self.concluded_before_cutoff,
                competition_ids=comps, since=since,
            ):
                for team_id, opponent_id, line, is_home, goals in (
                    (m.home_team_id, m.away_team_id, m.home_stats, True, m.ft_home_goals),
                    (m.away_team_id, m.home_team_id, m.away_stats, False, m.ft_away_goals),
                ):
                    out.append({
                        "match_id": m.id, "team_id": team_id, "opponent_id": opponent_id,
                        "is_home": is_home, "date": m.kickoff_date,
                        "competition_id": m.competition_id, "referee": m.referee,
                        "corners": line.corners, "yellows": line.yellow_cards,
                        "reds": line.red_cards, "fouls": line.fouls, "shots": line.shots,
                        "shots_on_target": line.shots_on_target, "goals": goals,
                    })
            if self.strict and out:
                latest = max(r["date"] for r in out)
                if latest >= self.as_of.date():
                    raise LeakageError(
                        f"stat_series returned a match dated {latest} at cutoff {self.as_of.date()}"
                    )
            return out

        stmt = (
            select(
                TeamMatchStats.match_id,
                TeamMatchStats.team_id,
                TeamMatchStats.is_home,
                TeamMatchStats.corners,
                TeamMatchStats.yellow_cards,
                TeamMatchStats.red_cards,
                TeamMatchStats.fouls,
                TeamMatchStats.shots,
                TeamMatchStats.shots_on_target,
                TeamMatchStats.goals,
                Match.kickoff_date,
                Match.competition_id,
                Match.home_team_id,
                Match.away_team_id,
                Match.referee,
            )
            .join(Match, Match.id == TeamMatchStats.match_id)
            .where(Match.sport == self.sport, self._completed_before_cutoff())
        )
        if competition_ids is not None:
            ids = list(competition_ids)
            if not ids:
                return []
            stmt = stmt.where(Match.competition_id.in_(ids))
        if since is not None:
            stmt = stmt.where(Match.kickoff_date >= since)

        out: list[dict] = []
        for row in self.session.execute(stmt.order_by(Match.kickoff_date)):
            opponent = row.away_team_id if row.is_home else row.home_team_id
            out.append(
                {
                    "match_id": row.match_id,
                    "team_id": row.team_id,
                    "opponent_id": opponent,
                    "is_home": bool(row.is_home),
                    "date": row.kickoff_date,
                    "competition_id": row.competition_id,
                    "referee": row.referee or "",
                    "corners": row.corners,
                    "yellows": row.yellow_cards,
                    "reds": row.red_cards,
                    "fouls": row.fouls,
                    "shots": row.shots,
                    "shots_on_target": row.shots_on_target,
                    "goals": row.goals,
                }
            )
        # Guard: the join must not have admitted a post-cutoff match.
        if self.strict and out:
            latest = max(r["date"] for r in out)
            if latest >= self.as_of.date():
                raise LeakageError(
                    f"stat_series returned a match dated {latest} at cutoff {self.as_of.date()}"
                )
        return out

    def stat_pair(self, match) -> tuple:
        """(home, away) statistic lines for a match, from whichever backing
        store this repository is using."""
        if isinstance(match, IndexedMatch):
            return match.home_stats, match.away_stats
        stats = self._stats_for(match.id)
        return stats.get(match.home_team_id), stats.get(match.away_team_id)

    def head_to_head(self, team_a: int, team_b: int, *, limit: int = 20) -> list[TeamMatchRow]:
        if self.index is not None:
            matches = self.index.head_to_head(
                team_a, team_b, concluded_before=self.concluded_before_cutoff, limit=limit
            )
            self._verify(matches)
            return [self._indexed_to_row(m, team_a) for m in matches]

        stmt = (
            select(Match)
            .where(
                Match.sport == self.sport,
                self._completed_before_cutoff(),
                or_(
                    and_(Match.home_team_id == team_a, Match.away_team_id == team_b),
                    and_(Match.home_team_id == team_b, Match.away_team_id == team_a),
                ),
            )
            .order_by(Match.kickoff_date.desc())
            .limit(limit)
        )
        matches = list(self.session.execute(stmt).scalars())
        self._verify(matches)
        return [self._to_team_row(m, team_a) for m in matches]

    def last_match_before(self, team_id: int) -> Optional[Match]:
        rows = self.completed_matches(team_id=team_id, limit=1)
        return rows[0] if rows else None

    def matches_in_window(self, team_id: int, days: int) -> list[Match]:
        start = (self.as_of - dt.timedelta(days=days)).date()
        return self.completed_matches(team_id=team_id, since=start)

    # -- player data -----------------------------------------------------
    def player_history(
        self, player_id: int, *, limit: Optional[int] = None
    ) -> list[PlayerMatchStats]:
        stmt = (
            select(PlayerMatchStats)
            .where(
                PlayerMatchStats.player_id == player_id,
                PlayerMatchStats.match_date < self.as_of.date(),
            )
            .order_by(PlayerMatchStats.match_date.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars())

    def squad(self, team_id: int) -> list[Player]:
        return list(
            self.session.execute(
                select(Player).where(Player.team_id == team_id, Player.sport == self.sport)
            ).scalars()
        )

    def team_player_history(
        self, team_id: int, *, since: Optional[dt.date] = None
    ) -> list[PlayerMatchStats]:
        stmt = select(PlayerMatchStats).where(
            PlayerMatchStats.team_id == team_id,
            PlayerMatchStats.match_date < self.as_of.date(),
        )
        if since is not None:
            stmt = stmt.where(PlayerMatchStats.match_date >= since)
        return list(self.session.execute(stmt.order_by(PlayerMatchStats.match_date.desc())).scalars())

    # -- time-varying records (observation-time bounded) ------------------
    def availability(self, team_id: int) -> list[PlayerAvailability]:
        """Latest availability record per player, as known at the cutoff."""
        rows = list(
            self.session.execute(
                select(PlayerAvailability)
                .where(
                    PlayerAvailability.team_id == team_id,
                    PlayerAvailability.observed_at <= self.as_of,
                    PlayerAvailability.retrieved_at <= self.as_of,
                )
                .order_by(PlayerAvailability.observed_at.desc())
            ).scalars()
        )
        latest: dict[int, PlayerAvailability] = {}
        for r in rows:
            latest.setdefault(r.player_id, r)
        return list(latest.values())

    def lineups(self, match_id: int) -> list[Lineup]:
        return list(
            self.session.execute(
                select(Lineup).where(
                    Lineup.match_id == match_id,
                    Lineup.retrieved_at <= self.as_of,
                )
            ).scalars()
        )

    def conditions(self, match_id: int) -> Optional[MatchConditions]:
        return self.session.execute(
            select(MatchConditions)
            .where(MatchConditions.match_id == match_id, MatchConditions.retrieved_at <= self.as_of)
            .order_by(MatchConditions.retrieved_at.desc())
            .limit(1)
        ).scalars().first()

    # -- lookups (identity data, not time-varying) -----------------------
    def team(self, team_id: int) -> Optional[Team]:
        return self.session.get(Team, team_id)

    def competition(self, competition_id: int) -> Optional[Competition]:
        return self.session.get(Competition, competition_id)

    def league_table(self, competition_id: int, season_start: dt.date) -> dict[int, dict]:
        """Standings computed from matches completed before the cutoff.

        Derived rather than fetched, which means it is automatically correct
        as of any historical date and cannot import a future standing.
        """
        matches = self.completed_matches(competition_ids=[competition_id], since=season_start)
        table: dict[int, dict] = {}

        def slot(tid: int) -> dict:
            return table.setdefault(
                tid, {"team_id": tid, "played": 0, "won": 0, "drawn": 0, "lost": 0,
                      "gf": 0, "ga": 0, "points": 0}
            )

        for m in matches:
            if m.ft_home_goals is None or m.ft_away_goals is None:
                continue
            h, a = slot(m.home_team_id), slot(m.away_team_id)
            h["played"] += 1
            a["played"] += 1
            h["gf"] += m.ft_home_goals
            h["ga"] += m.ft_away_goals
            a["gf"] += m.ft_away_goals
            a["ga"] += m.ft_home_goals
            if m.ft_home_goals > m.ft_away_goals:
                h["won"] += 1; a["lost"] += 1; h["points"] += 3
            elif m.ft_home_goals < m.ft_away_goals:
                a["won"] += 1; h["lost"] += 1; a["points"] += 3
            else:
                h["drawn"] += 1; a["drawn"] += 1; h["points"] += 1; a["points"] += 1

        ranked = sorted(
            table.values(),
            key=lambda r: (-r["points"], -(r["gf"] - r["ga"]), -r["gf"]),
        )
        for pos, row in enumerate(ranked, start=1):
            row["position"] = pos
            row["goal_difference"] = row["gf"] - row["ga"]
            row["ppg"] = round(row["points"] / row["played"], 3) if row["played"] else None
        return {r["team_id"]: r for r in ranked}
