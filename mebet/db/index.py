"""In-memory match index.

Feature building is inherently repetitive: the logistic model needs the
feature set as it stood before each of a thousand past matches, and the
backtester needs the same thing thousands of times over. Served straight from
SQL that is hundreds of queries per row and minutes per fit.

The index loads every match and its statistics once, then answers the same
questions from memory. Crucially it does *not* weaken the leakage guarantee:
the index is loaded with an upper bound, every read still takes an explicit
cutoff, filtering uses the same "concluded before the cutoff" rule as the SQL
path, and ``AsOfRepository`` still verifies what it hands back.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..logging_setup import get_logger
from .models import DataSource, Match, TeamMatchStats

log = get_logger("db.index")


@dataclass
class StatLine:
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    corners: Optional[int] = None
    fouls: Optional[int] = None
    yellow_cards: Optional[int] = None
    red_cards: Optional[int] = None
    possession: Optional[float] = None
    xg: Optional[float] = None
    xga: Optional[float] = None
    source_id: Optional[int] = None


@dataclass
class IndexedMatch:
    """Duck-type compatible with ``Match`` for the fields features use."""

    id: int
    sport: str
    competition_id: int
    season_id: Optional[int]
    kickoff_date: dt.date
    kickoff_utc: Optional[dt.datetime]
    kickoff_is_exact: bool
    status: str
    home_team_id: int
    away_team_id: int
    ft_home_goals: Optional[int]
    ft_away_goals: Optional[int]
    ht_home_goals: Optional[int]
    ht_away_goals: Optional[int]
    referee: str
    neutral_venue: bool
    source_id: Optional[int]
    home_stats: StatLine = field(default_factory=StatLine)
    away_stats: StatLine = field(default_factory=StatLine)

    @property
    def result(self) -> Optional[str]:
        if self.ft_home_goals is None or self.ft_away_goals is None:
            return None
        if self.ft_home_goals > self.ft_away_goals:
            return "H"
        if self.ft_home_goals < self.ft_away_goals:
            return "A"
        return "D"

    def stats_for(self, team_id: int) -> StatLine:
        return self.home_stats if team_id == self.home_team_id else self.away_stats


class MatchIndex:
    def __init__(
        self,
        session: Session,
        *,
        sport: str = "football",
        upper_bound: Optional[dt.date] = None,
        competition_ids: Optional[Iterable[int]] = None,
        played_only: bool = True,
    ) -> None:
        self.sport = sport
        self.upper_bound = upper_bound
        self.matches: list[IndexedMatch] = []
        self.by_id: dict[int, IndexedMatch] = {}
        self.by_team: dict[int, list[IndexedMatch]] = {}
        self.by_competition: dict[int, list[IndexedMatch]] = {}
        self._load(session, competition_ids, played_only)

    def _load(self, session: Session, competition_ids, played_only: bool) -> None:
        stmt = select(Match).where(Match.sport == self.sport)
        if played_only:
            stmt = stmt.where(Match.status == "played")
        if self.upper_bound is not None:
            stmt = stmt.where(Match.kickoff_date <= self.upper_bound)
        if competition_ids is not None:
            ids = list(competition_ids)
            if not ids:
                return
            stmt = stmt.where(Match.competition_id.in_(ids))

        rows = list(session.execute(stmt).scalars())
        if not rows:
            return
        match_ids = {m.id for m in rows}

        # Reliability ranking, used when two sources describe the same match.
        reliability = {
            s.id: s.reliability
            for s in session.execute(select(DataSource)).scalars()
        }

        stats_stmt = select(TeamMatchStats).where(TeamMatchStats.match_id.in_(match_ids)) \
            if len(match_ids) < 30000 else select(TeamMatchStats)
        best: dict[tuple[int, int], TeamMatchStats] = {}
        for st in session.execute(stats_stmt).scalars():
            if st.match_id not in match_ids:
                continue
            key = (st.match_id, st.team_id)
            current = best.get(key)
            if current is None:
                best[key] = st
                continue
            if reliability.get(st.source_id, 0.0) > reliability.get(current.source_id, 0.0):
                best[key] = st

        def to_line(st: Optional[TeamMatchStats]) -> StatLine:
            if st is None:
                return StatLine()
            return StatLine(
                shots=st.shots, shots_on_target=st.shots_on_target, corners=st.corners,
                fouls=st.fouls, yellow_cards=st.yellow_cards, red_cards=st.red_cards,
                possession=st.possession, xg=st.xg, xga=st.xga, source_id=st.source_id,
            )

        for m in rows:
            entry = IndexedMatch(
                id=m.id, sport=m.sport, competition_id=m.competition_id, season_id=m.season_id,
                kickoff_date=m.kickoff_date, kickoff_utc=m.kickoff_utc,
                kickoff_is_exact=bool(m.kickoff_is_exact), status=m.status,
                home_team_id=m.home_team_id, away_team_id=m.away_team_id,
                ft_home_goals=m.ft_home_goals, ft_away_goals=m.ft_away_goals,
                ht_home_goals=m.ht_home_goals, ht_away_goals=m.ht_away_goals,
                referee=m.referee or "", neutral_venue=bool(m.neutral_venue),
                source_id=m.source_id,
                home_stats=to_line(best.get((m.id, m.home_team_id))),
                away_stats=to_line(best.get((m.id, m.away_team_id))),
            )
            self.matches.append(entry)
            self.by_id[entry.id] = entry

        # Newest first everywhere, so windows are a simple slice.
        self.matches.sort(key=lambda m: (m.kickoff_date, m.id), reverse=True)
        for entry in self.matches:
            self.by_team.setdefault(entry.home_team_id, []).append(entry)
            self.by_team.setdefault(entry.away_team_id, []).append(entry)
            self.by_competition.setdefault(entry.competition_id, []).append(entry)

        log.info(
            "match index: %d matches, %d teams, %d competitions%s",
            len(self.matches), len(self.by_team), len(self.by_competition),
            f", up to {self.upper_bound}" if self.upper_bound else "",
        )

    # -- queries -----------------------------------------------------------
    def team_matches(self, team_id: int, *, concluded_before,
                     competition_ids: Optional[set[int]] = None,
                     since: Optional[dt.date] = None,
                     venue: Optional[str] = None,
                     limit: Optional[int] = None) -> list[IndexedMatch]:
        out = []
        for m in self.by_team.get(team_id, ()):
            if not concluded_before(m):
                continue
            if competition_ids is not None and m.competition_id not in competition_ids:
                continue
            if since is not None and m.kickoff_date < since:
                continue
            if venue == "home" and m.home_team_id != team_id:
                continue
            if venue == "away" and m.away_team_id != team_id:
                continue
            out.append(m)
            if limit is not None and len(out) >= limit:
                break
        return out

    def competition_matches(self, competition_id: int, *, concluded_before,
                            since: Optional[dt.date] = None,
                            limit: Optional[int] = None) -> list[IndexedMatch]:
        out = []
        for m in self.by_competition.get(competition_id, ()):
            if not concluded_before(m):
                continue
            if since is not None and m.kickoff_date < since:
                continue
            out.append(m)
            if limit is not None and len(out) >= limit:
                break
        return out

    def head_to_head(self, team_a: int, team_b: int, *, concluded_before,
                     limit: int = 20) -> list[IndexedMatch]:
        out = []
        for m in self.by_team.get(team_a, ()):
            if m.home_team_id != team_b and m.away_team_id != team_b:
                continue
            if not concluded_before(m):
                continue
            out.append(m)
            if len(out) >= limit:
                break
        return out

    def all_matches(self, *, concluded_before, competition_ids: Optional[set[int]] = None,
                    since: Optional[dt.date] = None) -> list[IndexedMatch]:
        out = []
        for m in self.matches:
            if competition_ids is not None and m.competition_id not in competition_ids:
                continue
            if since is not None and m.kickoff_date < since:
                continue
            if not concluded_before(m):
                continue
            out.append(m)
        return out
