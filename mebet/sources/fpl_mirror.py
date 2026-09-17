"""Adapter: the community mirror of Fantasy Premier League data.

Why this source matters: it is the only freely reachable feed found that
carries **per-player, per-match** statistics for the Premier League including
expected goals, expected assists and expected goals conceded, back several
seasons. That is what makes player-level predictions and xG-based team
ratings possible rather than aspirational.

It is a mirror maintained by a volunteer, so it can lag the live API by days.
The lag is measured (``detail['snapshot_lag_days']``) and handed to the data
quality layer instead of being ignored.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Optional

from ..logging_setup import get_logger
from ..normalize import AvailabilityRecord, MatchRecord, PlayerMatchRecord
from ..normalize.footballdata_csv import to_float, to_int
from .base import Capability, SourceAdapter, SourceResponse, SourceStatus
from .registry import register

log = get_logger("sources.fpl_mirror")

BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
#: Season in which FPL began publishing expected-goals columns.
XG_FROM_SEASON = "2022-23"


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _parse_kickoff(value: str) -> Optional[dt.datetime]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        return dt.datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError:
        return None


@register
class FplMirrorAdapter(SourceAdapter):
    key = "fpl_mirror"
    name = "Fantasy Premier League historical mirror (vaastav)"
    homepage = "https://github.com/vaastav/Fantasy-Premier-League"
    license = "Public data mirror; see repository"
    capabilities = (Capability.PLAYER_STATS, Capability.FIXTURES, Capability.MATCH_RESULTS)
    sports = ("football",)
    # Community mirror of an official feed: trustworthy values, possible lag.
    reliability = 0.7
    min_interval = 0.4

    #: Only the Premier League is covered by FPL.
    COMPETITIONS = {"ENG.1"}

    def status(self) -> SourceStatus:
        res = self.fetch(f"{BASE}/2024-25/teams.csv", ttl=86400)
        return SourceStatus(
            key=self.key, available=res.ok,
            detail="reachable" if res.ok else f"unreachable: {res.error}",
        )

    # -- helpers ---------------------------------------------------------
    def _teams(self, season: str) -> tuple[dict[int, str], str]:
        url = f"{BASE}/{season}/teams.csv"
        res = self.fetch(url, ttl=24 * 3600)
        if not res.ok:
            return {}, res.error
        mapping = {}
        for row in _rows(res.text):
            tid = to_int(row.get("id"))
            name = (row.get("name") or "").strip()
            if tid is not None and name:
                mapping[tid] = name
        return mapping, ""

    def _players(self, season: str) -> tuple[dict[int, dict], str]:
        url = f"{BASE}/{season}/players_raw.csv"
        res = self.fetch(url, ttl=12 * 3600)
        if not res.ok:
            return {}, res.error
        out = {}
        for row in _rows(res.text):
            pid = to_int(row.get("id"))
            if pid is None:
                continue
            out[pid] = row
        return out, ""

    # -- capabilities ----------------------------------------------------
    def fetch_player_stats(self, *, competition_key: str, season: str,
                           sport: str = "football") -> SourceResponse:
        if competition_key not in self.COMPETITIONS:
            return SourceResponse.failure(
                self.key, f"{self.key} covers the Premier League only, not {competition_key}"
            )
        teams, err = self._teams(season)
        if err:
            return SourceResponse.failure(self.key, f"team list unavailable for {season}: {err}")

        positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        players, _ = self._players(season)
        element_team = {
            pid: teams.get(to_int(row.get("team")) or -1, "")
            for pid, row in players.items()
        }

        records: list[PlayerMatchRecord] = []
        urls: list[str] = []
        latest_kickoff: Optional[dt.datetime] = None
        gw = 1
        misses = 0
        # Gameweeks are published one file at a time; stop after two
        # consecutive absences so a postponed-blank week does not truncate.
        while gw <= 47 and misses < 2:
            url = f"{BASE}/{season}/gws/gw{gw}.csv"
            res = self.fetch(url, ttl=6 * 3600)
            if not res.ok:
                misses += 1
                gw += 1
                continue
            misses = 0
            urls.append(url)
            for row in _rows(res.text):
                kickoff = _parse_kickoff(row.get("kickoff_time", ""))
                if kickoff is None:
                    continue
                if latest_kickoff is None or kickoff > latest_kickoff:
                    latest_kickoff = kickoff
                element = to_int(row.get("element"))
                team_name = (row.get("team") or "").strip() or element_team.get(element or -1, "")
                opp = teams.get(to_int(row.get("opponent_team")) or -1, "")
                pos = (row.get("position") or "").strip()
                if not pos and element in players:
                    pos = positions.get(to_int(players[element].get("element_type")) or -1, "")
                was_home = (row.get("was_home") or "").strip().lower() in {"true", "1"}
                records.append(
                    PlayerMatchRecord(
                        sport="football",
                        player_name=(row.get("name") or "").strip(),
                        team_name=team_name,
                        opponent_name=opp,
                        match_date=kickoff.date(),
                        was_home=was_home,
                        position=pos,
                        minutes=to_int(row.get("minutes")),
                        started=(to_int(row.get("starts")) or 0) > 0 if row.get("starts") else None,
                        goals=to_int(row.get("goals_scored")),
                        assists=to_int(row.get("assists")),
                        # FPL publishes expected goals/assists but not shots,
                        # so shots stay None rather than being approximated.
                        xg=to_float(row.get("expected_goals")),
                        xa=to_float(row.get("expected_assists")),
                        xgc=to_float(row.get("expected_goals_conceded")),
                        tackles=to_int(row.get("tackles")),
                        saves=to_int(row.get("saves")),
                        yellow_cards=to_int(row.get("yellow_cards")),
                        red_cards=to_int(row.get("red_cards")),
                        extra={
                            "round": to_int(row.get("round")),
                            "bps": to_int(row.get("bps")),
                            "goals_conceded": to_int(row.get("goals_conceded")),
                            "clean_sheets": to_int(row.get("clean_sheets")),
                        },
                        external_ids={"fpl_element": element},
                        source_key=self.key,
                        retrieved_at=res.fetched_at,
                    )
                )
            gw += 1

        if not records:
            return SourceResponse.failure(
                self.key, f"no gameweek files published yet for {season}", urls
            )

        lag = None
        if latest_kickoff is not None:
            lag = (dt.datetime.now(dt.timezone.utc) - latest_kickoff).days
        return SourceResponse(
            source_key=self.key, ok=True, records=records, urls=urls,
            detail={
                "season": season,
                "gameweeks_loaded": len(urls),
                "latest_match": latest_kickoff.isoformat() if latest_kickoff else None,
                "snapshot_lag_days": lag,
                "has_expected_goals": season >= XG_FROM_SEASON,
            },
        )

    def fetch_fixtures(self, *, competition_key: str, sport: str = "football",
                       season: Optional[str] = None) -> SourceResponse:
        if competition_key not in self.COMPETITIONS:
            return SourceResponse.failure(self.key, f"{competition_key} not covered")
        season = season or current_fpl_season()
        teams, err = self._teams(season)
        if err:
            return SourceResponse.failure(self.key, f"team list unavailable: {err}")
        url = f"{BASE}/{season}/fixtures.csv"
        res = self.fetch(url, ttl=3 * 3600)
        if not res.ok:
            return SourceResponse.failure(self.key, f"fixtures unavailable: {res.error}", [url])

        records: list[MatchRecord] = []
        for row in _rows(res.text):
            home = teams.get(to_int(row.get("team_h")) or -1, "")
            away = teams.get(to_int(row.get("team_a")) or -1, "")
            kickoff = _parse_kickoff(row.get("kickoff_time", ""))
            if not home or not away or kickoff is None:
                continue
            finished = (row.get("finished") or "").strip().lower() in {"true", "1"}
            hg, ag = to_int(row.get("team_h_score")), to_int(row.get("team_a_score"))
            records.append(
                MatchRecord(
                    sport="football",
                    competition_key=competition_key,
                    competition_name="Premier League",
                    season_label=season,
                    home_team=home,
                    away_team=away,
                    kickoff_date=kickoff.date(),
                    kickoff_utc=kickoff,
                    kickoff_is_exact=True,
                    status="played" if finished and hg is not None else "scheduled",
                    ft_home_goals=hg if finished else None,
                    ft_away_goals=ag if finished else None,
                    matchday=to_int(row.get("event")),
                    external_ids={"fpl_fixture": to_int(row.get("id"))},
                    source_key=self.key,
                    retrieved_at=res.fetched_at,
                )
            )
        played = sum(1 for r in records if r.status == "played")
        return SourceResponse(
            source_key=self.key, ok=True, records=records, urls=[url],
            retrieved_at=res.fetched_at, from_cache=res.from_cache,
            detail={"season": season, "fixtures": len(records), "played": played},
        )

    def fetch_matches(self, *, competition_key: str, season: str,
                      sport: str = "football") -> SourceResponse:
        return self.fetch_fixtures(competition_key=competition_key, sport=sport, season=season)


def current_fpl_season(today: Optional[dt.date] = None) -> str:
    """FPL seasons run August-May; before August we are still in last season."""
    today = today or dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"
