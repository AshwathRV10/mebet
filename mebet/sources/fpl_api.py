"""Adapter: the live Fantasy Premier League API.

This is the availability feed. For every Premier League player the official
endpoint publishes a squad status ('a' available, 'd' doubtful, 'i' injured,
's' suspended, 'u' unavailable, 'n' not in squad), a percentage chance of
playing the next round, a free-text news line and the timestamp that news was
added. That is real, first-party team news - not a guess derived from
absence.

It does not publish lineups, so lineups remain unavailable from this source
and the engine reports them as such.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from ..logging_setup import get_logger
from ..normalize import AvailabilityRecord, MatchRecord
from .base import Capability, SourceAdapter, SourceResponse, SourceStatus
from .registry import register

log = get_logger("sources.fpl_api")

BASE = "https://fantasy.premierleague.com/api"

#: FPL status code -> canonical availability status.
STATUS_MAP = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "unavailable",
}


@register
class FplApiAdapter(SourceAdapter):
    key = "fpl_api"
    name = "Fantasy Premier League official API"
    homepage = "https://fantasy.premierleague.com"
    license = "Public endpoint; unofficial for third-party use"
    capabilities = (Capability.AVAILABILITY, Capability.FIXTURES, Capability.PLAYER_STATS)
    sports = ("football",)
    # First-party feed for squad status: the most authoritative availability
    # signal reachable without a paid provider.
    reliability = 0.88
    min_interval = 2.0

    COMPETITIONS = {"ENG.1"}

    def status(self) -> SourceStatus:
        res = self.fetch(f"{BASE}/bootstrap-static/", ttl=1800)
        return SourceStatus(
            key=self.key, available=res.ok,
            detail="reachable" if res.ok else f"unreachable: {res.error}",
        )

    def _bootstrap(self, ttl: int = 1800):
        url = f"{BASE}/bootstrap-static/"
        res = self.fetch(url, ttl=ttl)
        if not res.ok:
            return None, url, res.error, res
        try:
            return res.json(), url, "", res
        except ValueError as exc:
            return None, url, f"malformed JSON: {exc}", res

    def fetch_availability(self, *, competition_key: str,
                           sport: str = "football") -> SourceResponse:
        if competition_key not in self.COMPETITIONS:
            return SourceResponse.failure(
                self.key, f"{self.key} covers the Premier League only, not {competition_key}"
            )
        payload, url, err, res = self._bootstrap()
        if payload is None:
            return SourceResponse.failure(self.key, f"availability unavailable: {err}", [url])

        teams = {t["id"]: t["name"] for t in payload.get("teams", [])}
        positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        records: list[AvailabilityRecord] = []
        for el in payload.get("elements", []):
            raw_status = (el.get("status") or "").lower()
            status = STATUS_MAP.get(raw_status, "unknown")
            chance = el.get("chance_of_playing_next_round")
            news = (el.get("news") or "").strip()
            news_added = el.get("news_added")
            observed = dt.datetime.now(dt.timezone.utc)
            if news_added:
                try:
                    observed = dt.datetime.fromisoformat(
                        str(news_added).replace("Z", "+00:00")
                    ).astimezone(dt.timezone.utc)
                except ValueError:
                    pass
            name = f"{el.get('first_name','')} {el.get('second_name','')}".strip()
            records.append(
                AvailabilityRecord(
                    sport="football",
                    player_name=name,
                    team_name=teams.get(el.get("team"), ""),
                    observed_at=observed,
                    status=status,
                    reason=news,
                    chance_of_playing=float(chance) / 100.0 if chance is not None else None,
                    position=positions.get(el.get("element_type"), ""),
                    external_ids={"fpl_element": el.get("id")},
                    source_key=self.key,
                    retrieved_at=res.fetched_at,
                )
            )

        flagged = sum(1 for r in records if r.status not in {"available", "unknown"})
        return SourceResponse(
            source_key=self.key, ok=True, records=records, urls=[url],
            retrieved_at=res.fetched_at, from_cache=res.from_cache,
            detail={"players": len(records), "flagged": flagged},
        )

    def fetch_fixtures(self, *, competition_key: str, sport: str = "football") -> SourceResponse:
        if competition_key not in self.COMPETITIONS:
            return SourceResponse.failure(self.key, f"{competition_key} not covered")
        payload, burl, err, _ = self._bootstrap()
        if payload is None:
            return SourceResponse.failure(self.key, f"team list unavailable: {err}", [burl])
        teams = {t["id"]: t["name"] for t in payload.get("teams", [])}

        url = f"{BASE}/fixtures/"
        res = self.fetch(url, ttl=1800)
        if not res.ok:
            return SourceResponse.failure(self.key, f"fixtures unavailable: {res.error}", [url])
        try:
            fixtures = res.json()
        except ValueError as exc:
            return SourceResponse.failure(self.key, f"malformed fixtures payload: {exc}", [url])

        season = _season_label_from(payload)
        records: list[MatchRecord] = []
        for fx in fixtures:
            kickoff_raw = fx.get("kickoff_time")
            if not kickoff_raw:
                continue
            try:
                kickoff = dt.datetime.fromisoformat(
                    str(kickoff_raw).replace("Z", "+00:00")
                ).astimezone(dt.timezone.utc)
            except ValueError:
                continue
            home, away = teams.get(fx.get("team_h"), ""), teams.get(fx.get("team_a"), "")
            if not home or not away:
                continue
            finished = bool(fx.get("finished"))
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
                    status="played" if finished else "scheduled",
                    ft_home_goals=fx.get("team_h_score") if finished else None,
                    ft_away_goals=fx.get("team_a_score") if finished else None,
                    matchday=fx.get("event"),
                    external_ids={"fpl_fixture": fx.get("id")},
                    source_key=self.key,
                    retrieved_at=res.fetched_at,
                )
            )
        return SourceResponse(
            source_key=self.key, ok=True, records=records, urls=[url],
            retrieved_at=res.fetched_at, from_cache=res.from_cache,
            detail={"fixtures": len(records)},
        )


def _season_label_from(payload: dict) -> str:
    events = payload.get("events") or []
    for ev in events:
        deadline = ev.get("deadline_time")
        if deadline:
            try:
                d = dt.datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
                start = d.year if d.month >= 7 else d.year - 1
                return f"{start}-{str(start + 1)[-2:]}"
            except ValueError:
                continue
    today = dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"
