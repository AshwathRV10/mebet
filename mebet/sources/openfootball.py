"""Adapter: openfootball/football.json.

Broad competition coverage (including cups and competitions the statistical
feeds omit) but results only - no shots, corners or cards. It is therefore
registered as a fixtures/results source and used to widen competition
coverage, not to feed the statistical models.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from ..logging_setup import get_logger
from ..normalize import MatchRecord
from .base import Capability, SourceAdapter, SourceResponse, SourceStatus
from .registry import register

log = get_logger("sources.openfootball")

BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

#: competition key -> (file stem, display name, country)
COMPETITIONS: dict[str, tuple[str, str, str]] = {
    "ENG.1": ("en.1", "Premier League", "England"),
    "ENG.2": ("en.2", "Championship", "England"),
    "ESP.1": ("es.1", "La Liga", "Spain"),
    "ITA.1": ("it.1", "Serie A", "Italy"),
    "GER.1": ("de.1", "Bundesliga", "Germany"),
    "FRA.1": ("fr.1", "Ligue 1", "France"),
    "AUT.1": ("at.1", "Austrian Bundesliga", "Austria"),
    "CL": ("cl", "UEFA Champions League", "Europe"),
}


@register
class OpenFootballAdapter(SourceAdapter):
    key = "openfootball"
    name = "openfootball / football.json"
    homepage = "https://github.com/openfootball/football.json"
    license = "Public domain"
    capabilities = (Capability.MATCH_RESULTS, Capability.FIXTURES)
    sports = ("football",)
    # Results are reliable; the feed is volunteer-updated and carries no stats.
    reliability = 0.6
    min_interval = 0.4

    def status(self) -> SourceStatus:
        res = self.fetch(f"{BASE}/2024-25/en.1.json", ttl=86400)
        return SourceStatus(
            key=self.key, available=res.ok,
            detail="reachable" if res.ok else f"unreachable: {res.error}",
        )

    def fetch_matches(self, *, competition_key: str, season: str,
                      sport: str = "football") -> SourceResponse:
        if competition_key not in COMPETITIONS:
            return SourceResponse.failure(self.key, f"competition {competition_key} not carried")
        stem, comp_name, country = COMPETITIONS[competition_key]
        url = f"{BASE}/{season}/{stem}.json"
        res = self.fetch(url, ttl=6 * 3600)
        if not res.ok:
            return SourceResponse.failure(self.key, f"could not retrieve {url}: {res.error}", [url])
        try:
            payload = res.json()
        except ValueError as exc:
            return SourceResponse.failure(self.key, f"malformed JSON: {exc}", [url])

        records: list[MatchRecord] = []
        for m in payload.get("matches", []):
            try:
                date = dt.date.fromisoformat(m["date"])
            except (KeyError, ValueError):
                continue
            home = (m.get("team1") or "").strip()
            away = (m.get("team2") or "").strip()
            if not home or not away:
                continue
            # Older files in this feed use a list for "score"; newer ones a
            # mapping. Anything unexpected is treated as "no score recorded".
            score = m.get("score")
            if not isinstance(score, dict):
                score = {}
            ft = score.get("ft") if isinstance(score.get("ft"), list) else []
            ht = score.get("ht") if isinstance(score.get("ht"), list) else []
            played = len(ft) == 2 and all(isinstance(v, int) for v in ft)
            kickoff, exact = None, False
            if m.get("time"):
                try:
                    hh, mm = str(m["time"]).split(":")[:2]
                    kickoff = dt.datetime(date.year, date.month, date.day, int(hh), int(mm),
                                          tzinfo=dt.timezone.utc)
                    exact = True
                except ValueError:
                    pass
            records.append(
                MatchRecord(
                    sport="football",
                    competition_key=competition_key,
                    competition_name=comp_name,
                    season_label=season,
                    home_team=home,
                    away_team=away,
                    kickoff_date=date,
                    kickoff_utc=kickoff,
                    kickoff_is_exact=exact,
                    status="played" if played else "scheduled",
                    ft_home_goals=ft[0] if played else None,
                    ft_away_goals=ft[1] if played else None,
                    ht_home_goals=ht[0] if len(ht) == 2 and all(isinstance(v, int) for v in ht) else None,
                    ht_away_goals=ht[1] if len(ht) == 2 and all(isinstance(v, int) for v in ht) else None,
                    stage=(m.get("round") or ""),
                    external_ids={"country": country},
                    source_key=self.key,
                    retrieved_at=res.fetched_at,
                )
            )
        return SourceResponse(
            source_key=self.key, ok=True, records=records, urls=[url],
            retrieved_at=res.fetched_at, from_cache=res.from_cache,
            detail={"competition": competition_key, "season": season},
        )

    def fetch_fixtures(self, *, competition_key: str, sport: str = "football",
                       season: Optional[str] = None) -> SourceResponse:
        season = season or _current_season()
        return self.fetch_matches(competition_key=competition_key, season=season, sport=sport)


def _current_season(today: Optional[dt.date] = None) -> str:
    today = today or dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"
