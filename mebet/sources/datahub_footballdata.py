"""Adapter: datahub.io ``football-datasets``.

Match-level data for the five major European leagues, seasons 1993-94 to the
current one, regenerated daily by CI from football-data.co.uk and published
under the Open Data Commons PDDL.

Each row carries full-time and half-time scores, referee, shots, shots on
target, fouls, corners and cards for both sides - which is what makes the
goals, corners and cards models possible.

Worth noting for the no-odds requirement: this feed's upstream processing
already drops every bookmaker column, so odds are structurally absent rather
than merely filtered. ``strip_odds_fields`` still runs, as a second barrier.
"""

from __future__ import annotations

import datetime as dt

from ..logging_setup import get_logger
from ..normalize.footballdata_csv import parse_matches
from .base import Capability, SourceAdapter, SourceResponse, SourceStatus
from .registry import register

log = get_logger("sources.datahub")

BASE = "https://raw.githubusercontent.com/datasets/football-datasets/master/datasets"

#: competition key -> (path segment, display name, country, tier)
COMPETITIONS: dict[str, tuple[str, str, str, int]] = {
    "ENG.1": ("premier-league", "Premier League", "England", 1),
    "ESP.1": ("la-liga", "La Liga", "Spain", 1),
    "ITA.1": ("serie-a", "Serie A", "Italy", 1),
    "GER.1": ("bundesliga", "Bundesliga", "Germany", 1),
    "FRA.1": ("ligue-1", "Ligue 1", "France", 1),
}


def season_to_code(season: str) -> str:
    """'2024-25' -> '2425'."""
    start, _, end = season.partition("-")
    if not end:
        raise ValueError(f"season must look like '2024-25', got {season!r}")
    return f"{start[-2:]}{end[-2:]}"



@register
class DatahubFootballDataAdapter(SourceAdapter):
    key = "datahub_footballdata"
    name = "datahub.io football-datasets (from football-data.co.uk)"
    homepage = "https://github.com/datasets/football-datasets"
    license = "ODC PDDL 1.0 (public domain dedication)"
    capabilities = (Capability.MATCH_RESULTS, Capability.MATCH_STATS)
    # A daily-regenerated mirror of a long-established feed: high confidence in
    # the values, one step removed from the publisher.
    reliability = 0.85
    min_interval = 0.4

    def status(self) -> SourceStatus:
        probe = self.fetch(f"{BASE}/premier-league/season-2425.csv", ttl=86400)
        return SourceStatus(
            key=self.key,
            available=probe.ok,
            detail="reachable" if probe.ok else f"unreachable: {probe.error}",
        )

    def url_for(self, competition_key: str, season: str) -> str:
        if competition_key not in COMPETITIONS:
            raise KeyError(competition_key)
        path = COMPETITIONS[competition_key][0]
        return f"{BASE}/{path}/season-{season_to_code(season)}.csv"

    def fetch_matches(self, *, competition_key: str, season: str,
                      sport: str = "football") -> SourceResponse:
        if sport != "football":
            return SourceResponse.failure(self.key, f"sport {sport} not supported")
        if competition_key not in COMPETITIONS:
            return SourceResponse.failure(
                self.key,
                f"competition {competition_key} not carried by this source "
                f"(available: {', '.join(sorted(COMPETITIONS))})",
            )
        url = self.url_for(competition_key, season)
        # Completed seasons never change; the current one is refreshed daily.
        res = self.fetch(url, ttl=6 * 3600)
        if not res.ok:
            return SourceResponse.failure(
                self.key, f"could not retrieve {competition_key} {season}: {res.error}", [url]
            )

        _, comp_name, country, _tier = COMPETITIONS[competition_key]
        records, detail = parse_matches(
            res.text,
            competition_key=competition_key,
            competition_name=comp_name,
            season_label=season,
            source_key=self.key,
            retrieved_at=res.fetched_at,
            country=country,
        )
        detail.update({"competition": competition_key, "season": season})

        return SourceResponse(
            source_key=self.key,
            ok=True,
            records=records,
            urls=[url],
            retrieved_at=res.fetched_at,
            from_cache=res.from_cache,
            detail=detail,
        )
