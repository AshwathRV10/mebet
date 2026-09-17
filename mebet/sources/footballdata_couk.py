"""Adapter: football-data.co.uk, the original publisher.

Same columns as the datahub mirror but fresher (updated twice weekly during
the season) and covering more competitions. The published files also carry
bookmaker columns; ``parse_matches`` strips them before any record is built,
so odds never enter the system.

Note: this host is not reachable from every environment. When it is blocked,
``status()`` reports that plainly and the ingestion layer falls back to the
mirror rather than proceeding with nothing.
"""

from __future__ import annotations

from ..logging_setup import get_logger
from ..normalize.footballdata_csv import parse_matches
from .base import Capability, SourceAdapter, SourceResponse, SourceStatus
from .registry import register

log = get_logger("sources.footballdata_couk")

BASE = "https://www.football-data.co.uk/mmz4281"

#: competition key -> (division code, display name, country, tier)
COMPETITIONS: dict[str, tuple[str, str, str, int]] = {
    "ENG.1": ("E0", "Premier League", "England", 1),
    "ENG.2": ("E1", "Championship", "England", 2),
    "ENG.3": ("E2", "League One", "England", 3),
    "ENG.4": ("E3", "League Two", "England", 4),
    "SCO.1": ("SC0", "Scottish Premiership", "Scotland", 1),
    "GER.1": ("D1", "Bundesliga", "Germany", 1),
    "GER.2": ("D2", "2. Bundesliga", "Germany", 2),
    "ITA.1": ("I1", "Serie A", "Italy", 1),
    "ITA.2": ("I2", "Serie B", "Italy", 2),
    "ESP.1": ("SP1", "La Liga", "Spain", 1),
    "ESP.2": ("SP2", "La Liga 2", "Spain", 2),
    "FRA.1": ("F1", "Ligue 1", "France", 1),
    "FRA.2": ("F2", "Ligue 2", "France", 2),
    "NED.1": ("N1", "Eredivisie", "Netherlands", 1),
    "BEL.1": ("B1", "Belgian Pro League", "Belgium", 1),
    "POR.1": ("P1", "Primeira Liga", "Portugal", 1),
    "TUR.1": ("T1", "Super Lig", "Turkey", 1),
    "GRE.1": ("G1", "Super League Greece", "Greece", 1),
}


def season_to_code(season: str) -> str:
    start, _, end = season.partition("-")
    if not end:
        raise ValueError(f"season must look like '2024-25', got {season!r}")
    return f"{start[-2:]}{end[-2:]}"


@register
class FootballDataCoUkAdapter(SourceAdapter):
    key = "footballdata_couk"
    name = "football-data.co.uk"
    homepage = "https://www.football-data.co.uk/data.php"
    license = "Free for personal use; see publisher's terms"
    capabilities = (Capability.MATCH_RESULTS, Capability.MATCH_STATS)
    # The publisher itself, so ranked above any mirror of it.
    reliability = 0.9
    min_interval = 2.0

    def status(self) -> SourceStatus:
        probe = self.fetch(f"{BASE}/2425/E0.csv", ttl=86400)
        return SourceStatus(
            key=self.key,
            available=probe.ok,
            detail="reachable" if probe.ok else f"unreachable: {probe.error}",
        )

    def url_for(self, competition_key: str, season: str) -> str:
        code = COMPETITIONS[competition_key][0]
        return f"{BASE}/{season_to_code(season)}/{code}.csv"

    def fetch_matches(self, *, competition_key: str, season: str,
                      sport: str = "football") -> SourceResponse:
        if sport != "football":
            return SourceResponse.failure(self.key, f"sport {sport} not supported")
        if competition_key not in COMPETITIONS:
            return SourceResponse.failure(
                self.key, f"competition {competition_key} not carried by this source"
            )
        url = self.url_for(competition_key, season)
        res = self.fetch(url, ttl=6 * 3600)
        if not res.ok:
            return SourceResponse.failure(
                self.key, f"could not retrieve {competition_key} {season}: {res.error}", [url]
            )
        _, comp_name, country, _ = COMPETITIONS[competition_key]
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
            source_key=self.key, ok=True, records=records, urls=[url],
            retrieved_at=res.fetched_at, from_cache=res.from_cache, detail=detail,
        )
