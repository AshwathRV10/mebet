"""Source adapter contract.

Adding a provider means implementing ``SourceAdapter`` and registering it.
Nothing else in the system changes - the engine asks the registry for the
sources that can supply a capability and works with whatever answers.

Adapters must obey one rule above all: **if data could not be retrieved, say
so.** ``SourceResponse.ok=False`` with an ``error`` is always correct;
returning plausible-looking records is never correct.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

from ..logging_setup import get_logger
from .http import FetchResult, HttpClient

log = get_logger("sources.base")


class Capability(str, Enum):
    """What a source can supply."""

    MATCH_RESULTS = "match_results"      # historical results
    MATCH_STATS = "match_stats"          # shots/corners/cards etc. per match
    FIXTURES = "fixtures"                # upcoming matches and kickoff times
    PLAYER_STATS = "player_stats"
    AVAILABILITY = "availability"        # injuries, suspensions, doubts
    LINEUPS = "lineups"
    WEATHER = "weather"
    STANDINGS = "standings"


@dataclass
class SourceStatus:
    key: str
    available: bool
    detail: str = ""
    checked_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    requires_credential: bool = False
    credential_present: bool = True


@dataclass
class SourceResponse:
    """Result of one adapter call. Carries provenance, including failures."""

    source_key: str
    ok: bool
    records: list[Any] = field(default_factory=list)
    error: str = ""
    urls: list[str] = field(default_factory=list)
    retrieved_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    from_cache: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, key: str, error: str, urls: Optional[Sequence[str]] = None) -> "SourceResponse":
        return cls(source_key=key, ok=False, error=error, urls=list(urls or []))

    def __len__(self) -> int:
        return len(self.records)


class SourceAdapter(ABC):
    """Base class for all data sources."""

    #: Stable identifier used in the database and the UI.
    key: str = ""
    name: str = ""
    homepage: str = ""
    license: str = ""
    sports: tuple[str, ...] = ("football",)
    capabilities: tuple[Capability, ...] = ()
    #: 0-1 prior on trustworthiness: official/structured feeds score higher
    #: than community mirrors, which score higher than best-effort scrapes.
    #: Used to break ties when sources disagree - never to invent precision.
    reliability: float = 0.5
    #: Seconds between requests to this provider.
    min_interval: float = 1.0
    requires_credential: bool = False
    #: Set when a provider publishes bulk data and robots need not be polled
    #: per request (e.g. raw file hosting), as opposed to page scraping.
    scrapes_html: bool = False

    def __init__(self, client: Optional[HttpClient] = None) -> None:
        self.client = client or HttpClient(min_interval=self.min_interval)

    # -- capability plumbing ---------------------------------------------
    def supports(self, capability: Capability, sport: str = "football") -> bool:
        return capability in self.capabilities and sport in self.sports

    @abstractmethod
    def status(self) -> SourceStatus:
        """Cheap check of whether this source is usable right now."""

    def fetch(self, url: str, **kwargs) -> FetchResult:
        kwargs.setdefault("check_robots", self.scrapes_html)
        return self.client.get(url, **kwargs)

    # -- optional capability methods -------------------------------------
    # Defaults return an explicit "not supported" rather than empty success,
    # so a missing capability can never be mistaken for "nothing to report".
    def _unsupported(self, what: str) -> SourceResponse:
        return SourceResponse.failure(self.key, f"{self.key} does not provide {what}")

    def fetch_matches(self, *, competition_key: str, season: str, sport: str = "football") -> SourceResponse:
        return self._unsupported("match results")

    def fetch_fixtures(self, *, competition_key: str, sport: str = "football") -> SourceResponse:
        return self._unsupported("fixtures")

    def fetch_player_stats(self, *, competition_key: str, season: str, sport: str = "football") -> SourceResponse:
        return self._unsupported("player statistics")

    def fetch_availability(self, *, competition_key: str, sport: str = "football") -> SourceResponse:
        return self._unsupported("availability/injury data")

    def fetch_lineups(self, *, match_external_id: str, sport: str = "football") -> SourceResponse:
        return self._unsupported("lineups")

    def fetch_weather(self, *, latitude: float, longitude: float, when: dt.datetime) -> SourceResponse:
        return self._unsupported("weather")

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "homepage": self.homepage,
            "license": self.license,
            "reliability": self.reliability,
            "capabilities": [c.value for c in self.capabilities],
            "sports": list(self.sports),
            "requires_credential": self.requires_credential,
        }
