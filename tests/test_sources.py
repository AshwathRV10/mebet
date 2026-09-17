"""Data acquisition behaviour.

The rule under test throughout: an unreachable source must produce an honest
failure, never substitute data.
"""

from __future__ import annotations

import datetime as dt

import pytest

from mebet.normalize.teams import TeamResolver, normalise_name
from mebet.sources.base import Capability, SourceAdapter, SourceResponse, SourceStatus
from mebet.sources.registry import build_sources, sources_for


class UnreachableSource(SourceAdapter):
    key = "unreachable_test"
    name = "Deliberately unreachable"
    capabilities = (Capability.MATCH_RESULTS,)

    def status(self) -> SourceStatus:
        return SourceStatus(key=self.key, available=False, detail="blocked")

    def fetch_matches(self, *, competition_key, season, sport="football"):
        return SourceResponse.failure(self.key, "network unreachable")


def test_failure_carries_no_records():
    response = UnreachableSource().fetch_matches(competition_key="ENG.1", season="2024-25")
    assert not response.ok
    assert response.records == []
    assert "unreachable" in response.error


def test_unsupported_capability_is_explicit():
    response = UnreachableSource().fetch_availability(competition_key="ENG.1")
    assert not response.ok
    assert "does not provide" in response.error


def test_registry_orders_by_reliability():
    ordered = sources_for(Capability.MATCH_STATS)
    reliabilities = [s.reliability for s in ordered]
    assert reliabilities == sorted(reliabilities, reverse=True)


def test_all_registered_sources_declare_metadata():
    for source in build_sources():
        described = source.describe()
        assert described["key"] and described["name"]
        assert described["capabilities"], f"{source.key} declares no capabilities"
        assert 0.0 <= described["reliability"] <= 1.0


# --- team resolution -------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("Man United", "Man Utd"),
    ("Tottenham", "Spurs"),
    ("Nott'm Forest", "Nottingham Forest"),
    ("Wolves", "Wolverhampton Wanderers"),
    ("Ath Madrid", "Atletico Madrid"),
    ("M'gladbach", "Borussia Monchengladbach"),
    ("Paris SG", "Paris Saint-Germain"),
    ("Inter", "Internazionale"),
    ("Brighton", "Brighton & Hove Albion"),
    ("Man City", "Manchester City"),
])
def test_equivalent_names_resolve_to_one_club(session, a, b):
    resolver = TeamResolver(session)
    assert resolver.resolve(a).id == resolver.resolve(b).id


@pytest.mark.parametrize("a,b", [
    ("Real Madrid", "Real Sociedad"),
    ("Manchester City", "Manchester United"),
    ("Sheffield United", "Sheffield Wednesday"),
    ("Paris Saint-Germain", "Paris FC"),
    ("Alpha FC", "Beta FC"),
])
def test_distinct_clubs_stay_distinct(session, a, b):
    resolver = TeamResolver(session)
    assert resolver.resolve(a).id != resolver.resolve(b).id


def test_normalisation_strips_noise_but_keeps_identity():
    assert normalise_name("Arsenal FC") == normalise_name("Arsenal")
    assert normalise_name("Bayern München") == normalise_name("Bayern Munchen")
    assert normalise_name("A.C. Milan") == normalise_name("AC Milan")
    assert normalise_name("Real Madrid") != normalise_name("Real Betis")


def test_resolver_can_refuse_to_create(session):
    resolver = TeamResolver(session)
    assert resolver.resolve("Entirely Unknown Club", create=False) is None
    assert "Entirely Unknown Club" in resolver.unresolved
