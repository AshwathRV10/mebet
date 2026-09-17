"""Canonical internal records.

Every adapter converts its provider-specific payload into these types, so
downstream layers never know or care which source a fact came from - only
that it is tagged with one.

``None`` means *not observed*. It is never a stand-in for zero.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TeamStatLine:
    """One team's measured output in a single match."""

    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    corners: Optional[int] = None
    fouls: Optional[int] = None
    yellow_cards: Optional[int] = None
    red_cards: Optional[int] = None
    possession: Optional[float] = None
    xg: Optional[float] = None
    xga: Optional[float] = None

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in self.__dataclass_fields__)


@dataclass
class MatchRecord:
    sport: str
    competition_key: str
    competition_name: str
    season_label: str
    home_team: str
    away_team: str
    kickoff_date: dt.date
    kickoff_utc: Optional[dt.datetime] = None
    kickoff_is_exact: bool = False
    status: str = "scheduled"          # scheduled | played | postponed
    ft_home_goals: Optional[int] = None
    ft_away_goals: Optional[int] = None
    ht_home_goals: Optional[int] = None
    ht_away_goals: Optional[int] = None
    referee: str = ""
    stage: str = ""
    matchday: Optional[int] = None
    neutral_venue: bool = False
    home_stats: TeamStatLine = field(default_factory=TeamStatLine)
    away_stats: TeamStatLine = field(default_factory=TeamStatLine)
    external_ids: dict[str, Any] = field(default_factory=dict)
    source_key: str = ""
    retrieved_at: Optional[dt.datetime] = None


@dataclass
class PlayerMatchRecord:
    sport: str
    player_name: str
    team_name: str
    match_date: dt.date
    opponent_name: str = ""
    was_home: Optional[bool] = None
    position: str = ""
    minutes: Optional[int] = None
    started: Optional[bool] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    key_passes: Optional[int] = None
    xg: Optional[float] = None
    xa: Optional[float] = None
    xgc: Optional[float] = None
    tackles: Optional[int] = None
    saves: Optional[int] = None
    yellow_cards: Optional[int] = None
    red_cards: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)
    external_ids: dict[str, Any] = field(default_factory=dict)
    source_key: str = ""
    retrieved_at: Optional[dt.datetime] = None


@dataclass
class AvailabilityRecord:
    sport: str
    player_name: str
    team_name: str
    observed_at: dt.datetime
    status: str = "unknown"            # available|doubtful|injured|suspended|unavailable|unknown
    reason: str = ""
    chance_of_playing: Optional[float] = None
    expected_return: Optional[dt.date] = None
    position: str = ""
    external_ids: dict[str, Any] = field(default_factory=dict)
    source_key: str = ""
    retrieved_at: Optional[dt.datetime] = None


@dataclass
class LineupRecord:
    sport: str
    team_name: str
    player_name: str
    kind: str = "predicted"            # predicted | confirmed
    is_starter: bool = True
    confidence: Optional[float] = None
    source_key: str = ""
    retrieved_at: Optional[dt.datetime] = None


@dataclass
class ConditionsRecord:
    temperature_c: Optional[float] = None
    wind_kph: Optional[float] = None
    precipitation_mm: Optional[float] = None
    humidity_pct: Optional[float] = None
    description: str = ""
    surface: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    source_key: str = ""
    retrieved_at: Optional[dt.datetime] = None
