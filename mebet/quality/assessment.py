"""Data quality assessment.

Separating four things the brief insists are not the same:

*   **Prediction** - the selection the model favours.
*   **Probability** - the model's estimate of how often that happens.
*   **Confidence** - how much the evidence behind that estimate can bear.
*   **Data quality** - what was actually retrieved, how fresh, how complete.

Probability comes from the models. Confidence and data quality come from
here, and low quality reduces confidence without silently distorting the
probabilities: a probability that has been quietly shaded toward 50% is no
longer a statement about the world.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QualityTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class QualityIssue:
    code: str
    severity: str           # info | warning | serious
    message: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "message": self.message,
                "detail": self.detail}


@dataclass
class DataQualityReport:
    tier: QualityTier = QualityTier.MEDIUM
    score: float = 0.0
    sources_used: list[str] = field(default_factory=list)
    sources_failed: list[dict] = field(default_factory=list)
    historical_matches: int = 0
    home_matches: int = 0
    away_matches: int = 0
    h2h_matches: int = 0
    player_data: bool = False
    availability_data: bool = False
    lineups: str = "unavailable"        # confirmed | predicted | unavailable
    weather: bool = False
    stats_coverage: dict = field(default_factory=dict)
    freshness_days: Optional[float] = None
    issues: list[QualityIssue] = field(default_factory=list)
    #: Multiplier applied to confidence, never to probabilities.
    confidence_multiplier: float = 1.0

    def add(self, code: str, severity: str, message: str, **detail) -> None:
        self.issues.append(QualityIssue(code, severity, message, detail))

    def summary_lines(self) -> list[str]:
        lines = [
            f"Data quality: {self.tier.value}",
            f"Sources used: {len(self.sources_used)} ({', '.join(self.sources_used) or 'none'})",
            f"Historical matches analysed: {self.historical_matches}",
        ]
        if self.h2h_matches:
            lines.append(f"Head-to-head meetings: {self.h2h_matches}")
        lines.append(f"Lineups: {self.lineups}")
        lines.append(
            "Team news: available" if self.availability_data else "Team news: unavailable"
        )
        return lines

    def as_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "score": round(self.score, 3),
            "confidence_multiplier": round(self.confidence_multiplier, 3),
            "sources_used": self.sources_used,
            "sources_failed": self.sources_failed,
            "historical_matches": self.historical_matches,
            "home_matches": self.home_matches,
            "away_matches": self.away_matches,
            "h2h_matches": self.h2h_matches,
            "player_data": self.player_data,
            "availability_data": self.availability_data,
            "lineups": self.lineups,
            "weather": self.weather,
            "stats_coverage": self.stats_coverage,
            "freshness_days": self.freshness_days,
            "issues": [i.as_dict() for i in self.issues],
            "summary": self.summary_lines(),
        }
