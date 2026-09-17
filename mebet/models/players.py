"""Player-level prediction.

Only statistics the data actually supports are predicted. The reachable
player feed carries minutes, goals, assists, expected goals and expected
assists - so goals, assists and involvement are modelled, while shots and key
passes are reported as unavailable rather than being back-derived from
expected goals, which would be a guess dressed up as a measurement.

Method: per-90 rates from a player's own history, blended with their expected
goals/assists (a more stable signal than goals over short samples), scaled by
predicted minutes and by the team's expected goals for this specific match,
then converted to probabilities with a Poisson tail.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import poisson

from ..db.repository import AsOfRepository
from ..logging_setup import get_logger

log = get_logger("models.players")

#: Minimum minutes before a per-90 rate is considered meaningful.
MIN_MINUTES = 270
#: Recency window for form.
RECENT_MATCHES = 8


@dataclass
class PlayerProjection:
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    position: str = ""
    expected_minutes: Optional[float] = None
    start_probability: Optional[float] = None
    expected_goals: Optional[float] = None
    expected_assists: Optional[float] = None
    prob_scores: Optional[float] = None
    prob_two_plus: Optional[float] = None
    prob_assists: Optional[float] = None
    prob_involvement: Optional[float] = None
    availability_status: str = "unknown"
    availability_note: str = ""
    sample_minutes: int = 0
    sample_matches: int = 0
    sufficient_data: bool = True
    note: str = ""
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "player": self.player_name,
            "team": self.team_name,
            "position": self.position,
            "expected_minutes": self.expected_minutes,
            "start_probability": self.start_probability,
            "expected_goals": self.expected_goals,
            "expected_assists": self.expected_assists,
            "prob_scores": self.prob_scores,
            "prob_two_plus": self.prob_two_plus,
            "prob_assists": self.prob_assists,
            "prob_goal_or_assist": self.prob_involvement,
            "availability": self.availability_status,
            "availability_note": self.availability_note,
            "sample_minutes": self.sample_minutes,
            "sample_matches": self.sample_matches,
            "sufficient_data": self.sufficient_data,
            "note": self.note,
            "evidence": self.evidence,
        }


class PlayerModel:
    """Not a ``PredictionModel``: it produces per-player projections rather
    than a match-outcome distribution, so it has its own interface."""

    key = "player_rates"
    name = "Player per-90 rates blended with expected goals/assists"

    def __init__(self, min_minutes: int = MIN_MINUTES,
                 xg_blend: float = 0.6, recent_matches: int = RECENT_MATCHES,
                 lookback_days: int = 400) -> None:
        self.lookback_days = lookback_days
        # Weight on expected goals versus actual goals. Expected goals is the
        # steadier estimator over the sample sizes available within a season.
        self.min_minutes = min_minutes
        self.xg_blend = xg_blend
        self.recent_matches = recent_matches

    def project_team(
        self,
        repo: AsOfRepository,
        team_id: int,
        *,
        team_expected_goals: Optional[float],
        availability: Optional[dict[int, dict]] = None,
        limit: int = 14,
    ) -> list[PlayerProjection]:
        team = repo.team(team_id)
        team_name = team.canonical_name if team else f"team:{team_id}"
        # A rolling window rather than season-to-date: in August a
        # season-to-date window holds a match or two, which would report every
        # player as "insufficient data" when a full season of evidence exists
        # just behind the season boundary.
        history = repo.team_player_history(
            team_id, since=repo.as_of.date() - dt.timedelta(days=self.lookback_days)
        )
        if not history:
            return []

        by_player: dict[int, list] = {}
        for row in history:
            by_player.setdefault(row.player_id, []).append(row)

        projections: list[PlayerProjection] = []
        for player_id, rows in by_player.items():
            rows.sort(key=lambda r: r.match_date, reverse=True)
            proj = self._project_player(repo, player_id, rows, team_id, team_name,
                                        team_expected_goals, availability)
            if proj is not None:
                projections.append(proj)

        # Most likely to matter first: expected involvement, then minutes.
        projections.sort(
            key=lambda p: (
                (p.expected_goals or 0) + (p.expected_assists or 0),
                p.expected_minutes or 0,
            ),
            reverse=True,
        )
        return projections[:limit]

    def _project_player(self, repo, player_id, rows, team_id, team_name,
                        team_expected_goals, availability) -> Optional[PlayerProjection]:
        from ..db.models import Player

        player = repo.session.get(Player, player_id)
        if player is None:
            return None

        total_minutes = sum(r.minutes or 0 for r in rows)
        appearances = sum(1 for r in rows if (r.minutes or 0) > 0)
        proj = PlayerProjection(
            player_id=player_id,
            player_name=player.full_name,
            team_id=team_id,
            team_name=team_name,
            position=player.position or "",
            sample_minutes=total_minutes,
            sample_matches=appearances,
        )

        avail = (availability or {}).get(player_id)
        if avail:
            proj.availability_status = avail.get("status", "unknown")
            proj.availability_note = avail.get("reason", "")

        if proj.availability_status in {"injured", "suspended", "unavailable"}:
            proj.sufficient_data = True
            proj.expected_minutes = 0.0
            proj.start_probability = 0.0
            proj.expected_goals = 0.0
            proj.expected_assists = 0.0
            proj.prob_scores = 0.0
            proj.prob_assists = 0.0
            proj.prob_involvement = 0.0
            proj.note = f"ruled out: {proj.availability_note or proj.availability_status}"
            return proj

        if total_minutes < self.min_minutes:
            proj.sufficient_data = False
            proj.note = (
                f"only {total_minutes} minutes recorded; below the {self.min_minutes}-minute "
                "threshold for a stable per-90 rate"
            )
            return proj

        recent = rows[: self.recent_matches]
        recent_minutes = [r.minutes or 0 for r in recent]
        expected_minutes = float(np.mean(recent_minutes)) if recent_minutes else 0.0
        starts = [r.started for r in recent if r.started is not None]
        start_prob = (sum(1 for s in starts if s) / len(starts)) if starts else None

        # A doubt scales expected minutes by the published chance of playing.
        chance = (avail or {}).get("chance_of_playing")
        if proj.availability_status == "doubtful" and chance is not None:
            expected_minutes *= float(chance)
            if start_prob is not None:
                start_prob *= float(chance)

        goals = sum(r.goals or 0 for r in rows)
        assists = sum(r.assists or 0 for r in rows)
        xg_values = [r.xg for r in rows if r.xg is not None]
        xa_values = [r.xa for r in rows if r.xa is not None]

        per90 = 90.0 / total_minutes
        goals_per90 = goals * per90
        assists_per90 = assists * per90

        if xg_values:
            xg_per90 = sum(xg_values) * per90
            goals_rate = self.xg_blend * xg_per90 + (1 - self.xg_blend) * goals_per90
            xg_used = True
        else:
            goals_rate = goals_per90
            xg_used = False
        if xa_values:
            xa_per90 = sum(xa_values) * per90
            assists_rate = self.xg_blend * xa_per90 + (1 - self.xg_blend) * assists_per90
        else:
            assists_rate = assists_per90

        # Scale by how many goals the team is expected to score in *this*
        # match relative to its recent scoring level, so a player facing a
        # strong defence is projected lower than the same player facing a weak one.
        scale = 1.0
        if team_expected_goals:
            recent_team_goals = _team_recent_goal_rate(repo, team_id)
            if recent_team_goals and recent_team_goals > 0.1:
                scale = float(np.clip(team_expected_goals / recent_team_goals, 0.5, 2.0))

        minutes_share = expected_minutes / 90.0
        exp_goals = goals_rate * minutes_share * scale
        exp_assists = assists_rate * minutes_share * scale

        proj.expected_minutes = round(expected_minutes, 1)
        proj.start_probability = round(start_prob, 3) if start_prob is not None else None
        proj.expected_goals = round(float(exp_goals), 3)
        proj.expected_assists = round(float(exp_assists), 3)
        proj.prob_scores = round(float(1 - poisson.pmf(0, max(exp_goals, 1e-6))), 4)
        proj.prob_two_plus = round(
            float(1 - poisson.cdf(1, max(exp_goals, 1e-6))), 4
        )
        proj.prob_assists = round(float(1 - poisson.pmf(0, max(exp_assists, 1e-6))), 4)
        proj.prob_involvement = round(
            float(1 - poisson.pmf(0, max(exp_goals + exp_assists, 1e-6))), 4
        )
        proj.evidence = {
            "goals": goals,
            "assists": assists,
            "minutes": total_minutes,
            "appearances": appearances,
            "goals_per90": round(goals_per90, 3),
            "assists_per90": round(assists_per90, 3),
            "expected_goals_available": xg_used,
            "xg_total": round(sum(xg_values), 2) if xg_values else None,
            "xa_total": round(sum(xa_values), 2) if xa_values else None,
            "team_strength_scale": round(scale, 3),
        }
        return proj


def _team_recent_goal_rate(repo: AsOfRepository, team_id: int, limit: int = 15) -> Optional[float]:
    rows = repo.team_history(team_id, limit=limit)
    goals = [r.goals_for for r in rows if r.goals_for is not None]
    return float(np.mean(goals)) if goals else None
