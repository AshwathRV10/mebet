"""Explanations generated from the data, not about it.

The rule that shapes this module: a factor may only be emitted if the numbers
behind it were actually retrieved, and those numbers travel with it in
``evidence``. There is no template that fires on the model's conclusion - each
factor is computed from a specific comparison, and if the inputs to that
comparison are missing, the factor is simply not produced.

That is why the explanations can be trusted as a description of *why* the
model said what it said, rather than a plausible story told afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..features.football import MatchFeatures, TeamForm
from ..logging_setup import get_logger

log = get_logger("engine.explain")

#: A difference must exceed this fraction of the league average before it is
#: described as meaningful, so noise is not narrated as insight.
MATERIAL_FRACTION = 0.12
#: Minimum matches behind each side of a comparison.
MIN_SAMPLE = 6


@dataclass
class Factor:
    category: str
    favours: str            # home | away | over | under | neutral
    statement: str
    impact: float           # 0-1, relative magnitude
    evidence: dict = field(default_factory=dict)
    market: str = "1x2"

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "favours": self.favours,
            "statement": self.statement,
            "impact": round(self.impact, 3),
            "evidence": self.evidence,
            "market": self.market,
        }


def _usable(form: TeamForm, window: str, stat: str, min_n: int = MIN_SAMPLE) -> Optional[float]:
    entry = form.windows.get(window, {}).get(stat)
    if entry and entry.is_usable and entry.n >= min_n:
        return entry.value
    return None


def _relative(a: float, b: float) -> float:
    """Signed relative gap between two values, capped for scoring purposes."""
    denominator = max(abs(a) + abs(b), 1e-6) / 2
    return (a - b) / denominator


class ExplanationEngine:
    def __init__(self, features: MatchFeatures) -> None:
        self.features = features
        self.home = features.home
        self.away = features.away

    def build(self, *, availability: Optional[dict] = None,
              model_diagnostics: Optional[dict] = None) -> list[Factor]:
        factors: list[Factor] = []
        for builder in (
            self._attacking_output,
            self._defensive_record,
            self._venue_split,
            self._shot_quality,
            self._recent_trend,
            self._league_position,
            self._rest_and_congestion,
            self._head_to_head,
            self._opponent_adjusted,
        ):
            try:
                factors.extend(builder())
            except Exception as exc:  # noqa: BLE001
                log.debug("explanation builder %s failed: %s", builder.__name__, exc)

        if availability:
            factors.extend(self._availability(availability))
        if model_diagnostics:
            factors.extend(self._model_internals(model_diagnostics))

        factors.sort(key=lambda f: f.impact, reverse=True)
        return factors

    # -- individual builders ----------------------------------------------
    def _attacking_output(self) -> list[Factor]:
        out = []
        for window in ("last10", "all"):
            h = _usable(self.home, window, "goals_for")
            a = _usable(self.away, window, "goals_for")
            if h is None or a is None:
                continue
            gap = _relative(h, a)
            if abs(gap) < MATERIAL_FRACTION:
                continue
            leader, trailer = (self.home, self.away) if gap > 0 else (self.away, self.home)
            hv, av = (h, a) if gap > 0 else (a, h)
            out.append(Factor(
                category="attack",
                favours="home" if gap > 0 else "away",
                statement=(
                    f"{leader.team_name} has scored more freely over the "
                    f"{'last 10 matches' if window == 'last10' else 'full recorded sample'}: "
                    f"{hv:.2f} goals per match against {trailer.team_name}'s {av:.2f}"
                ),
                impact=min(abs(gap), 1.0) * (0.9 if window == "last10" else 0.6),
                evidence={
                    "window": window,
                    "home_goals_per_match": round(h, 3),
                    "away_goals_per_match": round(a, 3),
                    "home_sample": self.home.windows[window]["goals_for"].n,
                    "away_sample": self.away.windows[window]["goals_for"].n,
                },
            ))
            break   # one attacking statement is enough; prefer the recent window
        return out

    def _defensive_record(self) -> list[Factor]:
        h = _usable(self.home, "last10", "goals_against")
        a = _usable(self.away, "last10", "goals_against")
        if h is None or a is None:
            return []
        gap = _relative(a, h)   # conceding fewer is better, so invert
        if abs(gap) < MATERIAL_FRACTION:
            return []
        better = self.home if gap > 0 else self.away
        worse = self.away if gap > 0 else self.home
        return [Factor(
            category="defence",
            favours="home" if gap > 0 else "away",
            statement=(
                f"{better.team_name} has the tighter recent defence, conceding "
                f"{(h if gap > 0 else a):.2f} per match against {worse.team_name}'s "
                f"{(a if gap > 0 else h):.2f}"
            ),
            impact=min(abs(gap), 1.0) * 0.85,
            evidence={
                "window": "last10",
                "home_conceded_per_match": round(h, 3),
                "away_conceded_per_match": round(a, 3),
                "home_sample": self.home.windows["last10"]["goals_against"].n,
                "away_sample": self.away.windows["last10"]["goals_against"].n,
            },
        )]

    def _venue_split(self) -> list[Factor]:
        out = []
        home_pts = self.home.home_splits.get("points")
        home_all = self.home.form_points.get("all")
        if home_pts and home_pts.is_usable and home_pts.n >= MIN_SAMPLE and home_all:
            delta = home_pts.value - home_all
            if abs(delta) >= 0.25:
                out.append(Factor(
                    category="venue",
                    favours="home" if delta > 0 else "away",
                    statement=(
                        f"{self.home.team_name} {'is markedly stronger' if delta > 0 else 'underperforms'} "
                        f"at home: {home_pts.value:.2f} points per home match against "
                        f"{home_all:.2f} overall, over {home_pts.n} home matches"
                    ),
                    impact=min(abs(delta) / 1.5, 1.0) * 0.8,
                    evidence={"home_ppg": round(home_pts.value, 3),
                              "overall_ppg": round(home_all, 3), "sample": home_pts.n},
                ))

        away_pts = self.away.away_splits.get("points")
        away_all = self.away.form_points.get("all")
        if away_pts and away_pts.is_usable and away_pts.n >= MIN_SAMPLE and away_all:
            delta = away_pts.value - away_all
            if abs(delta) >= 0.25:
                out.append(Factor(
                    category="venue",
                    favours="away" if delta > 0 else "home",
                    statement=(
                        f"{self.away.team_name} {'travels well' if delta > 0 else 'is weaker on the road'}: "
                        f"{away_pts.value:.2f} points per away match against {away_all:.2f} overall, "
                        f"over {away_pts.n} away matches"
                    ),
                    impact=min(abs(delta) / 1.5, 1.0) * 0.8,
                    evidence={"away_ppg": round(away_pts.value, 3),
                              "overall_ppg": round(away_all, 3), "sample": away_pts.n},
                ))

        # The away team's defensive record specifically on the road.
        away_conceded = self.away.away_splits.get("goals_against")
        if away_conceded and away_conceded.is_usable and away_conceded.n >= MIN_SAMPLE:
            league_avg = self.features.league_baseline.get("avg_home_goals")
            if league_avg and away_conceded.value > league_avg * 1.15:
                out.append(Factor(
                    category="defence",
                    favours="home",
                    statement=(
                        f"{self.away.team_name} concedes {away_conceded.value:.2f} per away match, "
                        f"above the league's {league_avg:.2f} average for home sides"
                    ),
                    impact=min((away_conceded.value / league_avg - 1), 1.0) * 0.7,
                    evidence={"away_goals_conceded": round(away_conceded.value, 3),
                              "league_home_goals": league_avg, "sample": away_conceded.n},
                ))
        return out

    def _shot_quality(self) -> list[Factor]:
        h = _usable(self.home, "last10", "shots_on_target")
        a = _usable(self.away, "last10", "shots_on_target")
        if h is None or a is None:
            return []
        gap = _relative(h, a)
        if abs(gap) < MATERIAL_FRACTION:
            return []
        leader = self.home if gap > 0 else self.away
        return [Factor(
            category="chance_creation",
            favours="home" if gap > 0 else "away",
            statement=(
                f"{leader.team_name} is generating more on-target volume: "
                f"{max(h, a):.1f} shots on target per match against {min(h, a):.1f}"
            ),
            impact=min(abs(gap), 1.0) * 0.65,
            evidence={"home_sot": round(h, 2), "away_sot": round(a, 2), "window": "last10"},
        )]

    def _recent_trend(self) -> list[Factor]:
        out = []
        for side, form in (("home", self.home), ("away", self.away)):
            if form.trend is None or abs(form.trend) < 0.4:
                continue
            direction = "improving" if form.trend > 0 else "falling away"
            out.append(Factor(
                category="form",
                favours=side if form.trend > 0 else ("away" if side == "home" else "home"),
                statement=(
                    f"{form.team_name} is {direction}: {form.form_points['last5']:.2f} points "
                    f"per match over the last five against a longer-run {form.form_points['all']:.2f}"
                ),
                impact=min(abs(form.trend) / 1.5, 1.0) * 0.6,
                evidence={"last5_ppg": form.form_points.get("last5"),
                          "baseline_ppg": form.form_points.get("all"),
                          "trend": form.trend},
            ))
        return out

    def _league_position(self) -> list[Factor]:
        h, a = self.home.league_position, self.away.league_position
        if h is None or a is None:
            return []
        gap = a - h
        if abs(gap) < 4:
            return []
        leader = self.home if gap > 0 else self.away
        return [Factor(
            category="standing",
            favours="home" if gap > 0 else "away",
            statement=(
                f"{leader.team_name} sits {abs(gap)} places higher in the table "
                f"({min(h, a)} against {max(h, a)})"
            ),
            impact=min(abs(gap) / 15.0, 1.0) * 0.55,
            evidence={"home_position": h, "away_position": a,
                      "home_points": self.home.league_points,
                      "away_points": self.away.league_points},
        )]

    def _rest_and_congestion(self) -> list[Factor]:
        out = []
        ctx = self.features.context
        advantage = ctx.get("home_rest_advantage_days")
        if advantage is not None and abs(advantage) >= 2:
            side = "home" if advantage > 0 else "away"
            rested = self.home if advantage > 0 else self.away
            out.append(Factor(
                category="rest",
                favours=side,
                statement=(
                    f"{rested.team_name} has had {abs(advantage)} more days' rest "
                    f"({self.home.rest_days} days for {self.home.team_name} against "
                    f"{self.away.rest_days} for {self.away.team_name} since their last match)"
                ),
                impact=min(abs(advantage) / 7.0, 1.0) * 0.45,
                evidence={"home_rest_days": self.home.rest_days,
                          "away_rest_days": self.away.rest_days},
            ))
        h_cong, a_cong = ctx.get("home_congestion_14d", 0), ctx.get("away_congestion_14d", 0)
        if abs(h_cong - a_cong) >= 2:
            side = "away" if h_cong > a_cong else "home"
            busier = self.home if h_cong > a_cong else self.away
            out.append(Factor(
                category="congestion",
                favours=side,
                statement=(
                    f"{busier.team_name} has played {max(h_cong, a_cong)} matches in the last "
                    f"fortnight against {min(h_cong, a_cong)}"
                ),
                impact=min(abs(h_cong - a_cong) / 4.0, 1.0) * 0.4,
                evidence={"home_matches_14d": h_cong, "away_matches_14d": a_cong},
            ))
        return out

    def _head_to_head(self) -> list[Factor]:
        h2h = self.features.h2h
        n = h2h.get("matches", 0)
        # Below five meetings the record is anecdote, not evidence.
        if n < 5:
            return []
        wins, losses = h2h.get("home_team_wins", 0), h2h.get("away_team_wins", 0)
        if abs(wins - losses) < 2:
            return []
        leader = self.home if wins > losses else self.away
        return [Factor(
            category="head_to_head",
            favours="home" if wins > losses else "away",
            statement=(
                f"{leader.team_name} has the better recent head-to-head record: "
                f"{max(wins, losses)} wins to {min(wins, losses)} in {n} meetings "
                f"(most recent {h2h.get('most_recent')})"
            ),
            impact=min(abs(wins - losses) / n, 1.0) * 0.4,
            evidence=h2h,
        )]

    def _opponent_adjusted(self) -> list[Factor]:
        h = self.home.opponent_adjusted.get("attack_index")
        a = self.away.opponent_adjusted.get("attack_index")
        if h is None or a is None:
            return []
        if self.home.opponent_adjusted.get("attack_sample", 0) < 8:
            return []
        if self.away.opponent_adjusted.get("attack_sample", 0) < 8:
            return []
        gap = _relative(h, a)
        if abs(gap) < MATERIAL_FRACTION:
            return []
        leader = self.home if gap > 0 else self.away
        return [Factor(
            category="strength_adjusted",
            favours="home" if gap > 0 else "away",
            statement=(
                f"Adjusted for the quality of opposition faced, {leader.team_name} has the "
                f"stronger attacking record (index {max(h, a):.2f} against {min(h, a):.2f}, "
                f"where 1.00 is par)"
            ),
            impact=min(abs(gap), 1.0) * 0.75,
            evidence={"home_attack_index": h, "away_attack_index": a,
                      "home_sample": self.home.opponent_adjusted.get("attack_sample"),
                      "away_sample": self.away.opponent_adjusted.get("attack_sample")},
        )]

    def _availability(self, availability: dict) -> list[Factor]:
        out = []
        for side, key in (("home", "home"), ("away", "away")):
            entries = availability.get(key) or []
            ruled_out = [e for e in entries if e.get("status") in {"injured", "suspended", "unavailable"}]
            if not ruled_out:
                continue
            team = self.home if side == "home" else self.away
            names = ", ".join(e["player"] for e in ruled_out[:4])
            more = f" and {len(ruled_out) - 4} others" if len(ruled_out) > 4 else ""
            out.append(Factor(
                category="availability",
                favours="away" if side == "home" else "home",
                statement=(
                    f"{team.team_name} is without {len(ruled_out)} squad member(s): {names}{more}"
                ),
                impact=min(len(ruled_out) / 6.0, 1.0) * 0.7,
                evidence={"side": side, "players": ruled_out[:8]},
            ))
        return out

    def _model_internals(self, diagnostics: dict) -> list[Factor]:
        """Expose what the fitted model itself believes about the two teams."""
        out = []
        if all(k in diagnostics for k in ("home_attack", "away_defence", "home_advantage")):
            out.append(Factor(
                category="model",
                favours="neutral",
                statement=(
                    f"The fitted goal model rates the home side's attack at "
                    f"{diagnostics['home_attack']:+.2f} and the away side's defence at "
                    f"{diagnostics['away_defence']:+.2f} (log scale, 0.00 is league average), "
                    f"with a home-advantage term of {diagnostics['home_advantage']:+.2f}"
                ),
                impact=0.3,
                evidence=diagnostics,
            ))
        if "home_rating" in diagnostics and "away_rating" in diagnostics:
            out.append(Factor(
                category="model",
                favours="neutral",
                statement=(
                    f"Elo ratings stand at {diagnostics['home_rating']:.0f} against "
                    f"{diagnostics['away_rating']:.0f}, a gap of "
                    f"{diagnostics['rating_difference']:+.0f} points before home advantage"
                ),
                impact=0.28,
                evidence=diagnostics,
            ))
        return out
