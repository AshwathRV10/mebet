"""Football feature engineering.

Everything here is derived from a single ``AsOfRepository``, so no feature can
reach data that postdates the prediction. The functions take the repository
rather than a session for exactly that reason.

Design notes worth stating:

*   **Opponent adjustment.** Raw goals-per-game flatters a team that has
    played weak opposition. Each match's output is compared with what the
    opponent typically concedes, giving a strength-adjusted figure.
*   **Missing data stays missing.** If corners were never recorded for a
    league, the corner features are ``None`` and the corners model declines to
    predict, rather than predicting from zeros.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..config import get_settings
from ..db.repository import AsOfRepository, TeamMatchRow
from ..logging_setup import get_logger
from .windows import WINDOWS, WeightedStat, decay_weights, rate, weighted_mean

log = get_logger("features.football")

#: Statistics extracted per match, mapped to the attribute on ``TeamMatchRow``.
PER_MATCH_STATS = {
    "goals_for": "goals_for",
    "goals_against": "goals_against",
    "shots": "shots",
    "shots_against": "shots_against",
    "shots_on_target": "shots_on_target",
    "sot_against": "sot_against",
    "corners": "corners",
    "corners_against": "corners_against",
    "fouls": "fouls",
    "fouls_against": "fouls_against",
    "yellows": "yellows",
    "yellows_against": "yellows_against",
    "reds": "reds",
    "xg": "xg",
    "xga": "xga",
}


@dataclass
class TeamForm:
    """Everything the models may know about one team at the cutoff."""

    team_id: int
    team_name: str
    as_of: dt.datetime
    matches_available: int = 0
    # window -> stat name -> WeightedStat
    windows: dict[str, dict[str, WeightedStat]] = field(default_factory=dict)
    home_splits: dict[str, WeightedStat] = field(default_factory=dict)
    away_splits: dict[str, WeightedStat] = field(default_factory=dict)
    decayed: dict[str, WeightedStat] = field(default_factory=dict)
    opponent_adjusted: dict[str, Optional[float]] = field(default_factory=dict)
    form_points: dict[str, Optional[float]] = field(default_factory=dict)
    clean_sheet_rate: Optional[WeightedStat] = None
    scoring_rate: Optional[WeightedStat] = None
    conceding_rate: Optional[WeightedStat] = None
    btts_rate: Optional[WeightedStat] = None
    first_half_share: Optional[float] = None
    rest_days: Optional[int] = None
    matches_last_14_days: int = 0
    league_position: Optional[int] = None
    league_points: Optional[int] = None
    league_ppg: Optional[float] = None
    trend: Optional[float] = None
    missing: list[str] = field(default_factory=list)

    def stat(self, window: str, name: str) -> Optional[float]:
        w = self.windows.get(window, {}).get(name)
        return w.value if w and w.is_usable else None

    def as_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "matches_available": self.matches_available,
            "windows": {w: {k: v.as_dict() for k, v in stats.items()}
                        for w, stats in self.windows.items()},
            "home_splits": {k: v.as_dict() for k, v in self.home_splits.items()},
            "away_splits": {k: v.as_dict() for k, v in self.away_splits.items()},
            "decayed": {k: v.as_dict() for k, v in self.decayed.items()},
            "opponent_adjusted": self.opponent_adjusted,
            "form_points": self.form_points,
            "clean_sheet_rate": self.clean_sheet_rate.as_dict() if self.clean_sheet_rate else None,
            "scoring_rate": self.scoring_rate.as_dict() if self.scoring_rate else None,
            "conceding_rate": self.conceding_rate.as_dict() if self.conceding_rate else None,
            "btts_rate": self.btts_rate.as_dict() if self.btts_rate else None,
            "first_half_share": self.first_half_share,
            "rest_days": self.rest_days,
            "matches_last_14_days": self.matches_last_14_days,
            "league_position": self.league_position,
            "league_points": self.league_points,
            "league_ppg": self.league_ppg,
            "trend": self.trend,
            "missing": self.missing,
        }


@dataclass
class MatchFeatures:
    home: TeamForm
    away: TeamForm
    as_of: dt.datetime
    competition_id: int
    competition_name: str = ""
    h2h: dict = field(default_factory=dict)
    league_baseline: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "competition": self.competition_name,
            "home": self.home.as_dict(),
            "away": self.away.as_dict(),
            "h2h": self.h2h,
            "league_baseline": self.league_baseline,
            "context": self.context,
            "missing": self.missing,
        }


def _series(rows: Sequence[TeamMatchRow], attr: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for r in rows:
        v = getattr(r, attr, None)
        out.append(float(v) if v is not None else None)
    return out


def season_start_for(date: dt.date) -> dt.date:
    """1 July of the season containing ``date``."""
    year = date.year if date.month >= 7 else date.year - 1
    return dt.date(year, 7, 1)


def build_team_form(
    repo: AsOfRepository,
    team_id: int,
    *,
    competition_ids: Optional[Sequence[int]] = None,
    half_life_days: Optional[float] = None,
    max_history: int = 80,
) -> TeamForm:
    settings = get_settings()
    half_life = half_life_days if half_life_days is not None else settings.time_decay_half_life_days

    team = repo.team(team_id)
    form = TeamForm(
        team_id=team_id,
        team_name=team.canonical_name if team else f"team:{team_id}",
        as_of=repo.as_of,
    )

    history = repo.team_history(team_id, limit=max_history, competition_ids=competition_ids)
    form.matches_available = len(history)
    if not history:
        form.missing.append("no completed matches before the cutoff")
        return form

    # -- windowed averages ------------------------------------------------
    for window_name, size in WINDOWS.items():
        subset = history[:size] if size else history
        stats: dict[str, WeightedStat] = {}
        for stat_name, attr in PER_MATCH_STATS.items():
            stats[stat_name] = weighted_mean(_series(subset, attr), window=window_name)
        form.windows[window_name] = stats

    # -- time-decayed (all history, recent matches weighted more) ---------
    weights = decay_weights([r.date for r in history], repo.as_of.date(), half_life)
    for stat_name, attr in PER_MATCH_STATS.items():
        form.decayed[stat_name] = weighted_mean(_series(history, attr), weights, window="decayed")

    # -- home / away splits ------------------------------------------------
    home_rows = [r for r in history if r.is_home]
    away_rows = [r for r in history if not r.is_home]
    for label, rows, target in (("home", home_rows, form.home_splits),
                                ("away", away_rows, form.away_splits)):
        if not rows:
            form.missing.append(f"no {label} matches before the cutoff")
            continue
        for stat_name, attr in PER_MATCH_STATS.items():
            target[stat_name] = weighted_mean(_series(rows, attr), window=label)
        target["points"] = weighted_mean([float(r.points) for r in rows if r.points is not None],
                                         window=label)

    # -- outcome rates -----------------------------------------------------
    form.clean_sheet_rate = rate([r.goals_against == 0 for r in history if r.goals_against is not None],
                                 window="all")
    form.scoring_rate = rate([(r.goals_for or 0) > 0 for r in history if r.goals_for is not None],
                             window="all")
    form.conceding_rate = rate([(r.goals_against or 0) > 0 for r in history if r.goals_against is not None],
                               window="all")
    form.btts_rate = rate(
        [(r.goals_for or 0) > 0 and (r.goals_against or 0) > 0
         for r in history if r.goals_for is not None and r.goals_against is not None],
        window="all",
    )

    # -- points per game per window ---------------------------------------
    for window_name, size in WINDOWS.items():
        subset = history[:size] if size else history
        pts = [float(r.points) for r in subset if r.points is not None]
        form.form_points[window_name] = round(sum(pts) / len(pts), 3) if pts else None

    # -- first-half share of goals ----------------------------------------
    ht = [(r.ht_goals_for, r.goals_for) for r in history
          if r.ht_goals_for is not None and r.goals_for]
    if ht:
        total_full = sum(full for _, full in ht)
        if total_full:
            form.first_half_share = round(sum(h for h, _ in ht) / total_full, 3)
    else:
        form.missing.append("no half-time scores recorded")

    # -- schedule ----------------------------------------------------------
    last = history[0]
    form.rest_days = (repo.as_of.date() - last.date).days
    form.matches_last_14_days = sum(
        1 for r in history if (repo.as_of.date() - r.date).days <= 14
    )

    # -- recent trend: last 5 points per game vs longer-run baseline -------
    short, long = form.form_points.get("last5"), form.form_points.get("all")
    if short is not None and long is not None:
        form.trend = round(short - long, 3)

    return form


def opponent_adjust(
    repo: AsOfRepository,
    form: TeamForm,
    *,
    competition_ids: Optional[Sequence[int]] = None,
    window: int = 20,
    league_average_goals: Optional[float] = None,
) -> None:
    """Rescale attack/defence output by the quality of opposition faced.

    For each match we ask: how many goals does this opponent normally concede?
    Scoring twice against a leaky defence is worth less than scoring twice
    against a miserly one, and over a 20-match window those differences add up
    to a materially different estimate of a team's true attacking strength.
    """
    history = repo.team_history(form.team_id, limit=window, competition_ids=competition_ids)
    if not history:
        return

    attack_ratios: list[float] = []
    defence_ratios: list[float] = []
    for row in history:
        if row.goals_for is None or row.goals_against is None:
            continue
        opp_history = repo.team_history(row.opponent_id, limit=30,
                                        competition_ids=competition_ids)
        # Only the opponent's matches before this one, so the adjustment
        # itself cannot use information from after the match being described.
        prior = [r for r in opp_history if r.date < row.date]
        conceded = [r.goals_against for r in prior if r.goals_against is not None]
        scored = [r.goals_for for r in prior if r.goals_for is not None]
        if len(conceded) >= 5:
            opp_concede = sum(conceded) / len(conceded)
            if opp_concede > 0.05:
                attack_ratios.append(row.goals_for / opp_concede)
        if len(scored) >= 5:
            opp_score = sum(scored) / len(scored)
            if opp_score > 0.05:
                defence_ratios.append(row.goals_against / opp_score)

    baseline = league_average_goals if league_average_goals else 1.35
    if attack_ratios:
        form.opponent_adjusted["attack_index"] = round(sum(attack_ratios) / len(attack_ratios), 3)
        form.opponent_adjusted["adjusted_goals_for"] = round(
            (sum(attack_ratios) / len(attack_ratios)) * baseline, 3
        )
        form.opponent_adjusted["attack_sample"] = len(attack_ratios)
    if defence_ratios:
        form.opponent_adjusted["defence_index"] = round(sum(defence_ratios) / len(defence_ratios), 3)
        form.opponent_adjusted["adjusted_goals_against"] = round(
            (sum(defence_ratios) / len(defence_ratios)) * baseline, 3
        )
        form.opponent_adjusted["defence_sample"] = len(defence_ratios)


def league_baseline(repo: AsOfRepository, competition_id: int, *, seasons: int = 3) -> dict:
    """League-wide averages, the reference point for every relative statement."""
    since = dt.date(repo.as_of.year - seasons, 7, 1)
    matches = repo.completed_matches(competition_ids=[competition_id], since=since)
    if not matches:
        return {"matches": 0}

    home_goals = [m.ft_home_goals for m in matches if m.ft_home_goals is not None]
    away_goals = [m.ft_away_goals for m in matches if m.ft_away_goals is not None]
    if not home_goals:
        return {"matches": len(matches)}

    n = len(home_goals)
    results = [m.result for m in matches if m.result]
    baseline = {
        "matches": n,
        "avg_home_goals": round(sum(home_goals) / n, 3),
        "avg_away_goals": round(sum(away_goals) / n, 3),
        "avg_total_goals": round((sum(home_goals) + sum(away_goals)) / n, 3),
        "home_win_rate": round(results.count("H") / len(results), 3) if results else None,
        "draw_rate": round(results.count("D") / len(results), 3) if results else None,
        "away_win_rate": round(results.count("A") / len(results), 3) if results else None,
        "since": since.isoformat(),
    }

    # Corner and card baselines only when actually recorded.
    corner_totals, card_totals = [], []
    for m in matches[:400]:
        home_line, away_line = repo.stat_pair(m)
        if home_line is None or away_line is None:
            continue
        if home_line.corners is not None and away_line.corners is not None:
            corner_totals.append(home_line.corners + away_line.corners)
        if home_line.yellow_cards is not None and away_line.yellow_cards is not None:
            card_totals.append(
                (home_line.yellow_cards or 0) + (away_line.yellow_cards or 0)
                + (home_line.red_cards or 0) + (away_line.red_cards or 0)
            )
    if corner_totals:
        baseline["avg_total_corners"] = round(sum(corner_totals) / len(corner_totals), 3)
        baseline["corner_sample"] = len(corner_totals)
    if card_totals:
        baseline["avg_total_cards"] = round(sum(card_totals) / len(card_totals), 3)
        baseline["card_sample"] = len(card_totals)
    return baseline


def head_to_head_summary(repo: AsOfRepository, home_id: int, away_id: int,
                         *, limit: int = 12) -> dict:
    rows = repo.head_to_head(home_id, away_id, limit=limit)
    if not rows:
        return {"matches": 0, "note": "no previous meetings before the cutoff"}
    played = [r for r in rows if r.goals_for is not None and r.goals_against is not None]
    if not played:
        return {"matches": 0, "note": "previous meetings have no recorded scores"}
    wins = sum(1 for r in played if r.outcome == "W")
    draws = sum(1 for r in played if r.outcome == "D")
    losses = sum(1 for r in played if r.outcome == "L")
    at_home = [r for r in played if r.is_home]
    return {
        "matches": len(played),
        "home_team_wins": wins,
        "draws": draws,
        "away_team_wins": losses,
        "avg_goals_home_team": round(sum(r.goals_for for r in played) / len(played), 2),
        "avg_goals_away_team": round(sum(r.goals_against for r in played) / len(played), 2),
        "avg_total_goals": round(
            sum(r.goals_for + r.goals_against for r in played) / len(played), 2),
        "meetings_at_this_venue": len(at_home),
        "most_recent": played[0].date.isoformat(),
        "results": [
            {"date": r.date.isoformat(), "home": r.is_home,
             "score": f"{r.goals_for}-{r.goals_against}"}
            for r in played[:6]
        ],
    }


def build_match_features(
    repo: AsOfRepository,
    home_id: int,
    away_id: int,
    competition_id: int,
    *,
    cross_competition: bool = True,
) -> MatchFeatures:
    """Assemble every feature both models and explanations draw on."""
    comp = repo.competition(competition_id)
    baseline = league_baseline(repo, competition_id)
    # Form is computed across competitions by default (a midweek cup game
    # still tires the squad and still reveals quality), while the baseline and
    # standings remain league-specific.
    comp_filter = None if cross_competition else [competition_id]

    home = build_team_form(repo, home_id, competition_ids=comp_filter)
    away = build_team_form(repo, away_id, competition_ids=comp_filter)

    avg_goals = baseline.get("avg_total_goals")
    per_side = (avg_goals / 2) if avg_goals else None
    opponent_adjust(repo, home, competition_ids=comp_filter, league_average_goals=per_side)
    opponent_adjust(repo, away, competition_ids=comp_filter, league_average_goals=per_side)

    table = repo.league_table(competition_id, season_start_for(repo.as_of.date()))
    for form in (home, away):
        row = table.get(form.team_id)
        if row:
            form.league_position = row["position"]
            form.league_points = row["points"]
            form.league_ppg = row["ppg"]

    features = MatchFeatures(
        home=home, away=away, as_of=repo.as_of, competition_id=competition_id,
        competition_name=comp.name if comp else "",
        h2h=head_to_head_summary(repo, home_id, away_id),
        league_baseline=baseline,
    )

    # Schedule context, expressed relative to the two teams.
    features.context = {
        "home_rest_days": home.rest_days,
        "away_rest_days": away.rest_days,
        # Positive means the HOME side has had more days off. Named explicitly
        # because an ambiguous sign here produces a confidently stated,
        # factually backwards explanation.
        "home_rest_advantage_days": (
            (home.rest_days - away.rest_days)
            if home.rest_days is not None and away.rest_days is not None else None
        ),
        "home_congestion_14d": home.matches_last_14_days,
        "away_congestion_14d": away.matches_last_14_days,
        "league_table_teams": len(table),
    }

    for label, form in (("home", home), ("away", away)):
        for note in form.missing:
            features.missing.append(f"{label} ({form.team_name}): {note}")
    if baseline.get("matches", 0) < 50:
        features.missing.append(
            f"league baseline rests on only {baseline.get('matches', 0)} matches"
        )
    return features
