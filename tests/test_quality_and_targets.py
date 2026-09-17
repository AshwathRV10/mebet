"""Data quality assessment and target derivation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from mebet.engine import targets as T
from mebet.quality import (
    DataQualityReport,
    QualityTier,
    check_sample_sizes,
    check_suspicious_values,
    finalise,
)


def test_large_clean_sample_scores_high():
    report = DataQualityReport(sources_used=["a", "b", "c"])
    check_sample_sizes(report, 40, 38, 10)
    report.availability_data = True
    report.lineups = "confirmed"
    finalise(report)
    assert report.tier == QualityTier.HIGH
    assert report.confidence_multiplier > 0.85


def test_tiny_sample_is_downgraded():
    report = DataQualityReport(sources_used=["a"])
    check_sample_sizes(report, 2, 30, 0)
    finalise(report)
    assert report.tier in {QualityTier.LOW, QualityTier.INSUFFICIENT}
    assert report.confidence_multiplier < 0.7
    assert any(i.code == "tiny_sample" for i in report.issues)


def test_no_history_is_insufficient():
    report = DataQualityReport()
    check_sample_sizes(report, 0, 20, 0)
    finalise(report)
    assert report.tier == QualityTier.INSUFFICIENT


def test_confidence_is_separate_from_probability():
    """Quality must change confidence, never the probabilities themselves."""
    high = DataQualityReport(sources_used=["a", "b", "c"])
    check_sample_sizes(high, 40, 40, 10)
    high.availability_data = True
    finalise(high)
    low = DataQualityReport(sources_used=["a"])
    check_sample_sizes(low, 6, 6, 0)
    finalise(low)
    assert high.confidence_multiplier > low.confidence_multiplier
    # Both remain usable multipliers, not probability adjustments.
    assert 0 < low.confidence_multiplier <= 1


def test_suspicious_values_are_flagged():
    from mebet.db.repository import TeamMatchRow

    row = TeamMatchRow(
        match_id=1, date=dt.date(2025, 1, 1), kickoff=None, competition_id=1,
        team_id=1, opponent_id=2, is_home=True, goals_for=99, goals_against=0,
        ht_goals_for=0, ht_goals_against=0, shots=200, shots_on_target=None,
        shots_against=None, sot_against=None, corners=None, corners_against=None,
        fouls=None, fouls_against=None, yellows=None, yellows_against=None,
        reds=None, reds_against=None, xg=None, xga=None,
    )
    report = DataQualityReport()
    flagged = check_suspicious_values(report, [row])
    assert flagged == 2
    assert any(i.code == "suspicious_value" for i in report.issues)


# --- targets ---------------------------------------------------------------
def _matrix(lam=1.4, mu=1.1, size=8):
    from scipy.stats import poisson
    goals = np.arange(size + 1)
    m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    return m / m.sum()


def test_over_under_probabilities_are_complementary():
    rows = T.goals_targets(_matrix(), "test", 0.9, "Home", "Away")
    by_market = {}
    for r in rows:
        by_market.setdefault(r.market, {})[r.selection] = r.probability
    for market, sides in by_market.items():
        if market.startswith("over_under") and "over" in sides and "under" in sides:
            assert abs(sides["over"] + sides["under"] - 1.0) < 1e-9


def test_btts_and_clean_sheets_come_from_the_same_matrix():
    matrix = _matrix()
    rows = {(_r.market, _r.selection): _r.probability for _r in
            T.goals_targets(matrix, "test", 0.9, "Home", "Away")}
    btts = rows[("btts", "yes")]
    assert abs(btts - float(matrix[1:, 1:].sum())) < 1e-9
    assert abs(rows[("clean_sheet", "home")] - float(matrix[:, 0].sum())) < 1e-9


def test_scoreline_targets_are_ordered_by_probability():
    rows = T.scoreline_targets(_matrix(), "test", 0.9, top_n=6)
    correct_score = [r.probability for r in rows if r.market == "correct_score"]
    assert len(correct_score) == 6
    assert correct_score == sorted(correct_score, reverse=True)


def test_conditional_scorelines_match_their_outcome():
    """The likeliest scoreline given a home win must actually be a home win."""
    rows = [r for r in T.scoreline_targets(_matrix(), "test", 0.9)
            if r.market == "most_likely_score_given"]
    assert {r.selection for r in rows} == {"home", "draw", "away"}
    for row in rows:
        home_goals, away_goals = (int(x) for x in row.note.split("-"))
        if row.selection == "home":
            assert home_goals > away_goals
        elif row.selection == "away":
            assert away_goals > home_goals
        else:
            assert home_goals == away_goals


def test_first_half_targets_withheld_without_half_time_data():
    rows = T.first_half_targets(_matrix(), None, None, "test", 0.9)
    assert len(rows) == 1 and not rows[0].sufficient_data
    assert "half-time" in rows[0].note


def test_count_targets_withheld_when_model_declines():
    class Declined:
        sufficient_data = False
        note = "corners were never recorded in this league"
        model_key = "corners"
    rows = T.count_targets(Declined(), "corners", (9.5,), 0.9, "Home", "Away")
    assert len(rows) == 1 and not rows[0].sufficient_data
    assert "never recorded" in rows[0].note


def test_player_markets_declare_what_the_feed_cannot_support():
    rows = T.player_targets([], 0.9)
    assert rows and not rows[0].sufficient_data
    withheld = {r.market for r in T.player_targets([], 0.9) if not r.sufficient_data}
    assert "player_goals" in withheld


def test_rest_advantage_sign_favours_the_better_rested_side(seeded, session):
    """A wrong sign here states the opposite of the truth, confidently."""
    import datetime as dt

    from mebet.db.repository import AsOfRepository
    from mebet.engine.explain import ExplanationEngine
    from mebet.features.football import build_match_features

    comp, teams = seeded["competition"], seeded["teams"]
    repo = AsOfRepository(session, dt.datetime(2025, 6, 1))
    features = build_match_features(repo, teams[0].id, teams[1].id, comp.id)

    home_rest = features.home.rest_days
    away_rest = features.away.rest_days
    advantage = features.context["home_rest_advantage_days"]
    assert advantage == home_rest - away_rest

    # Force a clear gap and check the explanation names the rested side.
    features.home.rest_days = 12
    features.away.rest_days = 3
    features.context["home_rest_advantage_days"] = 9
    factors = ExplanationEngine(features).build()
    rest = [f for f in factors if f.category == "rest"]
    assert rest, "expected a rest factor when the gap is nine days"
    assert rest[0].favours == "home"
    assert features.home.team_name in rest[0].statement
