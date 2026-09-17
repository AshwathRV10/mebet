"""Leakage tests.

These are the most important tests in the suite. If the feature layer can see
past its cutoff, every backtest number the system reports is fiction, and the
failure is silent - the predictions would look better, not broken.
"""

from __future__ import annotations

import datetime as dt

import pytest

from mebet.db.index import MatchIndex
from mebet.db.repository import AsOfRepository, LeakageError


def test_as_of_is_mandatory(session):
    with pytest.raises(ValueError):
        AsOfRepository(session, None)


def test_no_match_at_or_after_cutoff_is_returned(seeded, session):
    cutoff = dt.datetime(2025, 3, 1)
    repo = AsOfRepository(session, cutoff)
    matches = repo.completed_matches()
    assert matches, "expected some matches before the cutoff"
    for m in matches:
        assert m.kickoff_utc < cutoff


def test_in_progress_match_is_excluded(seeded, session):
    """A match that kicked off 30 minutes ago has no known result yet."""
    team = seeded["teams"][0]
    repo_all = AsOfRepository(session, dt.datetime(2026, 1, 1))
    latest = repo_all.completed_matches(team_id=team.id, limit=1)[0]

    just_after_kickoff = latest.kickoff_utc + dt.timedelta(minutes=30)
    repo = AsOfRepository(session, just_after_kickoff)
    returned = {m.id for m in repo.completed_matches(team_id=team.id)}
    assert latest.id not in returned

    well_after = latest.kickoff_utc + dt.timedelta(hours=3)
    repo2 = AsOfRepository(session, well_after)
    assert latest.id in {m.id for m in repo2.completed_matches(team_id=team.id)}


def test_team_history_respects_cutoff(seeded, session):
    team = seeded["teams"][1]
    cutoff = dt.datetime(2025, 2, 15)
    repo = AsOfRepository(session, cutoff)
    for row in repo.team_history(team.id):
        assert row.date < cutoff.date()


def test_head_to_head_respects_cutoff(seeded, session):
    a, b = seeded["teams"][0], seeded["teams"][1]
    cutoff = dt.datetime(2025, 2, 1)
    repo = AsOfRepository(session, cutoff)
    for row in repo.head_to_head(a.id, b.id):
        assert row.date < cutoff.date()


def test_league_table_is_derived_not_future(seeded, session):
    comp = seeded["competition"]
    early = AsOfRepository(session, dt.datetime(2025, 2, 1))
    late = AsOfRepository(session, dt.datetime(2025, 6, 1))
    early_table = early.league_table(comp.id, dt.date(2024, 7, 1))
    late_table = late.league_table(comp.id, dt.date(2024, 7, 1))
    early_played = sum(r["played"] for r in early_table.values())
    late_played = sum(r["played"] for r in late_table.values())
    assert early_played < late_played, "standings must grow with the cutoff"


def test_verification_catches_a_post_cutoff_row(seeded, session):
    """Directly feed the guard a match it should reject."""
    repo = AsOfRepository(session, dt.datetime(2025, 2, 1))
    future = AsOfRepository(session, dt.datetime(2026, 1, 1)).completed_matches(limit=1)
    with pytest.raises(LeakageError):
        repo._verify(future)


def test_index_and_sql_paths_agree(seeded, session):
    """The fast path must not be a different path."""
    from mebet.features.football import build_match_features

    comp = seeded["competition"]
    home, away = seeded["teams"][0], seeded["teams"][2]
    cutoff = dt.datetime(2025, 3, 15)
    index = MatchIndex(session)

    with_index = build_match_features(
        AsOfRepository(session, cutoff, index=index), home.id, away.id, comp.id)
    without = build_match_features(
        AsOfRepository(session, cutoff), home.id, away.id, comp.id)
    assert with_index.as_dict() == without.as_dict()


def test_index_respects_cutoff(seeded, session):
    index = MatchIndex(session)
    cutoff = dt.datetime(2025, 3, 1)
    repo = AsOfRepository(session, cutoff, index=index)
    for m in repo.completed_matches():
        assert m.kickoff_utc < cutoff
    for row in repo.stat_series():
        assert row["date"] < cutoff.date()


def test_features_at_earlier_cutoff_use_fewer_matches(seeded, session):
    from mebet.features.football import build_team_form

    team = seeded["teams"][0]
    early = build_team_form(AsOfRepository(session, dt.datetime(2025, 2, 1)), team.id)
    late = build_team_form(AsOfRepository(session, dt.datetime(2025, 5, 1)), team.id)
    assert early.matches_available < late.matches_available


def test_model_fitted_at_cutoff_ignores_later_results(seeded, session):
    """Two fits at different cutoffs must not be identical if data arrived between."""
    from mebet.models.base import TrainingContext
    from mebet.models.dixon_coles import DixonColesModel

    comp = seeded["competition"]
    def fit(when):
        repo = AsOfRepository(session, when)
        model = DixonColesModel(min_matches=20)
        model.fit(TrainingContext(repo=repo, competition_ids=[comp.id], as_of=when))
        return model

    early, late = fit(dt.datetime(2025, 3, 1)), fit(dt.datetime(2025, 6, 1))
    assert early.n_training_matches < late.n_training_matches
