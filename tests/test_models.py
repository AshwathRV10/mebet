"""Model behaviour."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from mebet.db.repository import AsOfRepository
from mebet.models.base import PredictionRequest, TrainingContext, outcome_probs_from_matrix
from mebet.models.counts import CardsModel, CornersModel
from mebet.models.dixon_coles import DixonColesModel
from mebet.models.elo import EloModel
from mebet.models.ensemble import EnsembleModel, fit_weights, rescale_matrix_to_outcomes


@pytest.fixture
def ctx(seeded, session):
    as_of = dt.datetime(2025, 8, 1)
    repo = AsOfRepository(session, as_of)
    return TrainingContext(repo=repo, competition_ids=[seeded["competition"].id], as_of=as_of)


def _request(seeded, as_of):
    return PredictionRequest(
        home_team_id=seeded["teams"][0].id, away_team_id=seeded["teams"][1].id,
        competition_id=seeded["competition"].id, kickoff=as_of,
    )


def test_dixon_coles_produces_a_proper_distribution(seeded, ctx):
    model = DixonColesModel(min_matches=20).fit(ctx)
    assert model.fitted
    pred = model.predict(_request(seeded, ctx.as_of))
    assert pred.sufficient_data
    pred.validate()
    assert abs(pred.score_matrix.sum() - 1.0) < 1e-9
    assert (pred.score_matrix >= 0).all()
    assert abs(sum(pred.outcome_probs.values()) - 1.0) < 1e-9


def test_score_matrix_margins_match_outcome_probabilities(seeded, ctx):
    model = DixonColesModel(min_matches=20).fit(ctx)
    pred = model.predict(_request(seeded, ctx.as_of))
    derived = outcome_probs_from_matrix(pred.score_matrix)
    for key in ("home", "draw", "away"):
        assert abs(derived[key] - pred.outcome_probs[key]) < 1e-9


def test_home_advantage_is_learned_not_assumed(seeded, ctx):
    model = DixonColesModel(min_matches=20).fit(ctx)
    assert model.fit_diagnostics["converged"]
    assert -1.0 < model.home_advantage < 1.5


def test_model_declines_when_data_is_thin(session, seeded):
    as_of = dt.datetime(2025, 1, 3)
    repo = AsOfRepository(session, as_of)
    ctx = TrainingContext(repo=repo, competition_ids=[seeded["competition"].id], as_of=as_of)
    model = DixonColesModel(min_matches=60).fit(ctx)
    pred = model.predict(_request(seeded, as_of))
    assert not pred.sufficient_data
    assert "training matches" in pred.note


def test_model_declines_for_an_unknown_team(seeded, ctx):
    # With the promoted-team prior disabled, an unseen team cannot be rated.
    model = DixonColesModel(min_matches=20, rate_unrated_teams=False).fit(ctx)
    request = PredictionRequest(
        home_team_id=999999, away_team_id=seeded["teams"][1].id,
        competition_id=seeded["competition"].id, kickoff=ctx.as_of,
    )
    pred = model.predict(request)
    assert not pred.sufficient_data


def test_elo_ratings_are_zero_sum_ish(seeded, ctx):
    model = EloModel(min_matches=20).fit(ctx)
    assert model.fitted
    mean = float(np.mean(list(model.ratings.values())))
    assert abs(mean - model.initial_rating) < 60


def test_elo_draw_probability_is_bounded(seeded, ctx):
    model = EloModel(min_matches=20).fit(ctx)
    for gap in (0, 100, 400, 1000):
        p = model.draw_probability(gap)
        assert 0.05 <= p <= 0.40


def test_count_models_produce_normalised_distributions(seeded, ctx):
    for Model in (CornersModel, CardsModel):
        model = Model(min_matches=20, min_team_matches=4).fit(ctx)
        assert model.fitted, model.fit_diagnostics
        pred = model.predict(_request(seeded, ctx.as_of))
        assert pred.sufficient_data, pred.note
        assert abs(pred.total_distribution.sum() - 1.0) < 1e-6
        assert pred.expected_home_count > 0


def test_ensemble_matrix_is_reconciled_with_its_outcome_probabilities(seeded, ctx):
    ensemble = EnsembleModel([DixonColesModel(min_matches=20), EloModel(min_matches=20)])
    ensemble.fit(ctx)
    pred = ensemble.predict(_request(seeded, ctx.as_of))
    assert pred.sufficient_data
    derived = outcome_probs_from_matrix(pred.score_matrix)
    for key in ("home", "draw", "away"):
        assert abs(derived[key] - pred.outcome_probs[key]) < 1e-6


def test_rescaling_preserves_total_mass():
    matrix = np.full((5, 5), 1 / 25)
    target = {"home": 0.5, "draw": 0.2, "away": 0.3}
    out = rescale_matrix_to_outcomes(matrix, target)
    assert abs(out.sum() - 1.0) < 1e-12
    derived = outcome_probs_from_matrix(out)
    for key, value in target.items():
        assert abs(derived[key] - value) < 1e-9


def test_weight_fitting_prefers_the_better_model():
    n = 400
    rng = np.random.default_rng(7)
    truth = rng.integers(0, 3, size=n)
    good = np.full((n, 3), 0.2)
    good[np.arange(n), truth] = 0.6
    bad = np.full((n, 3), 1 / 3)
    weights, diagnostics = fit_weights({"good": good, "bad": bad}, truth)
    assert weights["good"] > weights["bad"]
    assert diagnostics["log_loss_weighted"] <= diagnostics["log_loss_equal"] + 1e-9


def test_ensemble_reports_when_no_component_can_answer(seeded, session):
    as_of = dt.datetime(2025, 1, 2)
    repo = AsOfRepository(session, as_of)
    ctx = TrainingContext(repo=repo, competition_ids=[seeded["competition"].id], as_of=as_of)
    ensemble = EnsembleModel([DixonColesModel(min_matches=5000)])
    ensemble.fit(ctx)
    pred = ensemble.predict(_request(seeded, as_of))
    assert not pred.sufficient_data
    assert "no component model" in pred.note
