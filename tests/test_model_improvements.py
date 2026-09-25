"""Tests for the accuracy work on Dixon-Coles and the evaluation harness.

Each test pins a specific claim: the analytic gradient is correct, the
default configuration is still ordinary Dixon-Coles, shot conversion rates
come only from training data, the model's own half-life is honoured, the
prior rates teams that would otherwise be declined, and the evaluation
harness never fits on data from after the match it predicts.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from scipy.optimize import approx_fprime

import mebet.models.dixon_coles as dcmod
from mebet.backtest.evaluate import (
    EvaluationResult,
    MatchScore,
    paired_comparison,
    walk_forward,
)
from mebet.db.index import MatchIndex
from mebet.db.models import Match, Team, TeamMatchStats
from mebet.db.repository import AsOfRepository
from mebet.models.base import PredictionRequest, TrainingContext
from mebet.models.dixon_coles import DixonColesModel

AS_OF = dt.datetime(2025, 8, 1)


def _ctx(session, seeded, as_of=AS_OF, index=None):
    repo = AsOfRepository(session, as_of, index=index)
    return TrainingContext(repo=repo, competition_ids=[seeded["competition"].id],
                           as_of=as_of, index=index)


def _request(seeded, home=0, away=1):
    return PredictionRequest(seeded["teams"][home].id, seeded["teams"][away].id,
                             seeded["competition"].id, AS_OF)


@pytest.mark.parametrize("config", [
    {},
    {"sot_weight": 0.35, "shots_weight": 0.15, "ridge": 2.0, "promoted_offset": 0.3},
])
def test_analytic_gradient_matches_finite_differences(seeded, session, monkeypatch, config):
    errors = []
    real = dcmod.minimize

    def checking(fun, x0, **kw):
        rng = np.random.default_rng(1)
        for _ in range(3):
            x = x0 + rng.normal(0, 0.2, size=x0.shape)
            _, grad = fun(x)
            numeric = approx_fprime(x, lambda z: fun(z)[0], 1e-6)
            errors.append(np.max(np.abs(grad - numeric)) / max(1.0, np.max(np.abs(numeric))))
        return real(fun, x0, **kw)

    monkeypatch.setattr(dcmod, "minimize", checking)
    DixonColesModel(min_matches=20, **config).fit(_ctx(session, seeded))
    assert max(errors) < 1e-4


def test_neutral_settings_fit_on_goals_alone(seeded, session):
    model = DixonColesModel(min_matches=20, sot_weight=0, shots_weight=0, ridge=0,
                            promoted_offset=0, half_life_days=180)
    model.fit(_ctx(session, seeded))
    assert model.fit_diagnostics["rating_target"]["goals"] == 1.0
    assert model.conversion == {}  # shot data never consulted


def test_attack_and_defence_are_both_centred(seeded, session):
    """0.00 must mean 'league average' for defence too - the explanation text says so."""
    model = DixonColesModel(min_matches=20, sot_weight=0.35, shots_weight=0.15)
    model.fit(_ctx(session, seeded))
    assert abs(model.attack.sum()) < 1e-9
    assert abs(model.defence.sum()) < 1e-9


def test_shot_conversion_is_learned_from_training_matches_only(seeded, session):
    model = DixonColesModel(min_matches=20, sot_weight=0.5)
    ctx = _ctx(session, seeded)
    model.fit(ctx)

    goals = sot = 0
    for m in ctx.training_matches():
        home, away = ctx.repo.stat_pair(m)
        goals += m.ft_home_goals + m.ft_away_goals
        sot += home.shots_on_target + away.shots_on_target
        assert m.kickoff_date < AS_OF.date()
    assert model.conversion["per_shot_on_target"] == pytest.approx(goals / sot)


def test_model_half_life_is_honoured_over_the_context_value(seeded, session):
    """Regression: the context's half-life used to override the model's own."""
    short = DixonColesModel(min_matches=20, half_life_days=20).fit(_ctx(session, seeded))
    long = DixonColesModel(min_matches=20, half_life_days=2000).fit(_ctx(session, seeded))
    assert short.fit_diagnostics["half_life_days"] == 20
    assert long.fit_diagnostics["half_life_days"] == 2000
    assert not np.allclose(short.attack, long.attack)


def _add_promoted_team(session, seeded, n_matches: int) -> Team:
    """A club with no matches last season, and a few in the current one."""
    newcomer = Team(sport="football", canonical_name="Newcomer FC", country="Testland")
    session.add(newcomer)
    session.flush()
    opponent = seeded["teams"][0]
    for i in range(n_matches):
        date = dt.date(2025, 7, 10) + dt.timedelta(days=4 * i)
        m = Match(sport="football", competition_id=seeded["competition"].id,
                  season_id=seeded["season"].id, kickoff_date=date,
                  kickoff_utc=dt.datetime.combine(date, dt.time(15)), kickoff_is_exact=True,
                  home_team_id=newcomer.id, away_team_id=opponent.id, status="played",
                  ft_home_goals=0, ft_away_goals=2, source_id=seeded["source"].id)
        session.add(m)
        session.flush()
        for team, home, g, c in ((newcomer, True, 0, 2), (opponent, False, 2, 0)):
            session.add(TeamMatchStats(match_id=m.id, team_id=team.id, is_home=home,
                                       goals=g, goals_conceded=c, shots=8, shots_on_target=2,
                                       source_id=seeded["source"].id))
    session.commit()
    return newcomer


def test_promoted_team_is_detected_and_given_the_prior(seeded, session):
    newcomer = _add_promoted_team(session, seeded, n_matches=2)
    model = DixonColesModel(min_matches=20, ridge=1.0, promoted_offset=0.3)
    model.fit(_ctx(session, seeded))
    assert newcomer.id in model.promoted_teams
    assert not any(t.id in model.promoted_teams for t in seeded["teams"])


def test_thin_history_is_declined_without_a_prior_but_rated_with_one(seeded, session):
    newcomer = _add_promoted_team(session, seeded, n_matches=2)
    req = PredictionRequest(newcomer.id, seeded["teams"][1].id, seeded["competition"].id, AS_OF)

    plain = DixonColesModel(min_matches=20, ridge=0.0, rate_unrated_teams=False)
    plain.fit(_ctx(session, seeded))
    assert not plain.predict(req).sufficient_data

    with_prior = DixonColesModel(min_matches=20, ridge=1.0, promoted_offset=0.3)
    with_prior.fit(_ctx(session, seeded))
    pred = with_prior.predict(req)
    assert pred.sufficient_data
    assert pred.diagnostics["home_newly_promoted"] is True


def test_unseen_team_is_rated_from_the_prior_only_when_allowed(seeded, session):
    stranger = Team(sport="football", canonical_name="Never Seen FC")
    session.add(stranger)
    session.commit()
    req = PredictionRequest(stranger.id, seeded["teams"][1].id, seeded["competition"].id, AS_OF)

    declined = DixonColesModel(min_matches=20, ridge=1.0, rate_unrated_teams=False)
    declined.fit(_ctx(session, seeded))
    assert not declined.predict(req).sufficient_data

    allowed = DixonColesModel(min_matches=20, ridge=1.0, promoted_offset=0.3,
                              rate_unrated_teams=True).fit(_ctx(session, seeded))
    pred = allowed.predict(req)
    assert pred.sufficient_data
    assert pred.diagnostics["home_rated_from_prior"] is True
    # A typical promoted side should be the underdog against an established one.
    assert pred.outcome_probs["home"] < pred.outcome_probs["away"]


def test_invalid_blend_weights_are_rejected():
    with pytest.raises(ValueError):
        DixonColesModel(sot_weight=0.8, shots_weight=0.4)


# --- evaluation harness ------------------------------------------------------
def test_walk_forward_never_fits_on_data_after_the_match(seeded, session):
    fit_cutoffs = []

    class Spy(DixonColesModel):
        def fit(self, ctx):
            fit_cutoffs.append(ctx.as_of)
            return super().fit(ctx)

    index = MatchIndex(session)
    result = walk_forward(session, index, seeded["competition"].key,
                          dt.date(2025, 6, 1), dt.date(2025, 12, 31),
                          lambda: Spy(min_matches=20), refit_days=14)
    assert result.scores and result.fits == len(fit_cutoffs)
    for score in result.scores.values():
        latest_fit = max(c for c in fit_cutoffs if c.date() <= score.date)
        assert latest_fit.date() <= score.date


def _result(label, probs_by_id, outcome=0):
    r = EvaluationResult(label=label)
    for mid, p in probs_by_id.items():
        r.scores[mid] = MatchScore(mid, "TEST.1", dt.date(2025, 1, 1) + dt.timedelta(days=mid),
                                   outcome, p)
    return r


def test_paired_comparison_sign_and_shared_matches():
    good = _result("good", {i: (0.7, 0.2, 0.1) for i in range(60)})
    bad = _result("bad", {i: (0.4, 0.3, 0.3) for i in range(80)})
    cmp = paired_comparison(bad, good)
    assert cmp["n"] == 60                 # only matches both predicted
    assert cmp["diff"] < 0                # negative = candidate better
    assert cmp["ci95"][1] < 0
    assert cmp["p_better"] == pytest.approx(1.0)


def test_shipped_defaults_are_the_validated_configuration():
    """The defaults are what the final held-out test measured; changing them
    means re-running scripts/accuracy_study/final_test.py."""
    from mebet.models.elo import EloModel
    from mebet.models.registry import DEFAULT_ENSEMBLE_WEIGHTS, build_ensemble

    dc = DixonColesModel().params
    assert (dc["half_life_days"], dc["sot_weight"], dc["shots_weight"], dc["ridge"],
            dc["promoted_offset"], dc["rate_unrated_teams"]) == (270.0, 0.35, 0.15, 1.0, 0.3, True)
    assert EloModel().params["season_regression"] == 0.1
    assert abs(sum(DEFAULT_ENSEMBLE_WEIGHTS.values()) - 1.0) < 1e-3
    ensemble = build_ensemble()
    assert ensemble.weights == DEFAULT_ENSEMBLE_WEIGHTS
    assert ensemble.weight_source == "validated default"


def test_weights_from_an_older_model_version_are_not_reused(session):
    from mebet.backtest.runner import latest_ensemble_weights
    from mebet.db.models import BacktestRun
    from mebet.models.registry import MODEL_VERSION

    session.add(BacktestRun(label="old", spec={"competition": "TEST.1",
                                               "ensemble_weights": {"dixon_coles": 0.9, "elo": 0.1}}))
    session.commit()
    assert latest_ensemble_weights(session, "TEST.1") is None

    session.add(BacktestRun(label="current", spec={"competition": "TEST.1", "model_version": MODEL_VERSION,
                                                   "ensemble_weights": {"dixon_coles": 0.6, "elo": 0.4}}))
    session.commit()
    assert latest_ensemble_weights(session, "TEST.1") == {"dixon_coles": 0.6, "elo": 0.4}
