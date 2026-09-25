"""Dixon-Coles bivariate Poisson goal model.

The standard workhorse for football scorelines, and the right default here:
it estimates a per-team attack and defence rating plus a home-advantage term
by maximum likelihood, then gives a full joint distribution over scorelines -
which is what makes over/under, both-teams-to-score, correct score and clean
sheet probabilities fall out of one coherent model rather than being
estimated separately and contradicting each other.

Two refinements over naive independent Poisson, both from Dixon & Coles
(1997):

*   **Low-score dependence.** Independent Poisson understates 0-0 and 1-1 and
    overstates 1-0 and 0-1. A correction term ``tau`` with parameter ``rho``
    fixes those four cells.
*   **Time decay.** Each match's contribution to the likelihood is weighted by
    ``0.5 ** (age / half_life)``, so recent form counts for more without
    older matches being discarded.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from mebet.logging_setup import get_logger
from mebet.models.base import (
    ModelPrediction,
    PredictionModel,
    PredictionRequest,
    TrainingContext,
    outcome_probs_from_matrix,
)

log = get_logger("models.dixon_coles")

MAX_GOALS = 10


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles correction for the four low-scoring cells."""
    hg = np.asarray(home_goals)
    ag = np.asarray(away_goals)
    out = np.ones_like(lam, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    out = np.where(m00, 1.0 - lam * mu * rho, out)
    out = np.where(m01, 1.0 + lam * rho, out)
    out = np.where(m10, 1.0 + mu * rho, out)
    out = np.where(m11, 1.0 - rho, out)
    return out


class DixonColesModel(PredictionModel):
    key = "dixon_coles"
    name = "Dixon-Coles bivariate Poisson (time-decayed)"
    produces = ("1x2", "goals", "correct_score", "btts", "clean_sheet")
    yields_score_matrix = True

    def __init__(self, half_life_days: float = 180.0, max_goals: int = MAX_GOALS,
                 min_matches: int = 60, min_team_matches: int = 4) -> None:
        super().__init__(half_life_days=half_life_days, max_goals=max_goals,
                         min_matches=min_matches, min_team_matches=min_team_matches)
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.min_matches = min_matches
        self.min_team_matches = min_team_matches
        self.team_index: dict[int, int] = {}
        self.attack: np.ndarray = np.array([])
        self.defence: np.ndarray = np.array([])
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self.team_match_counts: dict[int, int] = {}

    # -- fitting ----------------------------------------------------------
    def fit(self, ctx: TrainingContext) -> "DixonColesModel":
        matches = [
            m for m in ctx.training_matches()
            if m.ft_home_goals is not None and m.ft_away_goals is not None
        ]
        self.n_training_matches = len(matches)
        self.training_cutoff = ctx.as_of
        if len(matches) < self.min_matches:
            self.fitted = False
            self.fit_diagnostics = {
                "error": f"only {len(matches)} training matches, need {self.min_matches}"
            }
            return self

        teams = sorted({m.home_team_id for m in matches} | {m.away_team_id for m in matches})
        self.team_index = {tid: i for i, tid in enumerate(teams)}
        n_teams = len(teams)

        hi = np.array([self.team_index[m.home_team_id] for m in matches])
        ai = np.array([self.team_index[m.away_team_id] for m in matches])
        hg = np.array([m.ft_home_goals for m in matches], dtype=float)
        ag = np.array([m.ft_away_goals for m in matches], dtype=float)

        as_of_date = ctx.as_of.date()
        ages = np.array([(as_of_date - m.kickoff_date).days for m in matches], dtype=float)
        half_life = ctx.half_life_days or self.half_life_days
        weights = np.power(0.5, np.maximum(ages, 0.0) / half_life) if half_life > 0 else np.ones_like(ages)

        counts: dict[int, int] = {}
        for m in matches:
            counts[m.home_team_id] = counts.get(m.home_team_id, 0) + 1
            counts[m.away_team_id] = counts.get(m.away_team_id, 0) + 1
        self.team_match_counts = counts

        # Parameters: [attack_0..n-2, defence_0..n-1, home_adv, rho].
        # The final attack parameter is fixed by the sum-to-zero identifiability
        # constraint rather than being optimised.
        def unpack(theta):
            attack_free = theta[: n_teams - 1]
            attack = np.concatenate([attack_free, [-attack_free.sum()]])
            defence = theta[n_teams - 1: 2 * n_teams - 1]
            return attack, defence, theta[-2], theta[-1]

        def negative_log_likelihood(theta):
            attack, defence, home_adv, rho = unpack(theta)
            lam = np.exp(attack[hi] - defence[ai] + home_adv)
            mu = np.exp(attack[ai] - defence[hi])
            lam = np.clip(lam, 1e-6, 25.0)
            mu = np.clip(mu, 1e-6, 25.0)
            correction = tau(hg, ag, lam, mu, rho)
            correction = np.clip(correction, 1e-9, None)
            ll = (
                np.log(correction)
                + hg * np.log(lam) - lam
                + ag * np.log(mu) - mu
            )
            return -float(np.sum(weights * ll))

        x0 = np.concatenate([
            np.zeros(n_teams - 1),          # attack
            np.zeros(n_teams),              # defence
            [0.25],                         # home advantage
            [-0.05],                        # rho
        ])
        bounds = (
            [(-3.0, 3.0)] * (n_teams - 1)
            + [(-3.0, 3.0)] * n_teams
            + [(-1.0, 1.5), (-0.4, 0.4)]
        )

        result = minimize(negative_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 400, "ftol": 1e-7})

        attack, defence, home_adv, rho = unpack(result.x)
        self.attack, self.defence = attack, defence
        self.home_advantage, self.rho = float(home_adv), float(rho)
        self.fitted = True
        self.fit_diagnostics = {
            "converged": bool(result.success),
            "negative_log_likelihood": round(float(result.fun), 2),
            "iterations": int(result.nit),
            "teams": n_teams,
            "matches": len(matches),
            "effective_sample": round(float(weights.sum()), 1),
            "home_advantage": round(self.home_advantage, 4),
            "rho": round(self.rho, 4),
            "half_life_days": half_life,
        }
        if not result.success:
            log.warning("Dixon-Coles optimiser did not converge: %s", result.message)
        return self

    # -- prediction --------------------------------------------------------
    def expected_goals(self, home_id: int, away_id: int,
                       neutral: bool = False) -> Optional[tuple[float, float]]:
        if not self.fitted:
            return None
        if home_id not in self.team_index or away_id not in self.team_index:
            return None
        h, a = self.team_index[home_id], self.team_index[away_id]
        advantage = 0.0 if neutral else self.home_advantage
        lam = float(np.exp(self.attack[h] - self.defence[a] + advantage))
        mu = float(np.exp(self.attack[a] - self.defence[h]))
        return min(lam, 8.0), min(mu, 8.0)

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        goals = np.arange(self.max_goals + 1)
        home_pmf = poisson.pmf(goals, lam)
        away_pmf = poisson.pmf(goals, mu)
        matrix = np.outer(home_pmf, away_pmf)
        # Apply the low-score correction.
        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho
        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def predict(self, request: PredictionRequest) -> ModelPrediction:
        if not self.fitted:
            return self._insufficient(
                self.fit_diagnostics.get("error", "model is not fitted")
            )
        missing = [
            tid for tid in (request.home_team_id, request.away_team_id)
            if tid not in self.team_index
        ]
        if missing:
            return self._insufficient(
                "no rated history for one or both teams in this competition "
                "(newly promoted, or first season in the dataset)"
            )
        thin = [
            tid for tid in (request.home_team_id, request.away_team_id)
            if self.team_match_counts.get(tid, 0) < self.min_team_matches
        ]
        if thin:
            return self._insufficient(
                f"fewer than {self.min_team_matches} rated matches for "
                f"{len(thin)} of the two teams"
            )

        lam, mu = self.expected_goals(request.home_team_id, request.away_team_id,
                                      request.neutral_venue)
        # Adjustments (e.g. availability) scale the rates multiplicatively and
        # are supplied by the engine, never invented here.
        lam *= float(request.adjustments.get("home_attack_multiplier", 1.0))
        mu *= float(request.adjustments.get("away_attack_multiplier", 1.0))
        lam = float(np.clip(lam, 0.05, 8.0))
        mu = float(np.clip(mu, 0.05, 8.0))

        matrix = self.score_matrix(lam, mu)
        return ModelPrediction(
            model_key=self.key,
            sufficient_data=True,
            outcome_probs=outcome_probs_from_matrix(matrix),
            score_matrix=matrix,
            expected_home_goals=lam,
            expected_away_goals=mu,
            diagnostics={
                "home_attack": round(float(self.attack[self.team_index[request.home_team_id]]), 4),
                "home_defence": round(float(self.defence[self.team_index[request.home_team_id]]), 4),
                "away_attack": round(float(self.attack[self.team_index[request.away_team_id]]), 4),
                "away_defence": round(float(self.defence[self.team_index[request.away_team_id]]), 4),
                "home_advantage": round(self.home_advantage, 4),
                "rho": round(self.rho, 4),
                "training_matches": self.n_training_matches,
            },
        )

    def team_ratings(self) -> list[dict]:
        if not self.fitted:
            return []
        return [
            {
                "team_id": tid,
                "attack": round(float(self.attack[i]), 4),
                "defence": round(float(self.defence[i]), 4),
                "matches": self.team_match_counts.get(tid, 0),
            }
            for tid, i in sorted(self.team_index.items(), key=lambda kv: kv[1])
        ]
