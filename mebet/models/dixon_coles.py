"""Dixon-Coles bivariate Poisson goal model.

The standard workhorse for football scorelines, and the right default here:
it estimates a per-team attack and defence rating plus a home-advantage term
by maximum likelihood, then gives a full joint distribution over scorelines -
which is what makes over/under, both-teams-to-score, correct score and clean
sheet probabilities fall out of one coherent model rather than being
estimated separately and contradicting each other.

Refinements over naive independent Poisson:

*   **Low-score dependence** (Dixon & Coles, 1997). Independent Poisson
    understates 0-0 and 1-1 and overstates 1-0 and 0-1. A correction term
    ``tau`` with parameter ``rho`` fixes those four cells.
*   **Time decay.** Each match's contribution to the likelihood is weighted by
    ``0.5 ** (age / half_life)``, so recent form counts for more without
    older matches being discarded.
*   **Shot-informed ratings.** Goals are a noisy measure of how well a team
    played: a side can create ten good chances and score none. Shots on
    target are a far steadier signal. The ratings can be fitted on a blend of
    actual goals and a shot-based expected-goals proxy - shots multiplied by
    the league's own conversion rate, estimated from the training window only
    - so the scale stays in goals while the noise drops. The low-score
    correction still uses the real scorelines.
*   **Shrinkage prior.** A team with little history gets a noisy maximum
    likelihood rating. A ridge prior pulls thin samples toward the league
    average - or, for a side that was not in the league last season, toward a
    "typical promoted team" rating - so early-season and newly promoted
    predictions stop being driven by two or three results.

Every refinement is off at its neutral value unless configured, and the
defaults are the values chosen by walk-forward validation on seasons that
precede the reported test window (see ``mebet.backtest.evaluate``).
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from ..logging_setup import get_logger
from .base import (
    ModelPrediction,
    PredictionModel,
    PredictionRequest,
    TrainingContext,
    outcome_probs_from_matrix,
)

log = get_logger("models.dixon_coles")

MAX_GOALS = 10
TAU_FLOOR = 1e-9


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


def season_start(date: dt.date) -> dt.date:
    return dt.date(date.year if date.month >= 7 else date.year - 1, 7, 1)


class DixonColesModel(PredictionModel):
    key = "dixon_coles"
    name = "Dixon-Coles bivariate Poisson (time-decayed, shot-informed)"
    produces = ("1x2", "goals", "correct_score", "btts", "clean_sheet")
    yields_score_matrix = True

    def __init__(
        self,
        half_life_days: Optional[float] = 270.0,
        max_goals: int = MAX_GOALS,
        min_matches: int = 60,
        min_team_matches: int = 4,
        sot_weight: float = 0.35,
        shots_weight: float = 0.15,
        ridge: float = 1.0,
        promoted_offset: float = 0.3,
        rate_unrated_teams: bool = True,
        seasons_back: Optional[int] = None,
    ) -> None:
        super().__init__(
            half_life_days=half_life_days, max_goals=max_goals, min_matches=min_matches,
            min_team_matches=min_team_matches, sot_weight=sot_weight,
            shots_weight=shots_weight, ridge=ridge, promoted_offset=promoted_offset,
            rate_unrated_teams=rate_unrated_teams, seasons_back=seasons_back,
        )
        if sot_weight < 0 or shots_weight < 0 or sot_weight + shots_weight > 1:
            raise ValueError("sot_weight and shots_weight must be >= 0 and sum to at most 1")
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.min_matches = min_matches
        self.min_team_matches = min_team_matches
        #: Share of the rating target taken from shots on target / all shots;
        #: the remainder is actual goals.
        self.sot_weight = sot_weight
        self.shots_weight = shots_weight
        #: Strength of the prior pulling ratings toward their prior mean.
        self.ridge = ridge
        #: Prior mean for a side not in the league last season: attack and
        #: defence both ``-promoted_offset`` (i.e. weaker than average).
        self.promoted_offset = promoted_offset
        #: Whether a team with no history at all may be rated from the prior
        #: alone rather than declined.
        self.rate_unrated_teams = rate_unrated_teams
        self.seasons_back = seasons_back

        self.team_index: dict[int, int] = {}
        self.attack: np.ndarray = np.array([])
        self.defence: np.ndarray = np.array([])
        self.intercept: float = 0.0
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self.team_match_counts: dict[int, int] = {}
        self.promoted_teams: set[int] = set()
        self.conversion: dict[str, float] = {}

    # -- training data -----------------------------------------------------
    def _targets(self, ctx: TrainingContext, matches) -> tuple[np.ndarray, np.ndarray]:
        """Rating targets per match: goals, optionally blended with shots.

        Conversion rates are estimated from these same training matches, so a
        backtested fit never learns them from the future.
        """
        hg = np.array([m.ft_home_goals for m in matches], dtype=float)
        ag = np.array([m.ft_away_goals for m in matches], dtype=float)
        if self.sot_weight == 0 and self.shots_weight == 0:
            return hg, ag

        n = len(matches)
        h_sot = np.full(n, np.nan)
        a_sot = np.full(n, np.nan)
        h_sh = np.full(n, np.nan)
        a_sh = np.full(n, np.nan)
        for i, m in enumerate(matches):
            home_line, away_line = ctx.repo.stat_pair(m)
            if home_line is None or away_line is None:
                continue
            if home_line.shots_on_target is not None and away_line.shots_on_target is not None:
                h_sot[i], a_sot[i] = home_line.shots_on_target, away_line.shots_on_target
            if home_line.shots is not None and away_line.shots is not None:
                h_sh[i], a_sh[i] = home_line.shots, away_line.shots

        def rate(shots_h, shots_a):
            ok = ~np.isnan(shots_h)
            total = np.nansum(shots_h[ok]) + np.nansum(shots_a[ok])
            goals = hg[ok].sum() + ag[ok].sum()
            return float(goals / total) if total > 0 else None

        c_sot = rate(h_sot, a_sot)
        c_sh = rate(h_sh, a_sh)
        self.conversion = {"per_shot_on_target": c_sot, "per_shot": c_sh}

        def blend(goals, sot, shots):
            out = (1.0 - self.sot_weight - self.shots_weight) * goals
            # A missing statistic falls back to the goals it would stand in for,
            # so the target stays on the same scale either way.
            sot_part = np.where(np.isnan(sot) | (c_sot is None), goals, (c_sot or 0) * np.nan_to_num(sot))
            shots_part = np.where(np.isnan(shots) | (c_sh is None), goals, (c_sh or 0) * np.nan_to_num(shots))
            return out + self.sot_weight * sot_part + self.shots_weight * shots_part

        return blend(hg, h_sot, h_sh), blend(ag, a_sot, a_sh)

    # -- fitting ----------------------------------------------------------
    def fit(self, ctx: TrainingContext) -> "DixonColesModel":
        if self.seasons_back is not None and self.seasons_back != ctx.seasons_back:
            ctx = TrainingContext(
                repo=ctx.repo, competition_ids=ctx.competition_ids, as_of=ctx.as_of,
                half_life_days=ctx.half_life_days, min_matches=ctx.min_matches,
                seasons_back=self.seasons_back, index=ctx.index,
            )
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
        n = len(teams)

        hi = np.array([self.team_index[m.home_team_id] for m in matches])
        ai = np.array([self.team_index[m.away_team_id] for m in matches])
        hg_int = np.array([m.ft_home_goals for m in matches])
        ag_int = np.array([m.ft_away_goals for m in matches])
        yh, ya = self._targets(ctx, matches)

        as_of_date = ctx.as_of.date()
        ages = np.array([(as_of_date - m.kickoff_date).days for m in matches], dtype=float)
        # The model's own half-life is authoritative; the context value is only
        # a fallback, so tuning this parameter actually changes the fit.
        half_life = self.half_life_days if self.half_life_days is not None else ctx.half_life_days
        w = np.power(0.5, np.maximum(ages, 0.0) / half_life) if half_life else np.ones_like(ages)

        counts: dict[int, int] = {}
        for m in matches:
            counts[m.home_team_id] = counts.get(m.home_team_id, 0) + 1
            counts[m.away_team_id] = counts.get(m.away_team_id, 0) + 1
        self.team_match_counts = counts

        # Promoted: in the training window, but absent from last season.
        this_season = season_start(as_of_date)
        last_season = dt.date(this_season.year - 1, 7, 1)
        played_last = {
            tid for m in matches if last_season <= m.kickoff_date < this_season
            for tid in (m.home_team_id, m.away_team_id)
        }
        self.promoted_teams = {tid for tid in teams if tid not in played_last} if played_last else set()
        prior_mean = np.zeros(n)
        if self.promoted_offset:
            for tid in self.promoted_teams:
                prior_mean[self.team_index[tid]] = -self.promoted_offset

        # Low-score cell masks, on the real scoreline.
        m00 = (hg_int == 0) & (ag_int == 0)
        m01 = (hg_int == 0) & (ag_int == 1)
        m10 = (hg_int == 1) & (ag_int == 0)
        m11 = (hg_int == 1) & (ag_int == 1)
        ridge = self.ridge

        # theta = [attack_free (n-1), defence_free (n-1), intercept, home, rho].
        # Attack and defence each sum to zero, so 0.00 genuinely means "league
        # average" for both, and the intercept carries the overall goal level.
        def unpack(theta):
            af = theta[: n - 1]
            df = theta[n - 1: 2 * n - 2]
            attack = np.concatenate([af, [-af.sum()]])
            defence = np.concatenate([df, [-df.sum()]])
            return attack, defence, theta[-3], theta[-2], theta[-1]

        def objective(theta):
            attack, defence, c0, home, rho = unpack(theta)
            log_lam = c0 + attack[hi] - defence[ai] + home
            log_mu = c0 + attack[ai] - defence[hi]
            lam, mu = np.exp(log_lam), np.exp(log_mu)

            t = np.ones_like(lam)
            t = np.where(m00, 1.0 - lam * mu * rho, t)
            t = np.where(m01, 1.0 + lam * rho, t)
            t = np.where(m10, 1.0 + mu * rho, t)
            t = np.where(m11, 1.0 - rho, t)
            clipped = t < TAU_FLOOR
            t = np.maximum(t, TAU_FLOOR)

            ll = np.log(t) + yh * log_lam - lam + ya * log_mu - mu
            value = -float(np.sum(w * ll))

            # d tau / d log(lambda), d log(mu), d rho for the four cells.
            dt_dlam = np.where(m00, -lam * mu * rho, np.where(m01, lam * rho, 0.0))
            dt_dmu = np.where(m00, -lam * mu * rho, np.where(m10, mu * rho, 0.0))
            dt_drho = np.where(m00, -lam * mu, np.where(m01, lam, np.where(m10, mu, np.where(m11, -1.0, 0.0))))
            inv_t = np.where(clipped, 0.0, 1.0 / t)

            g_lam = -w * (yh - lam + dt_dlam * inv_t)
            g_mu = -w * (ya - mu + dt_dmu * inv_t)
            g_rho = -float(np.sum(w * dt_drho * inv_t))

            g_att = np.bincount(hi, g_lam, n) + np.bincount(ai, g_mu, n)
            g_def = -np.bincount(ai, g_lam, n) - np.bincount(hi, g_mu, n)

            if ridge:
                value += 0.5 * ridge * (np.sum((attack - prior_mean) ** 2)
                                        + np.sum((defence - prior_mean) ** 2))
                g_att = g_att + ridge * (attack - prior_mean)
                g_def = g_def + ridge * (defence - prior_mean)

            grad = np.concatenate([
                g_att[:-1] - g_att[-1],
                g_def[:-1] - g_def[-1],
                [g_lam.sum() + g_mu.sum(), g_lam.sum(), g_rho],
            ])
            return value, grad

        x0 = np.concatenate([np.zeros(2 * n - 2), [0.2, 0.25, -0.05]])
        bounds = [(-3.0, 3.0)] * (2 * n - 2) + [(-3.0, 3.0), (-1.0, 1.5), (-0.4, 0.4)]
        result = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7})

        attack, defence, c0, home, rho = unpack(result.x)
        self.attack, self.defence = attack, defence
        self.intercept, self.home_advantage, self.rho = float(c0), float(home), float(rho)
        self.fitted = True
        self.fit_diagnostics = {
            "converged": bool(result.success),
            "negative_log_likelihood": round(float(result.fun), 2),
            "iterations": int(result.nit),
            "teams": n,
            "matches": len(matches),
            "effective_sample": round(float(w.sum()), 1),
            "home_advantage": round(self.home_advantage, 4),
            "rho": round(self.rho, 4),
            "half_life_days": half_life,
            "rating_target": {
                "goals": round(1.0 - self.sot_weight - self.shots_weight, 3),
                "shots_on_target": self.sot_weight,
                "shots": self.shots_weight,
            },
            "conversion": {k: (round(v, 4) if v else v) for k, v in self.conversion.items()},
            "ridge": self.ridge,
            # Teams in the window but absent last season. Includes clubs since
            # relegated, which is why this is not simply "promoted this year".
            "absent_last_season": len(self.promoted_teams),
            "promoted_and_playing": sum(
                1 for tid in self.promoted_teams
                if any(m.kickoff_date >= this_season and tid in (m.home_team_id, m.away_team_id)
                       for m in matches)
            ),
        }
        if not result.success:
            log.warning("Dixon-Coles optimiser did not converge: %s", result.message)
        return self

    # -- prediction --------------------------------------------------------
    def _rating(self, team_id: int) -> Optional[tuple[float, float]]:
        """(attack, defence) for a team, from the fit or the promoted prior."""
        if team_id in self.team_index:
            i = self.team_index[team_id]
            return float(self.attack[i]), float(self.defence[i])
        if self.rate_unrated_teams:
            # Never seen in this competition's window: treat as a typical
            # newly promoted side, which is what such a team almost always is.
            return -self.promoted_offset, -self.promoted_offset
        return None

    def expected_goals(self, home_id: int, away_id: int,
                       neutral: bool = False) -> Optional[tuple[float, float]]:
        if not self.fitted:
            return None
        home, away = self._rating(home_id), self._rating(away_id)
        if home is None or away is None:
            return None
        advantage = 0.0 if neutral else self.home_advantage
        lam = float(np.exp(self.intercept + home[0] - away[1] + advantage))
        mu = float(np.exp(self.intercept + away[0] - home[1]))
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
        ids = (request.home_team_id, request.away_team_id)
        if any(self._rating(tid) is None for tid in ids):
            return self._insufficient(
                "no rated history for one or both teams in this competition "
                "(newly promoted, or first season in the dataset)"
            )
        # With a prior in place, a thin sample is shrunk rather than trusted
        # blindly, so the minimum-history gate only applies without one.
        if not self.ridge:
            thin = [tid for tid in ids if self.team_match_counts.get(tid, 0) < self.min_team_matches]
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

        home_att, home_def = self._rating(request.home_team_id)
        away_att, away_def = self._rating(request.away_team_id)
        matrix = self.score_matrix(lam, mu)
        return ModelPrediction(
            model_key=self.key,
            sufficient_data=True,
            outcome_probs=outcome_probs_from_matrix(matrix),
            score_matrix=matrix,
            expected_home_goals=lam,
            expected_away_goals=mu,
            diagnostics={
                "home_attack": round(home_att, 4),
                "home_defence": round(home_def, 4),
                "away_attack": round(away_att, 4),
                "away_defence": round(away_def, 4),
                "home_advantage": round(self.home_advantage, 4),
                "rho": round(self.rho, 4),
                "training_matches": self.n_training_matches,
                "home_rated_from_prior": request.home_team_id not in self.team_index,
                "away_rated_from_prior": request.away_team_id not in self.team_index,
                "home_newly_promoted": request.home_team_id in self.promoted_teams,
                "away_newly_promoted": request.away_team_id in self.promoted_teams,
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
                "promoted": tid in self.promoted_teams,
            }
            for tid, i in sorted(self.team_index.items(), key=lambda kv: kv[1])
        ]
