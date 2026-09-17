"""Count models for corners and cards.

Structure is the standard multiplicative rate model used for football counts:

    expected(team) = league_average x attack_rate(team) x concede_rate(opponent)

where each rate is that team's output relative to the league, computed
separately for home and away because both corner and card counts differ
systematically by venue.

Distribution choice is decided by the data, not by preference. Corner and card
counts are usually overdispersed relative to Poisson, so the model measures
the variance-to-mean ratio in the training sample and uses a negative binomial
when overdispersion is real, falling back to Poisson when it is not.

Cards additionally account for the referee. Card counts vary more by official
than by either team, and the referee is recorded in the source data, so where
an official has enough matches their personal rate is used as a multiplier.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import nbinom, poisson

from ..logging_setup import get_logger
from .base import ModelPrediction, PredictionModel, PredictionRequest, TrainingContext

log = get_logger("models.counts")

MAX_COUNT = 30


@dataclass
class _Rates:
    for_home: dict[int, float]
    for_away: dict[int, float]
    against_home: dict[int, float]
    against_away: dict[int, float]
    counts: dict[int, int]


class CountMarketModel(PredictionModel):
    """Shared implementation; subclasses pick the statistic."""

    stat_key = ""            # key in the repository stat rows
    market = ""              # market name used downstream
    yields_score_matrix = False

    def __init__(self, min_matches: int = 150, min_team_matches: int = 8,
                 half_life_days: float = 365.0, seasons_back: int = 3,
                 shrinkage: float = 6.0) -> None:
        super().__init__(min_matches=min_matches, min_team_matches=min_team_matches,
                         half_life_days=half_life_days, seasons_back=seasons_back,
                         shrinkage=shrinkage)
        self.min_matches = min_matches
        self.min_team_matches = min_team_matches
        self.half_life_days = half_life_days
        self.seasons_back = seasons_back
        # Pulls small samples toward the league average: a team with 6 matches
        # should not be credited with an extreme rate.
        self.shrinkage = shrinkage
        self.league_home_mean = 0.0
        self.league_away_mean = 0.0
        self.rates: Optional[_Rates] = None
        self.dispersion: Optional[float] = None
        self.use_negative_binomial = False
        self.referee_multipliers: dict[str, float] = {}

    # -- helpers ----------------------------------------------------------
    def _value(self, row: dict) -> Optional[float]:
        raise NotImplementedError

    def _collect(self, ctx: TrainingContext) -> list[dict]:
        since = dt.date(ctx.as_of.year - self.seasons_back, 7, 1)
        rows = ctx.repo.stat_series(competition_ids=ctx.competition_ids, since=since)
        return [r for r in rows if self._value(r) is not None]

    def fit(self, ctx: TrainingContext) -> "CountMarketModel":
        self.training_cutoff = ctx.as_of
        rows = self._collect(ctx)
        self.n_training_matches = len({r["match_id"] for r in rows})

        if len(rows) < self.min_matches * 2:   # two rows per match
            self.fitted = False
            self.fit_diagnostics = {
                "error": (
                    f"{self.market}: only {len(rows)//2} matches record this statistic, "
                    f"need {self.min_matches}"
                )
            }
            return self

        home_vals = [self._value(r) for r in rows if r["is_home"]]
        away_vals = [self._value(r) for r in rows if not r["is_home"]]
        if not home_vals or not away_vals:
            self.fitted = False
            self.fit_diagnostics = {"error": f"{self.market}: no home/away split available"}
            return self

        self.league_home_mean = float(np.mean(home_vals))
        self.league_away_mean = float(np.mean(away_vals))
        if self.league_home_mean <= 0 or self.league_away_mean <= 0:
            self.fitted = False
            self.fit_diagnostics = {"error": f"{self.market}: league average is zero"}
            return self

        self.rates = self._team_rates(rows)
        self._fit_dispersion(home_vals, away_vals)
        self._fit_referees(rows)

        self.fitted = True
        self.fit_diagnostics = {
            "matches": self.n_training_matches,
            "league_home_mean": round(self.league_home_mean, 3),
            "league_away_mean": round(self.league_away_mean, 3),
            "variance_to_mean": round(self.dispersion, 3) if self.dispersion else None,
            "distribution": "negative_binomial" if self.use_negative_binomial else "poisson",
            "teams_rated": len(self.rates.counts),
            "referees_rated": len(self.referee_multipliers),
        }
        return self

    def _team_rates(self, rows: list[dict]) -> _Rates:
        as_of = self.training_cutoff.date() if self.training_cutoff else dt.date.today()

        def blank():
            return {}

        sums = {"fh": blank(), "fa": blank(), "ah": blank(), "aa": blank()}
        weights = {"fh": blank(), "fa": blank(), "ah": blank(), "aa": blank()}
        counts: dict[int, int] = {}

        for r in rows:
            value = self._value(r)
            if value is None:
                continue
            age = (as_of - r["date"]).days
            w = 0.5 ** (max(age, 0) / self.half_life_days) if self.half_life_days > 0 else 1.0
            team, opp = r["team_id"], r["opponent_id"]
            counts[team] = counts.get(team, 0) + 1
            # "for" from the team's perspective; "against" from the opponent's.
            fk, ak = ("fh", "aa") if r["is_home"] else ("fa", "ah")
            sums[fk][team] = sums[fk].get(team, 0.0) + value * w
            weights[fk][team] = weights[fk].get(team, 0.0) + w
            sums[ak][opp] = sums[ak].get(opp, 0.0) + value * w
            weights[ak][opp] = weights[ak].get(opp, 0.0) + w

        def to_rate(key: str, league_mean: float) -> dict[int, float]:
            out = {}
            for team, total in sums[key].items():
                w = weights[key].get(team, 0.0)
                if w <= 0:
                    continue
                observed = total / w
                # Shrink toward the league mean in proportion to sample size.
                shrunk = (total + self.shrinkage * league_mean) / (w + self.shrinkage)
                out[team] = shrunk / league_mean if league_mean > 0 else 1.0
            return out

        return _Rates(
            for_home=to_rate("fh", self.league_home_mean),
            for_away=to_rate("fa", self.league_away_mean),
            against_home=to_rate("ah", self.league_home_mean),
            against_away=to_rate("aa", self.league_away_mean),
            counts=counts,
        )

    def _fit_dispersion(self, home_vals: list[float], away_vals: list[float]) -> None:
        totals = np.array(home_vals) + np.array(away_vals[: len(home_vals)])
        if len(totals) < 50:
            self.dispersion = None
            return
        mean, var = float(np.mean(totals)), float(np.var(totals, ddof=1))
        self.dispersion = var / mean if mean > 0 else None
        # Overdispersion beyond 10% is treated as real; below that Poisson is
        # the simpler and equally good description.
        self.use_negative_binomial = bool(self.dispersion and self.dispersion > 1.1)

    def _fit_referees(self, rows: list[dict]) -> None:
        by_ref: dict[str, list[float]] = {}
        for r in rows:
            ref = (r.get("referee") or "").strip()
            value = self._value(r)
            if not ref or value is None:
                continue
            by_ref.setdefault(ref, []).append(value)
        overall = float(np.mean([v for vals in by_ref.values() for v in vals])) if by_ref else 0.0
        if overall <= 0:
            return
        for ref, vals in by_ref.items():
            # Require a real sample before crediting an official with a tendency.
            if len(vals) >= 20:
                multiplier = float(np.mean(vals)) / overall
                self.referee_multipliers[ref] = float(np.clip(multiplier, 0.7, 1.4))

    # -- distribution ------------------------------------------------------
    def _distribution(self, mean_total: float) -> np.ndarray:
        ks = np.arange(MAX_COUNT + 1)
        if self.use_negative_binomial and self.dispersion and self.dispersion > 1.0:
            # Parameterise NB by mean and variance = dispersion * mean.
            variance = self.dispersion * mean_total
            p = mean_total / variance
            n = mean_total * p / (1.0 - p) if p < 1 else mean_total
            if n <= 0 or not (0 < p < 1):
                return poisson.pmf(ks, mean_total)
            return nbinom.pmf(ks, n, p)
        return poisson.pmf(ks, mean_total)

    def predict(self, request: PredictionRequest) -> ModelPrediction:
        if not self.fitted or self.rates is None:
            return self._insufficient(
                self.fit_diagnostics.get("error", f"{self.market} model is not fitted")
            )
        home_id, away_id = request.home_team_id, request.away_team_id
        for tid in (home_id, away_id):
            if self.rates.counts.get(tid, 0) < self.min_team_matches:
                return self._insufficient(
                    f"fewer than {self.min_team_matches} matches with {self.market} data "
                    f"recorded for one of the teams"
                )

        home_attack = self.rates.for_home.get(home_id, 1.0)
        away_concede = self.rates.against_away.get(away_id, 1.0)
        away_attack = self.rates.for_away.get(away_id, 1.0)
        home_concede = self.rates.against_home.get(home_id, 1.0)

        expected_home = self.league_home_mean * home_attack * away_concede
        expected_away = self.league_away_mean * away_attack * home_concede

        referee = str(request.adjustments.get("referee", "")).strip()
        ref_multiplier = self.referee_multipliers.get(referee)
        if ref_multiplier:
            expected_home *= ref_multiplier
            expected_away *= ref_multiplier

        total = max(expected_home + expected_away, 0.05)
        distribution = self._distribution(total)
        distribution = distribution / distribution.sum()

        return ModelPrediction(
            model_key=self.key,
            sufficient_data=True,
            expected_home_count=round(float(expected_home), 2),
            expected_away_count=round(float(expected_away), 2),
            total_distribution=distribution,
            diagnostics={
                "league_home_mean": round(self.league_home_mean, 2),
                "league_away_mean": round(self.league_away_mean, 2),
                "home_attack_rate": round(home_attack, 3),
                "away_concede_rate": round(away_concede, 3),
                "away_attack_rate": round(away_attack, 3),
                "home_concede_rate": round(home_concede, 3),
                "referee": referee or None,
                "referee_multiplier": round(ref_multiplier, 3) if ref_multiplier else None,
                "distribution": "negative_binomial" if self.use_negative_binomial else "poisson",
                "variance_to_mean": round(self.dispersion, 3) if self.dispersion else None,
                "home_sample": self.rates.counts.get(home_id, 0),
                "away_sample": self.rates.counts.get(away_id, 0),
            },
        )


class CornersModel(CountMarketModel):
    key = "corners"
    name = "Corners (multiplicative rate model)"
    market = "corners"
    produces = ("corners",)
    stat_key = "corners"

    def _value(self, row: dict) -> Optional[float]:
        v = row.get("corners")
        return float(v) if v is not None else None


class CardsModel(CountMarketModel):
    key = "cards"
    name = "Cards (multiplicative rate model, referee-aware)"
    market = "cards"
    produces = ("cards",)
    stat_key = "cards"

    def _value(self, row: dict) -> Optional[float]:
        """Card points: a red counts as two, matching how cards are usually
        totalled and reflecting that a red is the more severe event."""
        y, r = row.get("yellows"), row.get("reds")
        if y is None and r is None:
            return None
        return float((y or 0) + 2 * (r or 0))
