"""Elo ratings with margin-of-victory scaling.

Elo is included because it is a genuinely different estimator, not because it
sounds impressive: it is sequential and self-correcting, so it reacts to a
change in a team's level faster than a likelihood fit over several seasons.
Where Dixon-Coles and Elo disagree, the disagreement is informative, and the
ensemble weighting decides which to trust based on measured performance.

Draw handling: Elo natively produces a win expectancy, not a three-way split.
The draw share is estimated from the rating gap using the empirical draw rate
observed in the training data, rather than a hard-coded constant.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Optional

import numpy as np

from ..logging_setup import get_logger
from .base import ModelPrediction, PredictionModel, PredictionRequest, TrainingContext

log = get_logger("models.elo")


class EloModel(PredictionModel):
    key = "elo"
    name = "Elo with margin-of-victory scaling"
    produces = ("1x2",)
    yields_score_matrix = False

    def __init__(self, k_factor: float = 20.0, initial_rating: float = 1500.0,
                 home_advantage: float = 60.0, scale: float = 400.0,
                 season_regression: float = 0.1, min_matches: int = 60) -> None:
        super().__init__(k_factor=k_factor, initial_rating=initial_rating,
                         home_advantage=home_advantage, scale=scale,
                         season_regression=season_regression, min_matches=min_matches)
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.scale = scale
        self.season_regression = season_regression
        self.min_matches = min_matches
        self.ratings: dict[int, float] = {}
        self.match_counts: dict[int, int] = {}
        #: Draw rate as a function of rating gap, learned from training data.
        self._draw_base = 0.26
        self._draw_decay = 0.0012

    def _expected(self, rating_diff: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, -rating_diff / self.scale))

    def fit(self, ctx: TrainingContext) -> "EloModel":
        matches = [
            m for m in ctx.training_matches()
            if m.ft_home_goals is not None and m.ft_away_goals is not None
        ]
        matches.sort(key=lambda m: (m.kickoff_date, m.id))
        self.n_training_matches = len(matches)
        self.training_cutoff = ctx.as_of
        if len(matches) < self.min_matches:
            self.fitted = False
            self.fit_diagnostics = {
                "error": f"only {len(matches)} training matches, need {self.min_matches}"
            }
            return self

        self.ratings = {}
        self.match_counts = {}
        gaps_and_draws: list[tuple[float, int]] = []
        current_season: Optional[int] = None

        for m in matches:
            season = m.kickoff_date.year if m.kickoff_date.month >= 7 else m.kickoff_date.year - 1
            if current_season is not None and season != current_season:
                # Between seasons, pull ratings toward the mean: squads change,
                # and last May's rating is a weaker guide to this August.
                for tid in self.ratings:
                    self.ratings[tid] += self.season_regression * (
                        self.initial_rating - self.ratings[tid]
                    )
            current_season = season

            home = self.ratings.setdefault(m.home_team_id, self.initial_rating)
            away = self.ratings.setdefault(m.away_team_id, self.initial_rating)
            diff = home + self.home_advantage - away
            expected_home = self._expected(diff)

            hg, ag = m.ft_home_goals, m.ft_away_goals
            if hg > ag:
                score = 1.0
            elif hg < ag:
                score = 0.0
            else:
                score = 0.5
                gaps_and_draws.append((abs(diff), 1))
            if hg != ag:
                gaps_and_draws.append((abs(diff), 0))

            # Margin-of-victory multiplier (FiveThirtyEight-style): a 4-0 moves
            # ratings more than a 1-0, with diminishing returns, and the
            # correction for rating difference stops runaway inflation.
            margin = abs(hg - ag)
            mov = math.log(margin + 1.0) * (2.2 / (0.001 * abs(diff) + 2.2))
            delta = self.k_factor * mov * (score - expected_home)

            self.ratings[m.home_team_id] = home + delta
            self.ratings[m.away_team_id] = away - delta
            self.match_counts[m.home_team_id] = self.match_counts.get(m.home_team_id, 0) + 1
            self.match_counts[m.away_team_id] = self.match_counts.get(m.away_team_id, 0) + 1

        self._fit_draw_curve(gaps_and_draws)
        self.fitted = True
        self.fit_diagnostics = {
            "matches": len(matches),
            "teams": len(self.ratings),
            "draw_base": round(self._draw_base, 4),
            "draw_decay": round(self._draw_decay, 6),
            "rating_spread": round(float(np.std(list(self.ratings.values()))), 1),
            "top_rating": round(max(self.ratings.values()), 1) if self.ratings else None,
        }
        return self

    def _fit_draw_curve(self, observations: list[tuple[float, int]]) -> None:
        """Estimate P(draw) as a function of |rating gap| from the data.

        Bucketed empirical rates fitted with a simple exponential decay - the
        draw rate is highest between evenly matched sides and falls as the gap
        widens. Falls back to the training-set average if the fit is degenerate.
        """
        if len(observations) < 100:
            return
        gaps = np.array([g for g, _ in observations])
        draws = np.array([d for _, d in observations], dtype=float)
        overall = float(draws.mean())
        edges = np.quantile(gaps, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (gaps >= lo) & (gaps <= hi)
            if mask.sum() >= 30:
                xs.append(float(gaps[mask].mean()))
                ys.append(float(draws[mask].mean()))
        if len(xs) < 3:
            self._draw_base = overall
            return
        xs_arr, ys_arr = np.array(xs), np.clip(np.array(ys), 1e-3, 0.999)
        # log(p) = log(base) - decay * gap
        slope, intercept = np.polyfit(xs_arr, np.log(ys_arr), 1)
        base = float(np.exp(intercept))
        decay = float(-slope)
        if not (0.05 <= base <= 0.6) or not (0.0 <= decay <= 0.02):
            self._draw_base = overall
            return
        self._draw_base, self._draw_decay = base, decay

    def draw_probability(self, rating_diff: float) -> float:
        p = self._draw_base * math.exp(-self._draw_decay * abs(rating_diff))
        return float(min(max(p, 0.05), 0.40))

    def predict(self, request: PredictionRequest) -> ModelPrediction:
        if not self.fitted:
            return self._insufficient(self.fit_diagnostics.get("error", "model is not fitted"))
        for tid in (request.home_team_id, request.away_team_id):
            if tid not in self.ratings:
                return self._insufficient("no Elo rating for one or both teams")
            if self.match_counts.get(tid, 0) < 4:
                return self._insufficient("fewer than 4 rated matches for one of the teams")

        home = self.ratings[request.home_team_id]
        away = self.ratings[request.away_team_id]
        advantage = 0.0 if request.neutral_venue else self.home_advantage
        diff = home + advantage - away

        expected_home = self._expected(diff)
        p_draw = self.draw_probability(diff)
        # Split the non-draw mass in proportion to the win expectancy.
        p_home = (1.0 - p_draw) * expected_home
        p_away = (1.0 - p_draw) * (1.0 - expected_home)
        total = p_home + p_draw + p_away

        return ModelPrediction(
            model_key=self.key,
            sufficient_data=True,
            outcome_probs={"home": p_home / total, "draw": p_draw / total, "away": p_away / total},
            diagnostics={
                "home_rating": round(home, 1),
                "away_rating": round(away, 1),
                "rating_difference": round(home - away, 1),
                "home_advantage_points": advantage,
                "win_expectancy": round(expected_home, 4),
            },
        )

    def rating_of(self, team_id: int) -> Optional[float]:
        return self.ratings.get(team_id)
