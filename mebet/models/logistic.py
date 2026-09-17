"""Multinomial logistic regression over engineered features.

Where Dixon-Coles sees only goals and Elo only results, this model can use
the wider feature set - shot volume, shot quality, opponent-adjusted output,
rest, congestion, home/away splits - and learn how much each is worth.

Feature construction for training rows is deliberately the same code path as
for prediction: each training row is built from a repository whose cutoff is
the kickoff of that very match. It is slower than vectorising over the whole
table, but it guarantees the training features are exactly what would have
been visible beforehand.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..db.repository import AsOfRepository
from ..logging_setup import get_logger
from .base import ModelPrediction, PredictionModel, PredictionRequest, TrainingContext

log = get_logger("models.logistic")

#: Ordered feature names; the order is fixed so coefficients stay interpretable.
FEATURE_NAMES = (
    "home_gf_10", "home_ga_10", "away_gf_10", "away_ga_10",
    "home_gf_home", "home_ga_home", "away_gf_away", "away_ga_away",
    "home_sot_10", "away_sot_10", "home_sot_against_10", "away_sot_against_10",
    "home_ppg_5", "away_ppg_5", "home_ppg_all", "away_ppg_all",
    "home_attack_index", "away_attack_index", "home_defence_index", "away_defence_index",
    "home_rest_advantage_days", "congestion_difference", "position_difference",
)


def _safe(value: Optional[float], default: float = 0.0) -> float:
    return default if value is None else float(value)


def feature_vector(features) -> np.ndarray:
    """Turn a ``MatchFeatures`` object into the fixed-order model input."""
    h, a = features.home, features.away
    ctx = features.context

    def split(form, side, name):
        table = form.home_splits if side == "home" else form.away_splits
        stat = table.get(name)
        return stat.value if stat and stat.is_usable else None

    position_diff = None
    if h.league_position is not None and a.league_position is not None:
        position_diff = a.league_position - h.league_position

    values = [
        _safe(h.stat("last10", "goals_for"), 1.3),
        _safe(h.stat("last10", "goals_against"), 1.3),
        _safe(a.stat("last10", "goals_for"), 1.3),
        _safe(a.stat("last10", "goals_against"), 1.3),
        _safe(split(h, "home", "goals_for"), _safe(h.stat("all", "goals_for"), 1.3)),
        _safe(split(h, "home", "goals_against"), _safe(h.stat("all", "goals_against"), 1.3)),
        _safe(split(a, "away", "goals_for"), _safe(a.stat("all", "goals_for"), 1.3)),
        _safe(split(a, "away", "goals_against"), _safe(a.stat("all", "goals_against"), 1.3)),
        _safe(h.stat("last10", "shots_on_target"), 4.5),
        _safe(a.stat("last10", "shots_on_target"), 4.5),
        _safe(h.stat("last10", "sot_against"), 4.5),
        _safe(a.stat("last10", "sot_against"), 4.5),
        _safe(h.form_points.get("last5"), 1.4),
        _safe(a.form_points.get("last5"), 1.4),
        _safe(h.form_points.get("all"), 1.4),
        _safe(a.form_points.get("all"), 1.4),
        _safe(h.opponent_adjusted.get("attack_index"), 1.0),
        _safe(a.opponent_adjusted.get("attack_index"), 1.0),
        _safe(h.opponent_adjusted.get("defence_index"), 1.0),
        _safe(a.opponent_adjusted.get("defence_index"), 1.0),
        _safe(ctx.get("home_rest_advantage_days"), 0.0),
        _safe(h.matches_last_14_days) - _safe(a.matches_last_14_days),
        _safe(position_diff, 0.0),
    ]
    return np.array(values, dtype=float)


class LogisticOutcomeModel(PredictionModel):
    key = "logistic"
    name = "Multinomial logistic regression on form features"
    produces = ("1x2",)
    yields_score_matrix = False

    def __init__(self, min_matches: int = 200, max_training_rows: int = 1200,
                 C: float = 0.35, half_life_days: float = 365.0) -> None:
        super().__init__(min_matches=min_matches, max_training_rows=max_training_rows, C=C,
                         half_life_days=half_life_days)
        self.min_matches = min_matches
        self.max_training_rows = max_training_rows
        self.C = C
        self.half_life_days = half_life_days
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.classes_: list[str] = []

    def fit(self, ctx: TrainingContext) -> "LogisticOutcomeModel":
        from ..features.football import build_match_features  # local import avoids a cycle

        matches = [
            m for m in ctx.training_matches()
            if m.ft_home_goals is not None and m.ft_away_goals is not None
        ]
        matches.sort(key=lambda m: (m.kickoff_date, m.id), reverse=True)
        # Most recent rows first, capped: building features per row is the
        # expensive part, and older rows contribute little after decay.
        matches = matches[: self.max_training_rows]
        self.training_cutoff = ctx.as_of
        self.n_training_matches = len(matches)

        if len(matches) < self.min_matches:
            self.fitted = False
            self.fit_diagnostics = {
                "error": f"only {len(matches)} training matches, need {self.min_matches}"
            }
            return self

        rows, labels, weights = [], [], []
        session = ctx.repo.session
        as_of_date = ctx.as_of.date()
        for m in matches:
            # Cutoff = this match's own kickoff, so its result is invisible.
            cutoff = m.kickoff_utc or dt.datetime.combine(m.kickoff_date, dt.time(0, 0))
            row_repo = AsOfRepository(session, cutoff, sport=ctx.repo.sport,
                                      index=ctx.index or ctx.repo.index)
            try:
                feats = build_match_features(row_repo, m.home_team_id, m.away_team_id,
                                             m.competition_id)
            except Exception as exc:  # noqa: BLE001
                log.debug("feature build failed for match %s: %s", m.id, exc)
                continue
            if feats.home.matches_available < 5 or feats.away.matches_available < 5:
                continue
            rows.append(feature_vector(feats))
            labels.append(m.result)
            age = (as_of_date - m.kickoff_date).days
            weights.append(0.5 ** (max(age, 0) / self.half_life_days)
                           if self.half_life_days > 0 else 1.0)

        if len(rows) < self.min_matches:
            self.fitted = False
            self.fit_diagnostics = {
                "error": f"only {len(rows)} usable feature rows, need {self.min_matches}"
            }
            return self

        X = np.vstack(rows)
        y = np.array(labels)
        w = np.array(weights)

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        # lbfgs fits the multinomial objective by default in current scikit-learn.
        self.model = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")
        self.model.fit(Xs, y, sample_weight=w)
        self.classes_ = list(self.model.classes_)
        self.fitted = True
        self.fit_diagnostics = {
            "rows": len(rows),
            "classes": self.classes_,
            "class_balance": {c: int((y == c).sum()) for c in self.classes_},
            "top_features": self._top_features(),
        }
        return self

    def _top_features(self, k: int = 6) -> dict:
        if self.model is None:
            return {}
        out = {}
        for idx, cls in enumerate(self.classes_):
            coefs = self.model.coef_[idx]
            order = np.argsort(np.abs(coefs))[::-1][:k]
            out[cls] = [
                {"feature": FEATURE_NAMES[i], "coefficient": round(float(coefs[i]), 4)}
                for i in order if i < len(FEATURE_NAMES)
            ]
        return out

    def predict(self, request: PredictionRequest) -> ModelPrediction:
        if not self.fitted or self.model is None or self.scaler is None:
            return self._insufficient(self.fit_diagnostics.get("error", "model is not fitted"))
        if request.features is None:
            return self._insufficient("feature set was not supplied to the logistic model")
        feats = request.features
        if feats.home.matches_available < 5 or feats.away.matches_available < 5:
            return self._insufficient("fewer than 5 prior matches for one of the teams")

        x = self.scaler.transform(feature_vector(feats).reshape(1, -1))
        probs = self.model.predict_proba(x)[0]
        mapping = {"H": "home", "D": "draw", "A": "away"}
        outcome = {mapping[c]: float(p) for c, p in zip(self.classes_, probs) if c in mapping}
        for key in ("home", "draw", "away"):
            outcome.setdefault(key, 0.0)
        total = sum(outcome.values())
        outcome = {k: v / total for k, v in outcome.items()}

        return ModelPrediction(
            model_key=self.key,
            sufficient_data=True,
            outcome_probs=outcome,
            diagnostics={"training_rows": self.fit_diagnostics.get("rows"),
                         "feature_count": len(FEATURE_NAMES)},
        )
