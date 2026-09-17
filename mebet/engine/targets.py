"""Deriving prediction targets from model output.

Everything here is a consequence of the distributions the models produced.
Over/under, both-teams-to-score, clean sheets and correct score all come from
the same scoreline matrix, so they cannot contradict one another.

Where the underlying data does not support a market, the market is emitted
with ``sufficient_data=False`` and an explanation. The brief is explicit that
"insufficient reliable data" is a better answer than a fabricated number, so
the market still appears - the user is told it was considered and why it was
withheld.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..logging_setup import get_logger

log = get_logger("engine.targets")

#: Total-goals lines to quote.
GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
TEAM_GOAL_LINES = (0.5, 1.5, 2.5)
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)


@dataclass
class Target:
    market: str
    selection: str
    probability: Optional[float] = None
    expected_value: Optional[float] = None
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    model_key: str = ""
    confidence: Optional[float] = None
    sufficient_data: bool = True
    note: str = ""
    subject_type: str = "match"
    subject_id: Optional[int] = None
    subject_name: str = ""

    def as_dict(self) -> dict:
        return {
            "market": self.market,
            "selection": self.selection,
            "probability": round(self.probability, 4) if self.probability is not None else None,
            "expected_value": (
                round(self.expected_value, 3) if self.expected_value is not None else None
            ),
            "interval": (
                [self.interval_low, self.interval_high]
                if self.interval_low is not None else None
            ),
            "model": self.model_key,
            "confidence": round(self.confidence, 3) if self.confidence is not None else None,
            "sufficient_data": self.sufficient_data,
            "note": self.note,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
        }


def unavailable(market: str, reason: str, selection: str = "-") -> Target:
    return Target(market=market, selection=selection, sufficient_data=False, note=reason)


# ---------------------------------------------------------------------------
# Match result and scorelines
# ---------------------------------------------------------------------------
def outcome_targets(probs: dict[str, float], model_key: str, confidence: float,
                    home_name: str, away_name: str) -> list[Target]:
    labels = {"home": home_name, "draw": "Draw", "away": away_name}
    out = []
    for selection in ("home", "draw", "away"):
        out.append(Target(
            market="1x2", selection=selection, probability=probs[selection],
            model_key=model_key, confidence=confidence, subject_name=labels[selection],
        ))
    # Double chance follows directly and is often the more useful statement
    # when a match is close.
    out.append(Target("double_chance", "home_or_draw", probs["home"] + probs["draw"],
                      model_key=model_key, confidence=confidence))
    out.append(Target("double_chance", "away_or_draw", probs["away"] + probs["draw"],
                      model_key=model_key, confidence=confidence))
    out.append(Target("double_chance", "home_or_away", probs["home"] + probs["away"],
                      model_key=model_key, confidence=confidence))
    return out


def scoreline_targets(matrix: np.ndarray, model_key: str, confidence: float,
                      top_n: int = 6) -> list[Target]:
    flat = [
        (matrix[h, a], h, a)
        for h in range(matrix.shape[0])
        for a in range(matrix.shape[1])
    ]
    flat.sort(reverse=True)
    out = []
    for prob, h, a in flat[:top_n]:
        out.append(Target(
            market="correct_score", selection=f"{h}-{a}", probability=float(prob),
            model_key=model_key, confidence=confidence,
        ))

    # The single most likely scoreline is often a draw even when one side is
    # clearly favoured, because the favourite's probability is spread over many
    # scorelines. Quoting the likeliest scoreline *within* each outcome makes
    # the headline and the scoreline consistent rather than seemingly at odds.
    best = {"home": None, "draw": None, "away": None}
    for prob, h, a in flat:
        key = "home" if h > a else ("away" if a > h else "draw")
        if best[key] is None:
            best[key] = (prob, h, a)
        if all(v is not None for v in best.values()):
            break
    for key, entry in best.items():
        if entry is None:
            continue
        prob, h, a = entry
        out.append(Target(
            market="most_likely_score_given", selection=key,
            probability=float(prob), model_key=model_key, confidence=confidence,
            note=f"{h}-{a}", subject_name=f"{h}-{a}",
        ))
    return out


def goals_targets(matrix: np.ndarray, model_key: str, confidence: float,
                  home_name: str, away_name: str) -> list[Target]:
    goals = np.arange(matrix.shape[0])
    home_margin = matrix.sum(axis=1)
    away_margin = matrix.sum(axis=0)
    exp_home = float((home_margin * goals).sum())
    exp_away = float((away_margin * goals).sum())

    totals = np.zeros(matrix.shape[0] + matrix.shape[1] - 1)
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            totals[h + a] += matrix[h, a]

    out = [
        Target("expected_goals", "total", expected_value=round(exp_home + exp_away, 3),
               model_key=model_key, confidence=confidence),
        Target("expected_goals", "home", expected_value=round(exp_home, 3),
               model_key=model_key, confidence=confidence, subject_name=home_name),
        Target("expected_goals", "away", expected_value=round(exp_away, 3),
               model_key=model_key, confidence=confidence, subject_name=away_name),
    ]

    for line in GOAL_LINES:
        over = float(totals[int(np.ceil(line)):].sum())
        out.append(Target(f"over_under_{line}", "over", over, model_key=model_key,
                          confidence=confidence))
        out.append(Target(f"over_under_{line}", "under", 1.0 - over, model_key=model_key,
                          confidence=confidence))

    # Both teams to score, and clean sheets, read straight off the matrix.
    btts = float(matrix[1:, 1:].sum())
    out.append(Target("btts", "yes", btts, model_key=model_key, confidence=confidence))
    out.append(Target("btts", "no", 1.0 - btts, model_key=model_key, confidence=confidence))
    out.append(Target("clean_sheet", "home", float(matrix[:, 0].sum()), model_key=model_key,
                      confidence=confidence, subject_name=home_name))
    out.append(Target("clean_sheet", "away", float(matrix[0, :].sum()), model_key=model_key,
                      confidence=confidence, subject_name=away_name))

    for line in TEAM_GOAL_LINES:
        idx = int(np.ceil(line))
        out.append(Target(f"home_goals_over_under_{line}", "over",
                          float(home_margin[idx:].sum()), model_key=model_key,
                          confidence=confidence, subject_name=home_name))
        out.append(Target(f"away_goals_over_under_{line}", "over",
                          float(away_margin[idx:].sum()), model_key=model_key,
                          confidence=confidence, subject_name=away_name))

    # Winning margin, a genuinely different question from the result.
    margins: dict[int, float] = {}
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            margins[h - a] = margins.get(h - a, 0.0) + float(matrix[h, a])
    for margin in (1, 2, 3):
        out.append(Target("winning_margin", f"home_by_{margin}", margins.get(margin, 0.0),
                          model_key=model_key, confidence=confidence, subject_name=home_name))
        out.append(Target("winning_margin", f"away_by_{margin}", margins.get(-margin, 0.0),
                          model_key=model_key, confidence=confidence, subject_name=away_name))
    return out


def first_half_targets(matrix: np.ndarray, home_share: Optional[float],
                       away_share: Optional[float], model_key: str,
                       confidence: float) -> list[Target]:
    """First-half goals, using each team's measured first-half goal share.

    Only produced when both teams have recorded half-time scores; otherwise
    the share would be a guess.
    """
    if home_share is None or away_share is None:
        return [unavailable(
            "first_half_goals",
            "half-time scores are not recorded for one or both teams in this dataset",
        )]
    goals = np.arange(matrix.shape[0])
    exp_home = float((matrix.sum(axis=1) * goals).sum()) * home_share
    exp_away = float((matrix.sum(axis=0) * goals).sum()) * away_share
    total = exp_home + exp_away
    from scipy.stats import poisson

    out = [Target("expected_first_half_goals", "total", expected_value=round(total, 3),
                  model_key=model_key, confidence=confidence * 0.85)]
    for line in (0.5, 1.5):
        over = float(1.0 - poisson.cdf(int(np.floor(line)), max(total, 1e-6)))
        out.append(Target(f"first_half_over_under_{line}", "over", over, model_key=model_key,
                          confidence=confidence * 0.85))
    return out


# ---------------------------------------------------------------------------
# Count markets
# ---------------------------------------------------------------------------
def count_targets(prediction, market: str, lines: Sequence[float], confidence: float,
                  home_name: str, away_name: str) -> list[Target]:
    if prediction is None or not prediction.sufficient_data:
        reason = (prediction.note if prediction is not None
                  else f"no {market} model available")
        return [unavailable(market, reason)]

    dist = prediction.total_distribution
    if dist is None:
        return [unavailable(market, f"{market} model produced no distribution")]

    exp_home = prediction.expected_home_count
    exp_away = prediction.expected_away_count
    total = (exp_home or 0) + (exp_away or 0)

    # An 80% central interval communicates spread honestly; a single expected
    # value invites false precision.
    cumulative = np.cumsum(dist)
    low = int(np.searchsorted(cumulative, 0.10))
    high = int(np.searchsorted(cumulative, 0.90))

    out = [
        Target(f"expected_{market}", "total", expected_value=round(total, 2),
               interval_low=low, interval_high=high,
               model_key=prediction.model_key, confidence=confidence),
        Target(f"expected_{market}", "home", expected_value=exp_home,
               model_key=prediction.model_key, confidence=confidence, subject_name=home_name),
        Target(f"expected_{market}", "away", expected_value=exp_away,
               model_key=prediction.model_key, confidence=confidence, subject_name=away_name),
    ]
    for line in lines:
        threshold = int(np.ceil(line))
        over = float(dist[threshold:].sum()) if threshold < len(dist) else 0.0
        out.append(Target(f"{market}_over_under_{line}", "over", over,
                          model_key=prediction.model_key, confidence=confidence))
        out.append(Target(f"{market}_over_under_{line}", "under", 1.0 - over,
                          model_key=prediction.model_key, confidence=confidence))
    return out


# ---------------------------------------------------------------------------
# Player markets
# ---------------------------------------------------------------------------
def player_targets(projections: Sequence, confidence: float) -> list[Target]:
    out: list[Target] = []
    if not projections:
        return [unavailable("player_goals",
                            "no player-level statistics are available for these teams")]
    for proj in projections:
        if not proj.sufficient_data:
            out.append(Target(
                market="player_goals", selection="anytime", sufficient_data=False,
                note=proj.note, subject_type="player", subject_id=proj.player_id,
                subject_name=proj.player_name,
            ))
            continue
        base = {
            "subject_type": "player",
            "subject_id": proj.player_id,
            "subject_name": proj.player_name,
            "model_key": "player_rates",
            "confidence": confidence * 0.8,   # player markets are noisier
        }
        out.append(Target(market="player_goals", selection="anytime",
                          probability=proj.prob_scores,
                          expected_value=proj.expected_goals, **base))
        out.append(Target(market="player_goals", selection="two_or_more",
                          probability=proj.prob_two_plus, **base))
        out.append(Target(market="player_assists", selection="anytime",
                          probability=proj.prob_assists,
                          expected_value=proj.expected_assists, **base))
        out.append(Target(market="player_involvement", selection="goal_or_assist",
                          probability=proj.prob_involvement, **base))
        out.append(Target(market="player_minutes", selection="expected",
                          expected_value=proj.expected_minutes, **base))
    # Markets the reachable player feed does not support are declared, not faked.
    out.append(unavailable(
        "player_shots",
        "the available player feed publishes minutes, goals, assists and expected "
        "goals, but not shot counts; a shots line would be an invention",
    ))
    out.append(unavailable(
        "player_key_passes",
        "key passes are not published by the available player feed",
    ))
    return out
