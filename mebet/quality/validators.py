"""Validation rules feeding the quality assessment.

These check what was retrieved, not whether the prediction looks plausible.
A model that confidently predicts a 4-0 is not a data problem; a league whose
corner counts were never recorded is.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence

from ..db.repository import AsOfRepository, TeamMatchRow
from ..logging_setup import get_logger
from .assessment import DataQualityReport, QualityTier

log = get_logger("quality.validators")

#: Sample-size thresholds, per team, for a window to be treated as meaningful.
MIN_MATCHES_STRONG = 20
MIN_MATCHES_USABLE = 10
MIN_MATCHES_WEAK = 5

#: Plausible per-team, per-match ranges. Values outside these are suspicious
#: and are flagged rather than silently modelled.
PLAUSIBLE_RANGES = {
    "goals_for": (0, 12),
    "shots": (0, 50),
    "shots_on_target": (0, 30),
    "corners": (0, 25),
    "fouls": (0, 40),
    "yellows": (0, 12),
    "reds": (0, 5),
}


def check_sample_sizes(report: DataQualityReport, home_rows: int, away_rows: int,
                       h2h: int) -> None:
    report.home_matches = home_rows
    report.away_matches = away_rows
    report.historical_matches = home_rows + away_rows
    report.h2h_matches = h2h

    for label, n in (("home", home_rows), ("away", away_rows)):
        if n == 0:
            report.add("no_history", "serious",
                       f"no completed matches on record for the {label} team", side=label)
        elif n < MIN_MATCHES_WEAK:
            report.add("tiny_sample", "serious",
                       f"only {n} prior matches for the {label} team; estimates are unstable",
                       side=label, matches=n)
        elif n < MIN_MATCHES_USABLE:
            report.add("small_sample", "warning",
                       f"only {n} prior matches for the {label} team", side=label, matches=n)

    if h2h == 0:
        report.add("no_h2h", "info",
                   "these teams have no recorded previous meetings")
    elif h2h < 3:
        report.add("thin_h2h", "info",
                   f"only {h2h} previous meeting(s); too few to carry weight",
                   matches=h2h)


def check_statistic_coverage(report: DataQualityReport, rows: Sequence[TeamMatchRow],
                             *, label: str = "") -> dict:
    """What fraction of matches actually recorded each statistic."""
    if not rows:
        return {}
    coverage = {}
    for stat in ("shots", "shots_on_target", "corners", "fouls", "yellows", "xg"):
        present = sum(1 for r in rows if getattr(r, stat, None) is not None)
        coverage[stat] = round(present / len(rows), 3)
    for stat, fraction in coverage.items():
        if fraction == 0.0:
            where = f" for the {label} team" if label else ""
            report.add(f"missing_{stat}_{label}", "warning",
                       f"{stat.replace('_', ' ')} was never recorded in this sample"
                       f"{where}; dependent predictions are withheld",
                       statistic=stat, side=label)
        elif fraction < 0.5:
            report.add(f"sparse_{stat}", "info",
                       f"{stat.replace('_', ' ')} recorded for only "
                       f"{fraction:.0%} of matches in this sample",
                       statistic=stat, coverage=fraction, side=label)
    return coverage


def check_suspicious_values(report: DataQualityReport, rows: Sequence[TeamMatchRow]) -> int:
    """Flag values outside physically plausible ranges."""
    flagged = 0
    for row in rows:
        for stat, (low, high) in PLAUSIBLE_RANGES.items():
            value = getattr(row, stat, None)
            if value is None:
                continue
            if value < low or value > high:
                flagged += 1
                report.add("suspicious_value", "warning",
                           f"implausible {stat} value {value} recorded on {row.date}",
                           statistic=stat, value=value, date=row.date.isoformat(),
                           match_id=row.match_id)
    return flagged


def check_freshness(report: DataQualityReport, rows: Sequence[TeamMatchRow],
                    as_of: dt.datetime, *, kickoff: Optional[dt.datetime] = None) -> None:
    """How stale is the most recent data relative to the prediction date?"""
    if not rows:
        return
    latest = max(r.date for r in rows)
    age = (as_of.date() - latest).days
    report.freshness_days = float(age)
    # Outside an international break, a month without a recorded match means
    # the feed has probably stopped updating.
    if age > 45:
        report.add("stale_data", "serious",
                   f"most recent recorded match is {age} days old; the feed may be behind",
                   days=age, latest=latest.isoformat())
    elif age > 21:
        report.add("ageing_data", "warning",
                   f"most recent recorded match is {age} days old", days=age)

    if kickoff is not None:
        lead = (kickoff.date() - as_of.date()).days
        if lead > 10:
            report.add("distant_fixture", "info",
                       f"kickoff is {lead} days away; team news and lineups will change",
                       days_to_kickoff=lead)


def check_lineups(report: DataQualityReport, lineups: Sequence, kickoff: dt.datetime,
                  as_of: dt.datetime) -> None:
    confirmed = [l for l in lineups if l.kind == "confirmed"]
    predicted = [l for l in lineups if l.kind == "predicted"]
    if confirmed:
        report.lineups = "confirmed"
        return
    if predicted:
        report.lineups = "predicted"
        report.add("lineups_predicted", "warning",
                   "lineups are projected, not confirmed; a rotation would change the picture")
        return
    report.lineups = "unavailable"
    hours_to_kickoff = (kickoff - as_of).total_seconds() / 3600 if kickoff else None
    if hours_to_kickoff is not None and hours_to_kickoff <= 2:
        report.add("lineups_missing_late", "serious",
                   "kickoff is imminent but no lineup was retrieved from any source")
    else:
        report.add("lineups_missing", "warning",
                   "no lineup information available from the configured sources")


def check_conflicts(report: DataQualityReport, conflicts: Sequence[str]) -> None:
    for message in conflicts:
        report.add("source_conflict", "warning", message)


def finalise(report: DataQualityReport) -> DataQualityReport:
    """Turn the issue list into a tier, a score and a confidence multiplier."""
    serious = sum(1 for i in report.issues if i.severity == "serious")
    warnings = sum(1 for i in report.issues if i.severity == "warning")

    score = 1.0
    # Sample size dominates: it is what actually determines whether the
    # estimates mean anything.
    smaller_sample = min(report.home_matches, report.away_matches)
    if smaller_sample == 0:
        score = 0.0
    elif smaller_sample < MIN_MATCHES_WEAK:
        score *= 0.35
    elif smaller_sample < MIN_MATCHES_USABLE:
        score *= 0.6
    elif smaller_sample < MIN_MATCHES_STRONG:
        score *= 0.85

    score *= max(0.3, 1.0 - 0.15 * serious)
    score *= max(0.5, 1.0 - 0.05 * warnings)

    if len(report.sources_used) >= 3:
        score *= 1.05
    elif len(report.sources_used) <= 1:
        score *= 0.9

    if report.lineups == "confirmed":
        score *= 1.08
    elif report.lineups == "unavailable":
        score *= 0.95

    if not report.availability_data:
        score *= 0.92

    if report.freshness_days is not None and report.freshness_days > 45:
        score *= 0.85

    report.score = float(min(max(score, 0.0), 1.0))

    if report.score <= 0.0 or smaller_sample == 0:
        report.tier = QualityTier.INSUFFICIENT
    elif report.score >= 0.8 and serious == 0:
        report.tier = QualityTier.HIGH
    elif report.score >= 0.55:
        report.tier = QualityTier.MEDIUM
    else:
        report.tier = QualityTier.LOW

    # Confidence scales with quality but is never zeroed for a usable tier.
    report.confidence_multiplier = round(0.35 + 0.65 * report.score, 3)
    return report
