"""Relational schema.

Two principles shape this schema:

*   **Provenance is not optional.** Every table that holds an observed fact
    carries ``source_id`` and ``retrieved_at``. A row that exists without a
    source could not have come from a real retrieval, so the ingestion layer
    cannot create one.
*   **Sport is a dimension, not an assumption.** Core entities carry a
    ``sport`` key and sport-specific measurements live in their own tables,
    so basketball/tennis/cricket adapters can be added without reshaping the
    core.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
class DataSource(Base):
    """A distinct external provider of data."""

    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    homepage: Mapped[str] = mapped_column(String(500), default="")
    license: Mapped[str] = mapped_column(String(200), default="")
    # 1.0 = primary/official structured feed, lower = derived or best-effort.
    reliability: Mapped[float] = mapped_column(Float, default=0.5)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class FetchLog(Base):
    """One row per outbound retrieval attempt - successful or not.

    This is what makes "the source was unavailable" an auditable statement
    rather than a claim.
    """

    __tablename__ = "fetch_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id"), index=True)
    url: Mapped[str] = mapped_column(String(1000))
    requested_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    from_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    content_sha256: Mapped[str] = mapped_column(String(64), default="")


# --------------------------------------------------------------------------
# Core entities
# --------------------------------------------------------------------------
class Competition(Base):
    __tablename__ = "competitions"
    __table_args__ = (UniqueConstraint("sport", "key", name="uq_competition_sport_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(32), index=True, default="football")
    key: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    country: Mapped[str] = mapped_column(String(80), default="")
    tier: Mapped[Optional[int]] = mapped_column(Integer)
    # "league" | "cup" | "tournament" - drives context features.
    format: Mapped[str] = mapped_column(String(32), default="league")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("competition_id", "label", name="uq_season_comp_label"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    label: Mapped[str] = mapped_column(String(32))  # e.g. "2025-26"
    start_date: Mapped[Optional[dt.date]] = mapped_column(Date)
    end_date: Mapped[Optional[dt.date]] = mapped_column(Date)

    competition: Mapped[Competition] = relationship()


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("sport", "canonical_name", name="uq_team_sport_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(32), index=True, default="football")
    canonical_name: Mapped[str] = mapped_column(String(200), index=True)
    short_name: Mapped[str] = mapped_column(String(80), default="")
    country: Mapped[str] = mapped_column(String(80), default="")
    external_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class TeamAlias(Base):
    """Name variants seen in the wild, mapped onto a canonical team."""

    __tablename__ = "team_aliases"
    __table_args__ = (UniqueConstraint("sport", "alias_norm", name="uq_alias_sport_norm"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(32), index=True, default="football")
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    alias: Mapped[str] = mapped_column(String(200))
    alias_norm: Mapped[str] = mapped_column(String(200), index=True)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id"))


class Venue(Base):
    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    city: Mapped[str] = mapped_column(String(120), default="")
    country: Mapped[str] = mapped_column(String(80), default="")
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    home_team_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teams.id"))
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id"))


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("sport", "full_name", "team_id", name="uq_player_sport_name_team"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(32), index=True, default="football")
    full_name: Mapped[str] = mapped_column(String(200), index=True)
    team_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teams.id"), index=True)
    position: Mapped[str] = mapped_column(String(32), default="")
    external_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id"))
    retrieved_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)


# --------------------------------------------------------------------------
# Matches and measurements
# --------------------------------------------------------------------------
class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("sport", "competition_id", "kickoff_date", "home_team_id", "away_team_id",
                         name="uq_match_identity"),
        Index("ix_match_kickoff", "kickoff_utc"),
        Index("ix_match_teams", "home_team_id", "away_team_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(32), index=True, default="football")
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    season_id: Mapped[Optional[int]] = mapped_column(ForeignKey("seasons.id"), index=True)

    # kickoff_utc may be unknown (older CSV rows carry a date only); kickoff_date
    # is always populated and is what as-of filtering falls back to.
    kickoff_utc: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    kickoff_date: Mapped[dt.date] = mapped_column(Date, index=True)
    kickoff_is_exact: Mapped[bool] = mapped_column(Boolean, default=False)

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    venue_id: Mapped[Optional[int]] = mapped_column(ForeignKey("venues.id"))

    status: Mapped[str] = mapped_column(String(24), default="scheduled", index=True)  # scheduled|played|postponed
    stage: Mapped[str] = mapped_column(String(64), default="")
    matchday: Mapped[Optional[int]] = mapped_column(Integer)
    neutral_venue: Mapped[bool] = mapped_column(Boolean, default=False)

    ft_home_goals: Mapped[Optional[int]] = mapped_column(Integer)
    ft_away_goals: Mapped[Optional[int]] = mapped_column(Integer)
    ht_home_goals: Mapped[Optional[int]] = mapped_column(Integer)
    ht_away_goals: Mapped[Optional[int]] = mapped_column(Integer)
    referee: Mapped[str] = mapped_column(String(120), default="")

    external_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id"), index=True)
    retrieved_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)

    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    competition: Mapped[Competition] = relationship()

    @property
    def result(self) -> Optional[str]:
        if self.ft_home_goals is None or self.ft_away_goals is None:
            return None
        if self.ft_home_goals > self.ft_away_goals:
            return "H"
        if self.ft_home_goals < self.ft_away_goals:
            return "A"
        return "D"


class TeamMatchStats(Base):
    """Per-team measurements for one match. NULL means 'not observed'."""

    __tablename__ = "team_match_stats"
    __table_args__ = (UniqueConstraint("match_id", "team_id", "source_id", name="uq_tms_match_team_source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean)

    goals: Mapped[Optional[int]] = mapped_column(Integer)
    goals_conceded: Mapped[Optional[int]] = mapped_column(Integer)
    ht_goals: Mapped[Optional[int]] = mapped_column(Integer)
    shots: Mapped[Optional[int]] = mapped_column(Integer)
    shots_on_target: Mapped[Optional[int]] = mapped_column(Integer)
    corners: Mapped[Optional[int]] = mapped_column(Integer)
    fouls: Mapped[Optional[int]] = mapped_column(Integer)
    yellow_cards: Mapped[Optional[int]] = mapped_column(Integer)
    red_cards: Mapped[Optional[int]] = mapped_column(Integer)
    possession: Mapped[Optional[float]] = mapped_column(Float)
    xg: Mapped[Optional[float]] = mapped_column(Float)
    xga: Mapped[Optional[float]] = mapped_column(Float)

    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id"), index=True)
    retrieved_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"
    __table_args__ = (UniqueConstraint("match_id", "player_id", "source_id", name="uq_pms_match_player_source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[Optional[int]] = mapped_column(ForeignKey("matches.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    team_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teams.id"), index=True)
    # Denormalised for as-of filtering without a join.
    match_date: Mapped[dt.date] = mapped_column(Date, index=True)

    minutes: Mapped[Optional[int]] = mapped_column(Integer)
    started: Mapped[Optional[bool]] = mapped_column(Boolean)
    goals: Mapped[Optional[int]] = mapped_column(Integer)
    assists: Mapped[Optional[int]] = mapped_column(Integer)
    shots: Mapped[Optional[int]] = mapped_column(Integer)
    shots_on_target: Mapped[Optional[int]] = mapped_column(Integer)
    key_passes: Mapped[Optional[int]] = mapped_column(Integer)
    xg: Mapped[Optional[float]] = mapped_column(Float)
    xa: Mapped[Optional[float]] = mapped_column(Float)
    xgc: Mapped[Optional[float]] = mapped_column(Float)
    tackles: Mapped[Optional[int]] = mapped_column(Integer)
    saves: Mapped[Optional[int]] = mapped_column(Integer)
    yellow_cards: Mapped[Optional[int]] = mapped_column(Integer)
    red_cards: Mapped[Optional[int]] = mapped_column(Integer)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id"), index=True)
    retrieved_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class PlayerAvailability(Base):
    """Injury / suspension / doubt records, valid as of a point in time."""

    __tablename__ = "player_availability"
    __table_args__ = (Index("ix_avail_player_asof", "player_id", "observed_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    team_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teams.id"), index=True)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    # available | doubtful | injured | suspended | unavailable | unknown
    status: Mapped[str] = mapped_column(String(24), default="unknown")
    reason: Mapped[str] = mapped_column(Text, default="")
    chance_of_playing: Mapped[Optional[float]] = mapped_column(Float)
    expected_return: Mapped[Optional[dt.date]] = mapped_column(Date)
    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id"), index=True)
    retrieved_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class Lineup(Base):
    """Predicted or confirmed lineups. ``kind`` distinguishes them so the
    engine never mistakes a projection for a confirmation."""

    __tablename__ = "lineups"
    __table_args__ = (
        UniqueConstraint("match_id", "team_id", "player_id", "kind", "source_id", name="uq_lineup_row"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="predicted")  # predicted|confirmed
    is_starter: Mapped[bool] = mapped_column(Boolean, default=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id"), index=True)
    retrieved_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class MatchConditions(Base):
    __tablename__ = "match_conditions"
    __table_args__ = (UniqueConstraint("match_id", "source_id", name="uq_conditions_match_source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float)
    wind_kph: Mapped[Optional[float]] = mapped_column(Float)
    precipitation_mm: Mapped[Optional[float]] = mapped_column(Float)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float)
    description: Mapped[str] = mapped_column(String(200), default="")
    surface: Mapped[str] = mapped_column(String(64), default="")
    raw: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id"), index=True)
    retrieved_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


# --------------------------------------------------------------------------
# Models, predictions, evaluation
# --------------------------------------------------------------------------
class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_key", "version", "training_cutoff", name="uq_model_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_key: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32), default="1")
    sport: Mapped[str] = mapped_column(String(32), default="football")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    training_cutoff: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    n_training_matches: Mapped[int] = mapped_column(Integer, default=0)
    trained_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class Prediction(Base):
    """A versioned snapshot. Recalculation creates a new row and marks the
    previous one superseded, so the history of what was believed when is
    preserved."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_pred_match_created", "match_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    as_of: Mapped[dt.datetime] = mapped_column(DateTime)
    engine_version: Mapped[str] = mapped_column(String(32), default="")
    primary_model: Mapped[str] = mapped_column(String(64), default="")
    data_quality_tier: Mapped[str] = mapped_column(String(16), default="UNKNOWN")
    data_quality: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_weights: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    superseded: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    change_summary: Mapped[str] = mapped_column(Text, default="")

    targets: Mapped[list["PredictionTarget"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )
    factors: Mapped[list["PredictionFactor"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class PredictionTarget(Base):
    __tablename__ = "prediction_targets"
    __table_args__ = (Index("ix_target_pred_market", "prediction_id", "market"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    market: Mapped[str] = mapped_column(String(64), index=True)   # e.g. "1x2", "over_under_2.5"
    selection: Mapped[str] = mapped_column(String(64))            # e.g. "home", "over"
    probability: Mapped[Optional[float]] = mapped_column(Float)
    expected_value: Mapped[Optional[float]] = mapped_column(Float)  # for count/continuous targets
    interval_low: Mapped[Optional[float]] = mapped_column(Float)
    interval_high: Mapped[Optional[float]] = mapped_column(Float)
    model_key: Mapped[str] = mapped_column(String(64), default="")
    # Confidence is a statement about the evidence, kept distinct from probability.
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    sufficient_data: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str] = mapped_column(Text, default="")
    subject_type: Mapped[str] = mapped_column(String(24), default="match")  # match|team|player
    subject_id: Mapped[Optional[int]] = mapped_column(Integer)
    subject_name: Mapped[str] = mapped_column(String(200), default="")

    prediction: Mapped[Prediction] = relationship(back_populates="targets")


class PredictionFactor(Base):
    """One explanation line, with the evidence that produced it."""

    __tablename__ = "prediction_factors"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    market: Mapped[str] = mapped_column(String(64), default="1x2")
    favours: Mapped[str] = mapped_column(String(32), default="")    # home|away|over|under|neutral
    category: Mapped[str] = mapped_column(String(48), default="")   # form|attack|defence|rest|availability|h2h
    statement: Mapped[str] = mapped_column(Text)
    impact: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    prediction: Mapped[Prediction] = relationship(back_populates="factors")


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    sport: Mapped[str] = mapped_column(String(32), default="football")
    label: Mapped[str] = mapped_column(String(200), default="")
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    from_date: Mapped[Optional[dt.date]] = mapped_column(Date)
    to_date: Mapped[Optional[dt.date]] = mapped_column(Date)
    n_matches: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")


class BacktestMetric(Base):
    __tablename__ = "backtest_metrics"
    __table_args__ = (Index("ix_metric_run_model", "run_id", "model_key", "market"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    model_key: Mapped[str] = mapped_column(String(64), index=True)
    market: Mapped[str] = mapped_column(String(64), index=True)
    metric: Mapped[str] = mapped_column(String(48))
    value: Mapped[float] = mapped_column(Float)
    n: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class BacktestPrediction(Base):
    """Individual out-of-sample predictions, retained for calibration curves."""

    __tablename__ = "backtest_predictions"
    __table_args__ = (Index("ix_bp_run_model_market", "run_id", "model_key", "market"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    model_key: Mapped[str] = mapped_column(String(64))
    market: Mapped[str] = mapped_column(String(64))
    selection: Mapped[str] = mapped_column(String(64), default="")
    probability: Mapped[Optional[float]] = mapped_column(Float)
    predicted_value: Mapped[Optional[float]] = mapped_column(Float)
    actual_outcome: Mapped[Optional[float]] = mapped_column(Float)
    correct: Mapped[Optional[bool]] = mapped_column(Boolean)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    finished_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    source_key: Mapped[str] = mapped_column(String(64), index=True)
    scope: Mapped[str] = mapped_column(String(200), default="")
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    records_written: Mapped[int] = mapped_column(Integer, default=0)
    records_skipped: Mapped[int] = mapped_column(Integer, default=0)
    conflicts: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
