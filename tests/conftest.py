"""Shared fixtures.

Tests run against a temporary database, never the user's real one.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def temp_db(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="mebet-test-")
    db_path = Path(tmpdir) / "test.db"
    monkeypatch.setenv("MEBET_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MEBET_DATA_DIR", tmpdir)
    monkeypatch.setenv("MEBET_CACHE_DIR", str(Path(tmpdir) / "cache"))
    monkeypatch.setenv("MEBET_LOG_DIR", str(Path(tmpdir) / "logs"))

    import mebet.config as config
    import mebet.db.session as session_module

    config._settings = None
    session_module.reset_engine()
    session_module.init_db()
    yield db_path
    session_module.reset_engine()
    config._settings = None


@pytest.fixture
def session(temp_db):
    from mebet.db.session import new_session

    s = new_session()
    yield s
    s.close()


@pytest.fixture
def seeded(session):
    """A small, fully synthetic league.

    Synthetic data is correct here: these tests check the machinery - cutoffs,
    aggregation, model plumbing - not the accuracy of any real prediction.
    Accuracy is measured by the backtester against real matches instead.
    """
    from mebet.db.models import Competition, DataSource, Match, Season, Team, TeamMatchStats

    source = DataSource(key="test_source", name="Test", reliability=0.9)
    session.add(source)
    comp = Competition(sport="football", key="TEST.1", name="Test League", country="Testland")
    session.add(comp)
    session.flush()
    season = Season(competition_id=comp.id, label="2024-25")
    session.add(season)

    teams = []
    for name in ["Alpha FC", "Beta FC", "Gamma FC", "Delta FC"]:
        t = Team(sport="football", canonical_name=name, country="Testland")
        session.add(t)
        teams.append(t)
    session.flush()

    # A round-robin repeated over 12 dates with deterministic, varied scores.
    start = dt.date(2025, 1, 1)
    day = 0
    for cycle in range(12):
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i == j:
                    continue
                date = start + dt.timedelta(days=day)
                day += 3
                hg = (i + cycle) % 4
                ag = (j + cycle) % 3
                m = Match(
                    sport="football", competition_id=comp.id, season_id=season.id,
                    kickoff_date=date,
                    kickoff_utc=dt.datetime.combine(date, dt.time(15, 0)),
                    kickoff_is_exact=True,
                    home_team_id=home.id, away_team_id=away.id, status="played",
                    ft_home_goals=hg, ft_away_goals=ag,
                    ht_home_goals=hg // 2, ht_away_goals=ag // 2,
                    referee="Test Referee", source_id=source.id,
                    retrieved_at=dt.datetime(2025, 1, 1),
                )
                session.add(m)
                session.flush()
                for team, is_home, goals, conceded in (
                    (home, True, hg, ag), (away, False, ag, hg)
                ):
                    session.add(TeamMatchStats(
                        match_id=m.id, team_id=team.id, is_home=is_home,
                        goals=goals, goals_conceded=conceded,
                        shots=10 + goals, shots_on_target=3 + goals,
                        corners=4 + (goals % 3), fouls=11, yellow_cards=1 + (goals % 2),
                        red_cards=0, source_id=source.id,
                        retrieved_at=dt.datetime(2025, 1, 1),
                    ))
    session.commit()
    return {"competition": comp, "teams": teams, "season": season, "source": source}
