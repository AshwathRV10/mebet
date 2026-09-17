"""Betting-market exclusion.

Requirement 7 is absolute: odds must not influence the prediction. These
tests check both halves of that - that real odds columns are recognised and
removed, and that real statistical columns are not removed by accident, since
silently discarding a genuine statistic would degrade the model invisibly.
"""

from __future__ import annotations

from mebet.normalize import is_odds_field, odds_fields_in, strip_odds_fields
from mebet.normalize.footballdata_csv import parse_matches

# Columns published by football-data.co.uk feeds.
ODDS_COLUMNS = """B365H B365D B365A BWH BWD BWA IWH IWD IWA PSH PSD PSA WHH WHD WHA VCH VCD VCA
MaxH MaxD MaxA AvgH AvgD AvgA B365>2.5 B365<2.5 P>2.5 P<2.5 Max>2.5 Avg>2.5 AHh B365AHH
B365AHA PAHH PAHA MaxAHH AvgAHH B365CH B365CD B365CA PSCH PSCD PSCA MaxCH AvgCH B365C>2.5
AvgC<2.5 AHCh BFH BFD BFA BFEH BFECH 1XBH 1XBD BMGMH GBH GBD LBH LBA SBH SJH SOH SYH BSH
BbMxH BbAvH BbOU BbMx>2.5 BbAv>2.5 BbAH BbAHh BbMxAHH BbAvAHH""".split()

STAT_COLUMNS = """Div Date Time HomeTeam AwayTeam FTHG FTAG FTR HTHG HTAG HTR Referee HS AS HST
AST HF AF HC AC HY AY HR AR HO AO HHW AHW HBP ABP Attendance SoT xG xGA xA possession
shots_on_target key_passes minutes assists saves tackles bps influence""".split()


def test_every_odds_column_is_recognised():
    missed = [c for c in ODDS_COLUMNS if not is_odds_field(c)]
    assert not missed, f"betting columns not excluded: {missed}"


def test_no_statistic_is_mistaken_for_odds():
    wrongly = [c for c in STAT_COLUMNS if is_odds_field(c)]
    assert not wrongly, f"statistics wrongly treated as odds: {wrongly}"


def test_hit_woodwork_is_a_statistic_not_a_handicap():
    # "AHW" is Away team Hit Woodwork; it must survive.
    assert not is_odds_field("AHW")
    assert not is_odds_field("HHW")
    # "AHh" is the Asian handicap line and must not.
    assert is_odds_field("AHh")


def test_strip_removes_only_odds():
    row = {"Date": "2024-08-16", "HomeTeam": "A", "HS": 12, "B365H": 1.9, "AvgD": 3.4}
    assert strip_odds_fields(row) == {"Date": "2024-08-16", "HomeTeam": "A", "HS": 12}


def test_parser_drops_odds_before_building_records():
    csv = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HC,AC,HY,AY,B365H,B365D,B365A,Avg>2.5\n"
        "2024-08-16,Alpha,Beta,2,1,14,9,6,3,1,2,1.85,3.60,4.20,1.95\n"
    )
    records, detail = parse_matches(
        csv, competition_key="TEST.1", competition_name="Test", season_label="2024-25",
        source_key="test", retrieved_at=__import__("datetime").datetime(2024, 8, 17),
    )
    assert len(records) == 1
    assert detail["odds_columns_dropped"] == 4
    record = records[0]
    # The statistics survive...
    assert record.ft_home_goals == 2 and record.home_stats.shots == 14
    assert record.home_stats.corners == 6 and record.away_stats.yellow_cards == 2
    # ...and nothing odds-shaped reaches the record.
    serialised = repr(record)
    for value in ("1.85", "3.60", "4.20", "1.95"):
        assert value not in serialised


def test_no_odds_terms_in_the_model_and_feature_code():
    """A structural check: the modelling code must not mention odds at all."""
    import pathlib

    banned = ("b365", "bookmaker", "betfair", "pinnacle", "implied_prob", "odds")
    roots = [pathlib.Path("mebet/models"), pathlib.Path("mebet/features"),
             pathlib.Path("mebet/engine")]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for term in banned:
                # The word may appear in a comment explaining the exclusion.
                for line in text.splitlines():
                    if term in line and not line.strip().startswith(("#", '"', "*")):
                        offenders.append(f"{path}: {line.strip()[:80]}")
    assert not offenders, f"betting-market references in modelling code: {offenders}"
