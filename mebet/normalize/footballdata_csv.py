"""Parser for the football-data.co.uk row format.

Used by both the direct adapter and the datahub mirror, which publish the
same columns. Keeping one parser means a fix to date handling or a newly
added statistic benefits both sources.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Optional

from ..logging_setup import get_logger
from . import odds_fields_in, strip_odds_fields
from .records import MatchRecord, TeamStatLine

log = get_logger("normalize.footballdata")


def to_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def to_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def to_date(value: str) -> Optional[dt.date]:
    """Accepts the several date spellings these feeds have used over 30 years."""
    v = (value or "").strip()
    if not v:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def parse_matches(
    text: str,
    *,
    competition_key: str,
    competition_name: str,
    season_label: str,
    source_key: str,
    retrieved_at: dt.datetime,
    country: str = "",
) -> tuple[list[MatchRecord], dict]:
    """Parse a football-data.co.uk style CSV into canonical match records.

    Returns ``(records, detail)``; ``detail`` reports rows skipped and how
    many betting-market columns were discarded, so the audit trail shows the
    odds exclusion actually happened.
    """
    reader = csv.DictReader(io.StringIO(text))
    dropped = odds_fields_in(reader.fieldnames or [])
    if dropped:
        log.info("%s: discarded %d betting-market columns", source_key, len(dropped))

    records: list[MatchRecord] = []
    skipped = 0
    for raw in reader:
        row = strip_odds_fields(raw)
        date = to_date(row.get("Date", ""))
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        if not date or not home or not away:
            skipped += 1
            continue

        fthg, ftag = to_int(row.get("FTHG")), to_int(row.get("FTAG"))
        played = fthg is not None and ftag is not None

        kickoff, exact = None, False
        time_str = (row.get("Time") or "").strip()
        if time_str:
            try:
                hh, mm = time_str.split(":")[:2]
                kickoff = dt.datetime(date.year, date.month, date.day, int(hh), int(mm),
                                      tzinfo=dt.timezone.utc)
                exact = True
            except ValueError:
                kickoff, exact = None, False

        records.append(
            MatchRecord(
                sport="football",
                competition_key=competition_key,
                competition_name=competition_name,
                season_label=season_label,
                home_team=home,
                away_team=away,
                kickoff_date=date,
                kickoff_utc=kickoff,
                kickoff_is_exact=exact,
                status="played" if played else "scheduled",
                ft_home_goals=fthg,
                ft_away_goals=ftag,
                ht_home_goals=to_int(row.get("HTHG")),
                ht_away_goals=to_int(row.get("HTAG")),
                referee=(row.get("Referee") or "").strip(),
                home_stats=TeamStatLine(
                    shots=to_int(row.get("HS")),
                    shots_on_target=to_int(row.get("HST")),
                    corners=to_int(row.get("HC")),
                    fouls=to_int(row.get("HF")),
                    yellow_cards=to_int(row.get("HY")),
                    red_cards=to_int(row.get("HR")),
                ),
                away_stats=TeamStatLine(
                    shots=to_int(row.get("AS")),
                    shots_on_target=to_int(row.get("AST")),
                    corners=to_int(row.get("AC")),
                    fouls=to_int(row.get("AF")),
                    yellow_cards=to_int(row.get("AY")),
                    red_cards=to_int(row.get("AR")),
                ),
                external_ids={"country": country} if country else {},
                source_key=source_key,
                retrieved_at=retrieved_at,
            )
        )

    return records, {"rows_skipped": skipped, "odds_columns_dropped": len(dropped)}
