"""Normalisation layer: provider payloads -> canonical records."""

import re

from .records import (  # noqa: F401
    AvailabilityRecord,
    ConditionsRecord,
    LineupRecord,
    MatchRecord,
    PlayerMatchRecord,
    TeamStatLine,
)

# ---------------------------------------------------------------------------
# Betting-market exclusion
# ---------------------------------------------------------------------------
# Requirement: betting odds must never influence a prediction. Several upstream
# CSV feeds ship dozens of odds columns beside the match statistics, so instead
# of relying on discipline further downstream, odds fields are removed at the
# parsing boundary and nothing after it ever sees them.
#
# The match is on the *shape* of an odds column (bookmaker token followed by an
# outcome token), not on a bare prefix. A bare-prefix rule would classify real
# statistics such as "SoT" (shots on target) as odds and silently discard them,
# which would degrade the model invisibly - the worst kind of failure here.

#: Bookmaker / aggregator tokens used by public football data feeds.
_BOOKMAKER_TOKENS = (
    "b365", "bw", "iw", "ps", "p", "wh", "vc", "gb", "bs", "so", "sj", "sy",
    "sb", "ls", "lb", "bf", "bfe", "bfc", "bfd", "avg", "max", "bb", "1xb",
    "bmgm", "pin", "bet365", "betfair",
)

#: Outcome tokens that follow a bookmaker token in these feeds:
#: h/d/a (1X2), ah* (Asian handicap), ou/o/u (totals), >2.5 / <2.5 (lines),
#: optionally preceded by "c" (closing price) and mx/av (max/average).
_ODDS_SUFFIX = re.compile(
    r"^c?(mx|av)?(ahh|ahca|ahch|aha|ah|ou|o|u|h|d|a)?([<>]=?\d+(\.\d+)?)?$"
)

#: Substrings that are unambiguous regardless of position.
ODDS_FIELD_SUBSTRINGS = (
    "odds", "bookmaker", "betfair", "pinnacle", "bet365", "implied_prob",
    "market_price", "closing_price", "closing_odds", "price_",
)

#: Exact names that carry handicap lines rather than measurements.
#: ("ahw"/"hhw" are *hit woodwork* statistics, not handicaps - see PROTECTED.)
ODDS_FIELD_EXACT = {"ahh", "ahca", "ahch", "bbah", "bbahh", "bbav", "bbmx"}

#: Statistical columns that must never be classified as odds.
PROTECTED_STAT_FIELDS = {
    "hs", "as", "hst", "ast", "hc", "ac", "hf", "af", "hy", "ay", "hr", "ar",
    "ho", "ao", "hhw", "ahw", "hbp", "abp", "fthg", "ftag", "ftr",
    "hthg", "htag", "htr", "sot", "shots", "shots_on_target", "possession",
    "corners", "fouls", "saves", "passes", "attendance", "referee", "date",
    "time", "div", "hometeam", "awayteam", "season", "xg", "xga", "xa",
}


def is_odds_field(name: str) -> bool:
    """True if a column name denotes betting-market information."""
    low = name.strip().lower().replace(" ", "_")
    if not low:
        return False
    if low in PROTECTED_STAT_FIELDS:
        return False
    if any(s in low for s in ODDS_FIELD_SUBSTRINGS):
        return True
    if low in ODDS_FIELD_EXACT:
        return True
    for token in _BOOKMAKER_TOKENS:
        if low.startswith(token):
            rest = low[len(token):]
            if rest == "":
                continue
            if _ODDS_SUFFIX.fullmatch(rest):
                return True
    return False


def strip_odds_fields(row: dict) -> dict:
    """Return ``row`` with every betting-market field removed."""
    return {k: v for k, v in row.items() if not is_odds_field(k)}


def odds_fields_in(names) -> list[str]:
    """Which of ``names`` are betting-market fields (for logging/audit)."""
    return [n for n in names if is_odds_field(n)]
