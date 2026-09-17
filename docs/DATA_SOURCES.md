# Data sources

## The rule

If a source cannot be reached, the system says so. It does not fall back to
plausible-looking values, and it does not present a stale cache as fresh
without labelling it. Every stored fact carries `source_id` and `retrieved_at`,
and every retrieval attempt — successful or not — is written to `fetch_log`.

Run `make sources` to see what your machine can currently reach.

## Reachability is environment-specific

Corporate networks, some VPNs and sandboxed environments block many of these
hosts. This matters enough to state plainly: **this project was developed in an
environment whose egress policy denied every dedicated sports host**
(`football-data.co.uk`, `api.football-data.org`, `thesportsdb.com`,
`site.api.espn.com`, `understat.com`, `fbref.com`, `api.open-meteo.com`), and
permitted only package registries and `raw.githubusercontent.com`.

That constraint shaped which adapters could be verified end to end rather than
which were written. The GitHub-hosted feeds below were exercised against live
data; the others are implemented against their documented APIs and will work on
an unrestricted network, where `make sources` will show them as reachable.

## Sources

### `datahub_footballdata` — reliability 0.85
[datasets/football-datasets](https://github.com/datasets/football-datasets)

Regenerated daily by CI from football-data.co.uk, published under the Open Data
Commons PDDL. Five leagues (England, Spain, Italy, Germany, France), seasons
1993-94 onward.

Per match: date, teams, full-time and half-time scores, referee, and for each
side shots, shots on target, fouls, corners, yellow and red cards.

Notable for requirement 7: the upstream processing drops every bookmaker
column, so odds are **structurally absent** rather than merely filtered. The
odds filter still runs as a second barrier.

### `footballdata_couk` — reliability 0.90
[football-data.co.uk](https://www.football-data.co.uk/data.php)

The original publisher. Same columns, fresher (updated twice weekly in season),
and far broader: 18 competitions including the English lower divisions,
Scotland, the Netherlands, Belgium, Portugal, Turkey and Greece.

Its files **do** carry dozens of bookmaker columns. `parse_matches` strips them
before any record is constructed — see `mebet/normalize/__init__.py`, where the
filter matches odds *shapes* (bookmaker token plus outcome token) rather than
bare prefixes, so that genuine statistics such as `SoT` or `AHW` ("away team
hit woodwork") are never discarded by accident. 74 real odds columns and 44
real statistic columns are asserted in `tests/test_no_odds.py`.

### `fpl_api` — reliability 0.88
[Fantasy Premier League API](https://fantasy.premierleague.com/api/bootstrap-static/)

**The injury and availability feed.** For every Premier League player it
publishes a squad status (available / doubtful / injured / suspended /
unavailable), a percentage chance of playing the next round, a free-text news
line and the timestamp that news was added. This is genuine first-party team
news rather than an inference from absence.

It does not publish lineups, so lineups remain unavailable from this source and
the system reports them as such.

Premier League only.

### `fpl_mirror` — reliability 0.70
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)

The only freely reachable feed found that carries **per-player, per-match**
statistics including expected goals, expected assists and expected goals
conceded, back several seasons. This is what makes player-level prediction
possible rather than aspirational.

Per player per gameweek: minutes, starts, goals, assists, xG, xA, xGC, tackles,
saves, cards, bonus points. Also fixtures with kickoff times and results.

It is a volunteer-maintained mirror and can lag the live API by days. The lag is
measured (`detail['snapshot_lag_days']`) and passed to the quality layer rather
than ignored.

### `openfootball` — reliability 0.60
[openfootball/football.json](https://github.com/openfootball/football.json)

Broad competition coverage including cups, but results only — no shots, corners
or cards. Registered as a fixtures/results source to widen coverage; it does not
feed the statistical models.

### `open_meteo` — reliability 0.80
[Open-Meteo](https://open-meteo.com)

No API key. Importantly it offers a **historical archive** endpoint as well as
forecasts, so a backtested match can be given the weather that was actually
recorded rather than today's forecast — using a forecast for a 2023 match would
be leakage.

## Sources deliberately not used

- **Any bookmaker or exchange feed.** Excluded by requirement, and the
  exclusion is enforced structurally rather than by convention.
- **FBref / Understat scraping.** Both publish excellent xG data, and adapters
  would fit the existing interface. They are not included by default because
  their terms and rate limits warrant a deliberate decision by the operator
  rather than a default-on scraper. The HTTP client already honours robots.txt
  and per-host rate limits for exactly this case.

## Adding a source

Implement `SourceAdapter`, declare what it provides, and register it:

```python
from .base import Capability, SourceAdapter, SourceResponse, SourceStatus
from .registry import register

@register
class MyAdapter(SourceAdapter):
    key = "my_source"
    name = "My provider"
    capabilities = (Capability.MATCH_STATS, Capability.LINEUPS)
    reliability = 0.8              # used to break ties when sources disagree

    def status(self) -> SourceStatus:
        probe = self.fetch("https://example.com/health")
        return SourceStatus(key=self.key, available=probe.ok, detail=probe.error)

    def fetch_matches(self, *, competition_key, season, sport="football"):
        result = self.fetch(f"https://example.com/{competition_key}/{season}")
        if not result.ok:
            return SourceResponse.failure(self.key, result.error)   # never fabricate
        return SourceResponse(self.key, ok=True, records=[...])
```

The pipeline picks it up automatically, ordered by `reliability`. When two
sources disagree about a recorded result, the more reliable one wins for the
match row, both stat rows are retained under their own `source_id`, and the
disagreement is counted and surfaced in the data-quality report.
