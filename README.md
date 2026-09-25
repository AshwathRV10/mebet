# mebet

[![tests](https://github.com/AshwathRV10/mebet/actions/workflows/tests.yml/badge.svg)](https://github.com/AshwathRV10/mebet/actions/workflows/tests.yml)

A locally run sports match prediction engine. You name a specific fixture; it
gathers real data, fits statistical models on history that existed before the
match, and produces probabilities with the evidence behind them.

Football is implemented. The architecture is multi-sport, and section
[Adding a sport](#adding-a-sport) describes what a second sport requires.

**No betting odds are used anywhere in the prediction path.** Odds columns are
stripped at the parsing boundary, and a test asserts that no odds terminology
appears in the modelling code at all.

---

## What it actually does

```
Enter a fixture  →  collect  →  validate  →  build features  →  fit models  →  predict
```

- **Real data only.** Every stored fact carries the source it came from and the
  time it was retrieved. A source that cannot be reached is reported as
  unavailable; nothing is filled in with a guess.
- **Leakage is structurally prevented.** Every read used for features or model
  fitting goes through `AsOfRepository`, which requires an explicit cutoff,
  filters on it in SQL, and verifies the rows it returns. Eleven tests target
  this specifically.
- **Markets are withheld when the data cannot support them.** "Insufficient
  reliable data" is a valid answer and is shown as one, with the reason.
- **Predictions are versioned.** Refreshing creates a new version and reports
  what changed rather than overwriting the old one.

## Measured performance

Not a claim — output from `scripts/accuracy_study/final_test.py`. Every model
setting was chosen on 2021-22 and 2022-23; the table below is the **one** run
on the later, untouched window. Walk-forward, all five leagues, models refitted
every 28 days, ensemble weights fitted on the validation seasons only.

**Test window 2023-08 to 2026-05 — 5,165 matches:**

| | log loss | Brier | accuracy |
|---|---|---|---|
| **current ensemble** | **0.9840** | **0.5863** | 0.524 |
| previous ensemble (before the accuracy study) | 0.9860 | 0.5876 | 0.525 |
| *base rate (predict league frequencies every time)* | *1.0754* | — | — |

That is **8.5% better log loss than the base rate**. For a model using no
market information that is a real but modest edge — which is what an honest
football model looks like.

What the accuracy study changed, measured on the same window:

| change | effect on the test window | 95% interval |
|---|---|---|
| whole system, 1X2 log loss | −0.0020 | −0.0031 to −0.0009 |
| goals model, over/under 2.5 log loss | −0.0088 | −0.0121 to −0.0057 |
| Elo alone | −0.0037 | −0.0054 to −0.0019 |
| matches that used to be declined | +49 now predicted, beating base rates by 0.16 log loss | −0.30 to −0.01 |

Honest reading: the gain is real in every league and clear pooled, but small
for match result. It is largest for **goals markets**, because the change that
mattered most — fitting team ratings on shots as well as goals — improves
expected goals directly. Hit rate barely moves: the gain is in how good the
probabilities are, not in picking different favourites. It was also smaller
here than on the validation seasons, which is exactly why the test window was
held back.

What was tried and **not** kept, because it did not help: a stronger shrinkage
prior (hurt established teams), a draw-rate correction (draws were already
calibrated to within 0.1 percentage points), and pure shot-based ratings
(worse than the goals/shots blend). Details in
[`scripts/accuracy_study/`](scripts/accuracy_study/README.md).

**Known weaknesses.** The corners model is the least well calibrated component.
The remaining gap to a sharp bookmaker is mostly information this data does
not contain — confirmed lineups, and injury news outside the Premier League.

## Quick start

```bash
make setup                      # virtualenv + dependencies
make init                       # create the database
make sources                    # which providers can this machine reach?
make ingest                     # ~10 seasons across 5 leagues (a few minutes)
make serve                      # http://127.0.0.1:8000
```

`make setup` installs the package, so `mebet` is on the path:

```bash
.venv/bin/mebet analyze \
  --competition ENG.1 --home "Manchester City" --away "Liverpool" --date 2026-09-26

.venv/bin/mebet backtest \
  --competition ENG.1 --from 2024-08-01 --to 2026-05-31
```

## Data sources

Which sources are reachable depends on your network. `make sources` tells you.
See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for the full list, what each
provides, and how to add another.

| source | provides | notes |
|---|---|---|
| football-data.co.uk | results, shots, corners, fouls, cards | the publisher; 20+ leagues |
| datahub `football-datasets` | the same fields, 5 leagues, 1993– | daily mirror, public domain, **no odds columns at all** |
| Fantasy Premier League API | injuries, suspensions, doubts, per-player stats | first-party team news |
| FPL historical mirror | per-player per-match stats incl. xG/xA | several seasons |
| openfootball | results and fixtures | broad competition coverage, no statistics |
| Open-Meteo | weather, incl. a historical archive | no key required |

Adding a provider means implementing `SourceAdapter` and registering it.
Nothing else changes.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). In brief:

```
sources/     adapters, caching, rate limiting, robots, provenance
normalize/   canonical records, team identity resolution, ingestion
db/          schema, as-of repository (the leakage boundary), in-memory index
features/    form windows, home/away splits, opponent adjustment, time decay
models/      Dixon-Coles, Elo, logistic, corners, cards, players, ensemble
engine/      orchestration, prediction targets, explanations, versioning
backtest/    walk-forward evaluation, scoring rules, calibration
quality/     missing, stale, conflicting and suspicious data; confidence
api/ web/    local HTTP API and a zero-build frontend
```

## Adding a sport

The core tables carry a `sport` column and sport-specific measurements live in
their own tables. A new sport needs: a source adapter, a feature builder, at
least one model, and its prediction targets. Shared infrastructure — storage,
cutoffs, quality assessment, backtesting, explanations, the API and the UI —
is reused unchanged.

## Configuration

All settings come from the environment or an optional `.env`; see
`.env.example`. API keys are read only by the backend and are never sent to the
browser. Binding to `0.0.0.0` exposes the app on a LAN; put a reverse proxy in
front of it before doing so.

## Testing

```bash
make test      # 77 tests
```

The suite covers leakage (11 tests), betting-odds exclusion, team-identity
resolution, model distributions, data-quality tiers and target derivation.

It is hermetic — synthetic fixtures and temporary databases, no third-party
feed — and is verified to pass with every outbound socket blocked, so CI never
depends on someone else's uptime. GitHub Actions runs it on Python 3.11 and
3.12, plus a separate job that installs the pinned `requirements.txt` so the
documented setup cannot rot unnoticed.

## Honest limitations

- **Lineups.** No reachable free source publishes confirmed lineups. The system
  reports them as unavailable and reduces confidence accordingly; it does not
  guess. The lineup tables, ingestion and quality handling are built, so an
  adapter for a lineup provider is a drop-in.
- **Injuries.** The FPL API is a genuine first-party feed, but it covers the
  Premier League only. Other leagues report no team news.
- **Expected goals.** Available per player (FPL), not per team from the bulk
  match feed. Team xG features exist in the schema and are used when present.
- **Draw prediction.** Like every football model, this one is weakest on draws;
  they are rarely the argmax even when correctly assigned ~25% probability.
- **These are probabilities, not forecasts of certainty.** A 58% home win
  should lose roughly 42% of the time. Calibration is measured and reported
  precisely so that claim can be checked.
