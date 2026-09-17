# mebet

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

Not a claim — output from `mebet backtest`, walk-forward over Premier League
matches with models refitted every 30 days and features built at each match's
own kickoff.

**3 seasons, 1,123 out-of-sample matches (2023-08 to 2026-05):**

| model | accuracy | log loss | Brier | calibration error |
|---|---|---|---|---|
| **dixon_coles** | **0.520** | **0.9803** | **0.5840** | **0.0201** |
| ensemble (equal weights) | 0.527 | 0.9827 | 0.5848 | 0.0253 |
| elo | 0.524 | 0.9916 | 0.5902 | 0.0302 |
| logistic | 0.525 | 1.0081 | 0.5988 | 0.0321 |
| *base rate (constant)* | *0.417* | *1.0720* | *0.6538* | — |

That is **8.55% better log loss than predicting the historical base rate every
time**, and 10 accuracy points better. For a model using no market information,
that is a real but modest edge — which is what an honest football model looks
like.

Other markets, same run:

| market | metric | value | base rate |
|---|---|---|---|
| over/under 2.5 goals | log loss | 0.6838 | 0.5895 |
| both teams to score | log loss | 0.6861 | 0.5841 |
| total goals | MAE / RMSE | 1.32 / 1.65 | — |
| total corners | MAE / RMSE | 2.82 / 3.50 | — |
| total cards | MAE / RMSE | 1.82 / 2.33 | — |

**Ensemble weights are earned, not assumed.** Weights are fitted by log-loss
minimisation on one slice and validated on a later, unseen slice:

```
fitted on 673 matches, scored on 450 later matches
weighted ensemble   1.01728   <- best
equal weights       1.01856
best single model   1.01891   (dixon_coles)
```

The margin is small. It is reported rather than inflated.

**Known weaknesses**, visible in the same output: the corners model is the
least well calibrated (calibration error ~0.07 against ~0.02 for match result)
and over-predicts slightly (bias +0.33 corners). Expected-goals data is absent
from the bulk feed, so xG-based features are unavailable for team ratings.

## Quick start

```bash
make setup                      # virtualenv + dependencies
make init                       # create the database
make sources                    # which providers can this machine reach?
make ingest                     # ~10 seasons across 5 leagues (a few minutes)
make serve                      # http://127.0.0.1:8000
```

Or from the command line:

```bash
.venv/bin/python -m mebet.cli analyze \
  --competition ENG.1 --home "Manchester City" --away "Liverpool" --date 2026-09-26

.venv/bin/python -m mebet.cli backtest \
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
make test      # 62 tests
```

The suite covers leakage (11 tests), betting-odds exclusion, team-identity
resolution, model distributions, data-quality tiers and target derivation.

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
