# Architecture

## Layers

```
   mebet/web          zero-build SPA (no framework, no toolchain)
   mebet/api          FastAPI: analyze, refresh, history, backtests, sources
   mebet/engine       orchestration, targets, explanations, versioning
   mebet/models       Dixon-Coles, Elo, logistic, corners, cards, players, ensemble
   mebet/features     form windows, splits, opponent adjustment, time decay
   mebet/quality      missing / stale / conflicting / suspicious data, confidence
   mebet/backtest     walk-forward evaluation, scoring rules, calibration
   mebet/db           schema, AS-OF repository, in-memory index
   mebet/normalize    canonical records, team identity, ingestion
   mebet/sources      provider adapters, HTTP cache, rate limits, robots
```

Dependencies point downward only. `models` and `features` never touch a source
adapter; adapters never touch the database.

## The leakage boundary

This is the single most important design decision in the project, because a
leak does not produce an error — it produces flattering numbers.

Every read used for feature building or model fitting goes through
`AsOfRepository`. Its constructor *requires* a cutoff:

```python
AsOfRepository(session, as_of)          # fine
AsOfRepository(session, None)           # ValueError
```

Two distinct cutoffs are enforced:

**Event time.** A match may inform a prediction only if it had *finished*
before the cutoff. A fixture that kicked off twenty minutes earlier has no
known result, so it is excluded — `MATCH_DURATION` (2h30) is subtracted from
the cutoff for matches with an exact kickoff time. Matches known only by date
must fall on a strictly earlier calendar date.

**Observation time.** Time-varying records — injuries, lineups, weather
forecasts — may be used only if they were observed at or before the cutoff.
Today's injury list must not inform a prediction dated 2023.

Both are applied in SQL, and `_verify()` re-checks every row returned. A test
feeds the guard a deliberately post-cutoff row and asserts it raises.

Standings are **derived** from matches before the cutoff rather than fetched,
so a historical league table is automatically correct as of any date and cannot
import a future standing.

## The in-memory index

Building features from SQL costs hundreds of queries per match. The logistic
model needs the feature set as it stood before each of a thousand past matches,
and the backtester needs that thousands of times over; served from SQL a single
fit took minutes.

`MatchIndex` loads every match and its statistics once and answers from memory.
It does not weaken the guarantee: the index is loaded with an upper bound, every
read still takes an explicit cutoff, filtering applies the same rule, and
verification still runs. `test_index_and_sql_paths_agree` asserts the two paths
produce byte-identical features.

Measured: 590ms → 9.6ms per feature build, with identical output.

## Models

| model | method | produces |
|---|---|---|
| `dixon_coles` | time-decayed bivariate Poisson, low-score correction, MLE | full scoreline matrix |
| `elo` | sequential ratings, margin-of-victory scaling, learned draw curve | 1X2 |
| `logistic` | multinomial regression on 23 engineered features | 1X2 |
| `corners` | multiplicative rate model, Poisson or negative binomial | corner totals |
| `cards` | as above, plus per-referee rates where the sample supports it | card totals |
| `players` | per-90 rates blended with xG/xA, scaled by match context | player markets |
| `ensemble` | log-loss-minimising mixture of the outcome models | 1X2 + scorelines |

Design notes worth knowing:

- **Dixon-Coles is primary** because a single joint distribution over scorelines
  makes over/under, both-teams-to-score, correct score and clean sheets mutually
  consistent. Estimating them separately invites contradictions.
- **The ensemble reconciles its scoreline matrix** with its own 1X2 output by
  rescaling the three outcome regions, so the "most likely score" can never
  contradict the "most likely result".
- **Distribution choice is measured.** The count models compute the
  variance-to-mean ratio of the training sample and use a negative binomial only
  when overdispersion is real (>1.1), Poisson otherwise.
- **Every model can decline.** `sufficient_data=False` with a reason is a valid
  answer and propagates to the UI.

## Backtesting

Walk-forward. Models are refitted at periodic checkpoints; a model predicting a
match was always fitted at the checkpoint at or before it — never later.
Refitting before every match would be the theoretical ideal but is wasteful;
periodic refitting errs toward a *staler* model, which is the safe direction.

Features, by contrast, are rebuilt at each match's own kickoff, so form and
standings are exactly what was visible that day.

Scored per market and per model: accuracy, log loss, multi-class Brier,
calibration bins, expected calibration error, MAE and RMSE for counts — against
base-rate and always-home baselines, because accuracy alone flatters any
football model.

Ensemble weights are fitted on one slice and validated on a later unseen slice,
so the benefit of weighting is measured rather than asserted.

## Prediction versioning

Recalculating does not overwrite. The previous row is marked superseded, a new
version is written, and `change_summary` records what moved:

```
changes: home +3.1%, away -2.4%; data quality MEDIUM -> HIGH
```

## Probability, confidence and data quality

Three different things, deliberately kept apart:

- **Probability** comes from the models.
- **Confidence** reflects how much the evidence can bear — sample sizes, source
  count, lineup status, freshness.
- **Data quality** is a tier (HIGH / MEDIUM / LOW / INSUFFICIENT) over what was
  retrieved.

Low quality reduces *confidence*. It does not quietly shade probabilities toward
50%, because a probability that has been nudged is no longer a statement about
the world.

## Multi-sport extension

Core entities carry a `sport` column; sport-specific measurements live in their
own tables (`team_match_stats`, `player_match_stats`). A new sport supplies a
source adapter, a feature builder, models and prediction targets, and reuses
storage, cutoffs, quality assessment, backtesting, explanations, the API and the
UI unchanged.
