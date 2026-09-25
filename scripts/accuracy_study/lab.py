"""Experiment driver. Validation window is for choosing; test window is touched once."""
import datetime as dt, hashlib, json, os, pickle, time
from multiprocessing import Pool

LEAGUES = ["ENG.1", "ESP.1", "ITA.1", "GER.1", "FRA.1"]
VALID = (dt.date(2021, 8, 1), dt.date(2023, 6, 30))
TEST = (dt.date(2023, 8, 1), dt.date(2026, 5, 31))
CACHE = os.path.join(os.path.dirname(__file__), "cache")

# The configurations before this study. Spelled out, because the model
# defaults are now the tuned values: passing {} would silently run the new model.
OLD_DC = {"half_life_days": 180, "sot_weight": 0.0, "shots_weight": 0.0, "ridge": 0.0,
          "promoted_offset": 0.0, "rate_unrated_teams": False}
OLD_ELO = {"k_factor": 20, "home_advantage": 60, "season_regression": 0.25}
os.makedirs(CACHE, exist_ok=True)

_S = _IDX = None

def _init():
    global _S, _IDX
    import logging; logging.disable(logging.INFO)
    from mebet.db.session import new_session
    from mebet.db.index import MatchIndex
    _S = new_session(); _IDX = MatchIndex(_S)

def _factory(cls_name, kwargs):
    from mebet.models.dixon_coles import DixonColesModel
    from mebet.models.elo import EloModel
    from mebet.models.logistic import LogisticOutcomeModel
    cls = {"dc": DixonColesModel, "elo": EloModel, "logistic": LogisticOutcomeModel}[cls_name]
    return lambda: cls(**kwargs)

def _job(args):
    cls_name, kwargs, league, window, refit = args
    from mebet.backtest.evaluate import walk_forward
    return walk_forward(_S, _IDX, league, window[0], window[1], _factory(cls_name, kwargs),
                        refit_days=refit, needs_features=(cls_name == "logistic"))

_POOL = None
def pool():
    global _POOL
    if _POOL is None: _POOL = Pool(4, initializer=_init)
    return _POOL

def _key(cls_name, kwargs, league, window, refit):
    raw = json.dumps([cls_name, sorted(kwargs.items()), league, str(window), refit], default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def run_many(specs, window, refit=7, leagues=LEAGUES):
    """specs: list of (label, cls_name, kwargs). Returns {label: merged EvaluationResult}."""
    from mebet.backtest.evaluate import merge
    jobs, where = [], []
    out_parts = {label: {} for label, _, _ in specs}
    for label, cls_name, kwargs in specs:
        for lg in leagues:
            path = os.path.join(CACHE, _key(cls_name, kwargs, lg, window, refit) + ".pkl")
            if os.path.exists(path):
                out_parts[label][lg] = pickle.load(open(path, "rb"))
            else:
                jobs.append((cls_name, kwargs, lg, window, refit)); where.append((label, lg, path))
    if jobs:
        for (label, lg, path), res in zip(where, pool().map(_job, jobs, chunksize=1)):
            pickle.dump(res, open(path, "wb")); out_parts[label][lg] = res
    return {label: merge([parts[lg] for lg in leagues], label) for label, parts in out_parts.items()}

def table(results, baseline_label, metric="log_loss"):
    from mebet.backtest.evaluate import paired_comparison
    base = results[baseline_label]
    shared = set(base.scores)
    for r in results.values(): shared &= set(r.scores)
    print(f"  common matches: {len(shared)}")
    print(f"  {'config':38}{'n':>6}{'logloss':>9}{'brier':>8}{'acc':>7}{'O/U2.5':>8}   diff vs base [95% CI]   P(better)")
    for label, r in results.items():
        s = r.summary(shared)
        line = f"  {label:38}{s['n']:>6}{s['log_loss']:>9.4f}{s['brier']:>8.4f}{s['accuracy']:>7.3f}"
        line += f"{s.get('ou25_log_loss', float('nan')):>8.4f}"
        if label != baseline_label:
            c = paired_comparison(base, r, metric=metric)
            line += f"   {c['diff']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]  {c['p_better']:.2f}"
        print(line)
