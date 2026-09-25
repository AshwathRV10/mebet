import sys, time, datetime as dt, numpy as np; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
from mebet.models.ensemble import fit_weights
from mebet.backtest.evaluate import EvaluationResult, MatchScore, paired_comparison
DC_NEW = {"sot_weight": 0.35, "shots_weight": 0.15, "half_life_days": 270, "ridge": 1.0,
          "promoted_offset": 0.3, "rate_unrated_teams": True}
ELO_NEW = {"season_regression": 0.1}
t = time.time()
res = run_many([("dc_old", "dc", OLD_DC), ("elo_old", "elo", OLD_ELO), ("dc_new", "dc", DC_NEW),
                ("elo_new", "elo", ELO_NEW), ("logistic", "logistic", {})], VALID, refit=28)
print(f"ran in {time.time()-t:.0f}s")

shared = set.intersection(*[set(r.scores) for r in res.values()])
split = dt.date(2022, 7, 1)
fit_ids = sorted(m for m in shared if res["dc_new"].scores[m].date < split)
eval_ids = sorted(m for m in shared if res["dc_new"].scores[m].date >= split)
print(f"weights fitted on 2021-22 ({len(fit_ids)} matches), scored on 2022-23 ({len(eval_ids)} matches)\n")

def arr(key, ids): return np.array([res[key].scores[m].probs for m in ids])
truth_fit = np.array([res["dc_new"].scores[m].outcome for m in fit_ids])

def blended(label, keys, weights):
    out = EvaluationResult(label=label)
    w = np.array([weights[k] for k in keys]); w = w / w.sum()
    for m in eval_ids:
        p = sum(wi * np.array(res[k].scores[m].probs) for wi, k in zip(w, keys))
        s0 = res["dc_new"].scores[m]
        out.scores[m] = MatchScore(m, s0.competition, s0.date, s0.outcome, tuple(p))
    return out

cands = {}
for name, keys in [("OLD ensemble dc+elo+logistic", ["dc_old", "elo_old", "logistic"]),
                   ("NEW dc+elo", ["dc_new", "elo_new"]),
                   ("NEW dc+elo+logistic", ["dc_new", "elo_new", "logistic"])]:
    w, _ = fit_weights({k: arr(k, fit_ids) for k in keys}, truth_fit)
    cands[f"{name}  w={ {k: round(v,2) for k,v in w.items()} }"] = blended(name, keys, w)
singles = {k: blended(k, [k], {k: 1.0}) for k in ("dc_old", "elo_old", "logistic", "dc_new", "elo_new")}

base = singles["dc_old"]
print(f"{'on 2022-23':72}{'logloss':>9}   diff vs dc_old [95% CI]")
for label, r in list(singles.items()) + list(cands.items()):
    s = r.summary(); c = paired_comparison(base, r)
    tail = "" if label == "dc_old" else f"   {c['diff']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]"
    print(f"  {label:70}{s['log_loss']:>9.4f}{tail}")
new2 = [v for k, v in cands.items() if k.startswith("NEW dc+elo ")][0]
new3 = [v for k, v in cands.items() if k.startswith("NEW dc+elo+logistic")][0]
c = paired_comparison(new2, new3)
print(f"\ndoes adding logistic to the new dc+elo help?  {c['diff']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]  P(better)={c['p_better']:.2f}")
