"""FINAL held-out test. Run once. Test window was not used for any choice above."""
import sys, json, numpy as np; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
from mebet.models.ensemble import fit_weights
from mebet.models.dixon_coles import DixonColesModel
from mebet.models.elo import EloModel
from mebet.backtest.evaluate import EvaluationResult, MatchScore, paired_comparison

OLD_DC = {"half_life_days": 180, "sot_weight": 0.0, "shots_weight": 0.0, "ridge": 0.0, "promoted_offset": 0.0, "rate_unrated_teams": False}
NEW_DC = {"half_life_days": 270, "sot_weight": 0.35, "shots_weight": 0.15, "ridge": 1.0, "promoted_offset": 0.3, "rate_unrated_teams": True}
OLD_ELO = {"k_factor": 20, "home_advantage": 60, "season_regression": 0.25}
NEW_ELO = {"k_factor": 20, "home_advantage": 60, "season_regression": 0.1}
assert all(DixonColesModel().params[k] == v for k, v in NEW_DC.items()), "shipped DC defaults != tested"
assert all(EloModel().params[k] == v for k, v in NEW_ELO.items()), "shipped Elo defaults != tested"

SPECS = [("dc_old", "dc", OLD_DC), ("elo_old", "elo", OLD_ELO), ("dc_new", "dc", NEW_DC),
         ("elo_new", "elo", NEW_ELO), ("logistic", "logistic", {})]
SYSTEMS = {"OLD": ["dc_old", "elo_old", "logistic"], "NEW": ["dc_new", "elo_new", "logistic"]}

# 1) Ensemble weights, fitted on the validation window only.
val = run_many(SPECS, VALID, refit=28)
weights = {}
for name, keys in SYSTEMS.items():
    ids = sorted(set.intersection(*[set(val[k].scores) for k in keys]))
    w, _ = fit_weights({k: np.array([val[k].scores[m].probs for m in ids]) for k in keys},
                       np.array([val[keys[0]].scores[m].outcome for m in ids]))
    weights[name] = w
    print(f"{name} weights (fitted on {len(ids)} validation matches): {w}")

# 2) The test window.
test = run_many(SPECS, TEST, refit=28)

def system(name, ids, league=None):
    out = EvaluationResult(label=name)
    for m in ids:
        avail = [k for k in SYSTEMS[name] if m in test[k].scores]
        if not avail: continue
        w = np.array([weights[name][k] for k in avail]); w /= w.sum()
        p = sum(wi * np.array(test[k].scores[m].probs) for wi, k in zip(w, avail))
        s0 = test[avail[0]].scores[m]
        if league and s0.competition != league: continue
        out.scores[m] = MatchScore(m, s0.competition, s0.date, s0.outcome, tuple(p))
    return out

def line(label, base, cand, metric="log_loss"):
    c = paired_comparison(base, cand, metric=metric)
    print(f"  {label:34}{c['n']:>6}{c['baseline']:>10.4f}{c['candidate']:>10.4f}{c['diff']:>+10.4f}"
          f"   [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]   {c['p_better']:.3f}")
    return c

print(f"\n{'TEST 2023-08 .. 2026-05':36}{'n':>6}{'old':>10}{'new':>10}{'diff':>10}   95% CI              P(better)")
all_ids = set().union(*[set(test[k].scores) for k, _, _ in SPECS])
shared = sorted(set.intersection(*[set(test[k].scores) for k, _, _ in SPECS]))
out = {}
out["system_all"] = line("ENSEMBLE, all 5 leagues", system("OLD", shared), system("NEW", shared))
for lg in LEAGUES:
    out[f"system_{lg}"] = line(f"  ensemble, {lg}", system("OLD", shared, lg), system("NEW", shared, lg))
out["dc"] = line("Dixon-Coles alone (1X2)", test["dc_old"], test["dc_new"])
out["dc_ou"] = line("Dixon-Coles alone (O/U 2.5)", test["dc_old"], test["dc_new"], "ou25_log_loss")
out["elo"] = line("Elo alone", test["elo_old"], test["elo_new"])

# 3) Accuracy / Brier / context on the shared set.
o, n = system("OLD", shared), system("NEW", shared)
so, sn = o.summary(), n.summary()
y = np.array([o.scores[m].outcome for m in shared])
from mebet.db.session import new_session
from mebet.db.index import MatchIndex
from mebet.db.models import Competition
import logging; logging.disable(logging.INFO)
s = new_session(); idx = MatchIndex(s); rates = {}
for c in s.query(Competition):
    ms = [m for m in idx.by_competition[c.id] if m.kickoff_date < TEST[0]]
    r = np.array([sum(m.result == x for m in ms) for x in "HDA"], float); rates[c.key] = r / r.sum()
base_ll = np.mean([-np.log(rates[o.scores[m].competition][o.scores[m].outcome]) for m in shared])
print(f"\n  shared matches: {len(shared)}   base-rate log loss: {base_ll:.4f}")
print(f"  OLD  logloss {so['log_loss']:.4f}  brier {so['brier']:.4f}  accuracy {so['accuracy']:.3f}  "
      f"-> {100*(base_ll-so['log_loss'])/base_ll:.2f}% better than base rate")
print(f"  NEW  logloss {sn['log_loss']:.4f}  brier {sn['brier']:.4f}  accuracy {sn['accuracy']:.3f}  "
      f"-> {100*(base_ll-sn['log_loss'])/base_ll:.2f}% better than base rate")

# 4) Coverage: matches the old system could not predict at all.
new_all = system("NEW", sorted(all_ids)); old_all = system("OLD", sorted(all_ids))
extra = [m for m in new_all.scores if m not in old_all.scores]
d = np.array([new_all.scores[m].log_loss + np.log(rates[new_all.scores[m].competition][new_all.scores[m].outcome]) for m in extra])
boots = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(4000)]
print(f"\n  COVERAGE: old system predicted {len(old_all.scores)}, new predicts {len(new_all.scores)} "
      f"(+{len(extra)})")
print(f"  on those {len(extra)} extra matches: new {np.mean([new_all.scores[m].log_loss for m in extra]):.4f} "
      f"vs base-rate fallback {np.mean(d*-1 + [new_all.scores[m].log_loss for m in extra]):.4f}  "
      f"diff {d.mean():+.4f} CI [{np.percentile(boots,2.5):+.4f},{np.percentile(boots,97.5):+.4f}]")
json.dump({"weights": weights, "results": out}, open(__file__.rsplit("/",1)[0] + "/final_results.json", "w"), indent=1, default=str)
