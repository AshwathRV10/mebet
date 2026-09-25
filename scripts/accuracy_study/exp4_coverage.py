import sys, time, numpy as np; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
B = {"sot_weight": 0.35, "shots_weight": 0.15, "half_life_days": 270}
specs = [("baseline DC (current)", "dc", OLD_DC)]
for ridge, off in [(1, 0.0), (1, 0.15), (1, 0.3), (3, 0.3), (10, 0.3)]:
    specs.append((f"ridge={ridge} promoted={off} +unrated", "dc", {**B, "ridge": ridge, "promoted_offset": off, "rate_unrated_teams": True}))
res = run_many(specs, VALID)

# Honest fallback for a declined match: that league's pre-validation outcome frequencies.
import logging; logging.disable(logging.INFO)
from mebet.db.session import new_session
from mebet.db.index import MatchIndex
from mebet.db.models import Competition
s = new_session(); idx = MatchIndex(s)
rates = {}
for c in s.query(Competition):
    ms = [m for m in idx.by_competition[c.id] if m.kickoff_date < VALID[0]]
    r = np.array([sum(m.result == x for m in ms) for x in "HDA"], float); rates[c.key] = r / r.sum()

base_ids = set(res["baseline DC (current)"].scores)
print(f"{'config':34}{'extra n':>8}{'model LL':>10}{'base-rate LL':>13}{'diff':>9}   verdict")
for label, r in res.items():
    if label.startswith("baseline"): continue
    extra = [r.scores[m] for m in r.scores if m not in base_ids]
    ll_m = np.array([x.log_loss for x in extra])
    ll_b = np.array([-np.log(rates[x.competition][x.outcome]) for x in extra])
    d = ll_m - ll_b
    boots = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{label:34}{len(extra):>8}{ll_m.mean():>10.4f}{ll_b.mean():>13.4f}{d.mean():>+9.4f}   CI [{lo:+.3f},{hi:+.3f}]")
