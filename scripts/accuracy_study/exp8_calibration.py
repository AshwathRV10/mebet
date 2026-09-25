import sys, datetime as dt, numpy as np; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
from mebet.models.ensemble import fit_weights
DC_NEW = {"sot_weight": 0.35, "shots_weight": 0.15, "half_life_days": 270, "ridge": 1.0, "promoted_offset": 0.3, "rate_unrated_teams": True}
res = run_many([("dc_new", "dc", DC_NEW), ("elo_new", "elo", {"season_regression": 0.1}),
                ("logistic", "logistic", {})], VALID, refit=28)
keys = ["dc_new", "elo_new", "logistic"]
shared = sorted(set.intersection(*[set(r.scores) for r in res.values()]))
split = dt.date(2022, 7, 1)
A = [m for m in shared if res["dc_new"].scores[m].date < split]
B = [m for m in shared if res["dc_new"].scores[m].date >= split]
P = lambda ids: {k: np.array([res[k].scores[m].probs for m in ids]) for k in keys}
Y = lambda ids: np.array([res["dc_new"].scores[m].outcome for m in ids])
w, _ = fit_weights(P(A), Y(A))
mix = lambda ids: sum(w[k] * P(ids)[k] for k in keys)
pa, pb, ya, yb = mix(A), mix(B), Y(A), Y(B)

print("OVERALL: mean predicted vs observed frequency (2022-23, out of sample)")
for i, name in enumerate(["home", "draw", "away"]):
    print(f"  {name:5} predicted {pb[:, i].mean():.3f}   observed {np.mean(yb == i):.3f}   gap {np.mean(yb == i) - pb[:, i].mean():+.3f}")

print("\nRELIABILITY by predicted-probability band (2022-23):")
for i, name in enumerate(["home", "draw", "away"]):
    edges = [0, .15, .25, .35, .45, .55, .65, 1.0]
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pb[:, i] >= lo) & (pb[:, i] < hi)
        if m.sum() >= 40:
            cells.append(f"{pb[m, i].mean():.2f}->{np.mean(yb[m] == i):.2f}(n{m.sum()})")
    print(f"  {name:5} " + "  ".join(cells))

# Would a one-parameter draw correction help? Fit on 2021-22, score on 2022-23.
def adjust(p, k):
    q = p.copy(); q[:, 1] *= k; return q / q.sum(1, keepdims=True)
ll = lambda p, y: -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))
ks = np.linspace(0.85, 1.25, 81)
best_k = ks[np.argmin([ll(adjust(pa, k), ya) for k in ks])]
d = np.log(np.clip(adjust(pb, best_k)[np.arange(len(yb)), yb], 1e-12, 1)) - np.log(np.clip(pb[np.arange(len(yb)), yb], 1e-12, 1))
boots = [(-d[np.random.default_rng(i).integers(0, len(d), len(d))]).mean() for i in range(4000)]
print(f"\nDRAW CORRECTION: best multiplier on 2021-22 = {best_k:.3f}")
print(f"  2022-23 log loss {ll(pb, yb):.4f} -> {ll(adjust(pb, best_k), yb):.4f}   "
      f"diff {-d.mean():+.4f}  CI [{np.percentile(boots,2.5):+.4f},{np.percentile(boots,97.5):+.4f}]")
