import sys, time; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
specs = [("baseline Elo (current)", "elo", OLD_ELO)]
for k in (10, 15, 20, 30):
    for ha in (40, 60, 80):
        for sr in (0.1, 0.25, 0.4):
            specs.append((f"k={k:<3} home={ha:<3} regress={sr}", "elo", {"k_factor": k, "home_advantage": ha, "season_regression": sr}))
t = time.time(); res = run_many(specs, VALID); print(f"ran in {time.time()-t:.0f}s")
base = res["baseline Elo (current)"]
shared = set(base.scores)
for r in res.values(): shared &= set(r.scores)
ranked = sorted(res.items(), key=lambda kv: kv[1].summary(shared)["log_loss"])
top = dict([("baseline Elo (current)", base)] + [kv for kv in ranked[:8] if kv[0] != "baseline Elo (current)"])
print("top 8 of", len(res) - 1, "configs:"); table(top, "baseline Elo (current)")
print("\nworst 3:"); table(dict([("baseline Elo (current)", base)] + ranked[-3:]), "baseline Elo (current)")
