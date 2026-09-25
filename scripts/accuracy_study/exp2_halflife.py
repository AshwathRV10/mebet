import sys, time; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
specs = [("baseline DC (current)", "dc", OLD_DC)]
for blend_name, blend in [("goals", {}), ("blend .35/.15", {"sot_weight": 0.35, "shots_weight": 0.15})]:
    for hl in (60, 90, 120, 180, 270, 365, 540):
        specs.append((f"{blend_name:14} hl={hl}", "dc", {**blend, "half_life_days": hl}))
t = time.time(); res = run_many(specs, VALID); print(f"ran in {time.time()-t:.0f}s")
table(res, "baseline DC (current)")
