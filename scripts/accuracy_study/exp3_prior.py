import sys, time; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
B = {"sot_weight": 0.35, "shots_weight": 0.15, "half_life_days": 270}
specs = [("baseline DC (current)", "dc", OLD_DC), ("blend+hl270 (no prior)", "dc", B)]
for ridge in (1, 3, 10, 30):
    for off in (0.0, 0.15, 0.3):
        specs.append((f"  ridge={ridge:<3} promoted={off}", "dc", {**B, "ridge": ridge, "promoted_offset": off}))
t = time.time(); res = run_many(specs, VALID); print(f"ran in {time.time()-t:.0f}s\n")
print("vs CURRENT baseline:"); table(res, "baseline DC (current)")
print("\nvs blend+hl270 (isolates the prior):"); table({k: v for k, v in res.items() if k != "baseline DC (current)"}, "blend+hl270 (no prior)")
print("\ncoverage (matches predicted / declined):")
for k in ("baseline DC (current)", "blend+hl270 (no prior)", "  ridge=10  promoted=0.15"):
    print(f"  {k:30} {len(res[k].scores):>5} predicted, {res[k].declined:>3} declined")
