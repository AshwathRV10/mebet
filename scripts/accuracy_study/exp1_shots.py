import sys, time; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
specs = [("baseline DC (current)", "dc", OLD_DC)]
for sot, sh in [(0.25,0),(0.5,0),(0.75,0),(1.0,0),(0,0.5),(0.25,0.25),(0.5,0.25),(0.4,0.4),(0.3,0.6),(0.6,0.4)]:
    specs.append((f"sot={sot} shots={sh}", "dc", {"sot_weight": sot, "shots_weight": sh}))
t = time.time(); res = run_many(specs, VALID); print(f"ran in {time.time()-t:.0f}s")
table(res, "baseline DC (current)")
