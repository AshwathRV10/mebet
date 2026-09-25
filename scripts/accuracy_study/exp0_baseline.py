import sys, time; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
t = time.time()
res = run_many([("baseline DC (current)", "dc", OLD_DC),
                ("baseline Elo (current)", "elo", OLD_ELO)], VALID)
print(f"ran in {time.time()-t:.0f}s | fits: {res['baseline DC (current)'].fits} | "
      f"declined: DC {res['baseline DC (current)'].declined}, Elo {res['baseline Elo (current)'].declined}")
table(res, "baseline DC (current)")
