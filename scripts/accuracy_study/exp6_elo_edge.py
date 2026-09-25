import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import *
specs = [("baseline Elo (current)", "elo", OLD_ELO)] + [
    (f"k=20 home=60 regress={sr}", "elo", {**OLD_ELO, "season_regression": sr}) for sr in (0.0, 0.05, 0.1, 0.15)]
table(run_many(specs, VALID), "baseline Elo (current)")
