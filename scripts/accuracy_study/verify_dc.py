import datetime as dt, sys, time
import numpy as np
from scipy.optimize import approx_fprime, minimize as real_minimize
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mebet.models.dixon_coles as dcmod
from dc_reference import DixonColesModel as RefDC
from mebet.db.session import session_scope
from mebet.db.index import MatchIndex
from mebet.db.repository import AsOfRepository
from mebet.db.models import Competition, Match
from mebet.models.base import TrainingContext, PredictionRequest

# 1) Gradient check: intercept the objective handed to the optimiser.
grad_errs = []
def checking_minimize(fun, x0, **kw):
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = x0 + rng.normal(0, 0.15, size=x0.shape)
        f, g = fun(x)
        num = approx_fprime(x, lambda z: fun(z)[0], 1e-6)
        grad_errs.append(np.max(np.abs(g - num)) / max(1.0, np.max(np.abs(num))))
    return real_minimize(fun, x0, **kw)

with session_scope() as s:
    idx = MatchIndex(s)
    comp = s.query(Competition).filter_by(key="ENG.1").one()
    for variant in [dict(), dict(sot_weight=0.5, shots_weight=0.2, ridge=5.0, promoted_offset=0.2)]:
        dcmod.minimize = checking_minimize
        as_of = dt.datetime(2022, 11, 1)
        ctx = TrainingContext(repo=AsOfRepository(s, as_of, index=idx), competition_ids=[comp.id], as_of=as_of, index=idx)
        dcmod.DixonColesModel(**variant).fit(ctx)
        dcmod.minimize = real_minimize
    print(f"1) GRADIENT: max relative error vs finite differences = {max(grad_errs):.2e}  "
          f"({'PASS' if max(grad_errs) < 1e-4 else 'FAIL'})")

    # 2) Equivalence with the reference model at default settings.
    diffs, t_new, t_ref = [], 0.0, 0.0
    for comp_key in ("ENG.1", "ESP.1", "GER.1"):
        c = s.query(Competition).filter_by(key=comp_key).one()
        for as_of in (dt.datetime(2019, 10, 1), dt.datetime(2022, 2, 1), dt.datetime(2025, 3, 1)):
            # half_life_days=180 on the context too: the reference model reads its half-life
            # from the context (the precedence bug fixed in the new model).
            ctx = TrainingContext(repo=AsOfRepository(s, as_of, index=idx), competition_ids=[c.id], as_of=as_of, index=idx, half_life_days=180)
            t = time.time(); new = dcmod.DixonColesModel(half_life_days=180, sot_weight=0, shots_weight=0, ridge=0, promoted_offset=0, rate_unrated_teams=False).fit(ctx); t_new += time.time() - t
            t = time.time(); ref = RefDC().fit(ctx); t_ref += time.time() - t
            upcoming = (s.query(Match).filter(Match.competition_id == c.id, Match.kickoff_date >= as_of.date())
                        .order_by(Match.kickoff_date).limit(15).all())
            for m in upcoming:
                req = PredictionRequest(m.home_team_id, m.away_team_id, c.id, as_of)
                a, b = new.predict(req), ref.predict(req)
                if a.sufficient_data and b.sufficient_data:
                    diffs.append(max(abs(a.outcome_probs[k] - b.outcome_probs[k]) for k in a.outcome_probs))
    print(f"2) EQUIVALENCE: {len(diffs)} predictions, max |prob diff| vs old model = {max(diffs):.2e}  "
          f"({'PASS' if max(diffs) < 2e-3 else 'FAIL'})")
    print(f"3) SPEED: 9 fits  new {t_new:.2f}s  vs  old {t_ref:.2f}s  -> {t_ref/t_new:.1f}x faster")
