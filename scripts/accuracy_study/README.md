# Accuracy study, September 2026

The experiments behind the current model defaults, in the order they were run.
Every choice was made on the **validation window** (2021-22 and 2022-23, all
five leagues). The **test window** (2023-08 to 2026-05) was scored once, by
`final_test.py`, after every choice was fixed.

| script | question |
|---|---|
| `verify_dc.py` | does the rewritten fit match the old one at neutral settings, and is its gradient correct? |
| `exp0_baseline.py` | where do the old models stand on validation? |
| `exp1_shots.py` | should ratings be fitted on shots as well as goals, and in what blend? |
| `exp2_halflife.py` | how fast should old matches be forgotten, with and without shots? |
| `exp3_prior.py` | does a shrinkage prior help teams the model already rates? |
| `exp4_coverage.py` | does it beat base rates on matches the old model declined? |
| `exp5_elo.py`, `exp6_elo_edge.py` | Elo K-factor, home advantage and season regression |
| `exp7_ensemble.py` | which models belong in the ensemble, with weights fitted out of sample |
| `exp8_calibration.py` | are draws and extremes calibrated; does a draw correction help? |
| `final_test.py` | the one held-out comparison of old against new |

Run any of them from the repository root, e.g.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/accuracy_study/exp1_shots.py

Results are cached under `cache/` (not committed). `lab.py` holds the shared
driver and the window definitions. `dc_reference.py` is the pre-study
Dixon-Coles, kept only as the reference for `verify_dc.py`.
