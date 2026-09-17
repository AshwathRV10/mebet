"""mebet - a local-first, evidence-based sports prediction engine.

Design rules enforced throughout this package:

1. No fabricated data. Every stored observation carries the source it came
   from and the time it was retrieved. If a source cannot be reached, the
   absence is recorded as an absence -- never filled in with a guess.
2. No betting odds anywhere in the feature or model path. See
   ``mebet.normalize.ODDS_COLUMN_DENYLIST``.
3. No leakage. Every feature read goes through ``AsOfRepository``, which
   requires an explicit cutoff and filters on it in SQL.
"""

__version__ = "0.1.0"
