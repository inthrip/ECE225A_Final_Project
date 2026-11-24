"""Shared constants: group names, feature lists, and the single random-number generator."""
import numpy as np

RNG = np.random.default_rng(225)   # one seeded generator shared by all modules (reproducible)

GROUP_ORDER = ["Control", "Parkinson", "Huntington", "ALS"]
GROUP_PREFIXES = {"control": "Control", "park": "Parkinson",
                  "hunt": "Huntington", "als": "ALS"}
FEATURES = ["stride_mean", "stride_sd", "stride_cv", "lag1", "skew",
            "ex_kurtosis", "drift_pct_per_min", "log_var_ratio"]
# diagnostics whose null value is 0 under iid / normal / stationary assumptions
ASSUMPTION_FEATURES = ["lag1", "skew", "ex_kurtosis", "drift_pct_per_min", "log_var_ratio"]
