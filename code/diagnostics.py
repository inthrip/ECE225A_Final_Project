"""Parts 2 and 4: per-subject diagnostics and variance-of-sums quantities."""
import numpy as np
import pandas as pd
from scipy import stats

from config import FEATURES


def series_stats(x, t):
    """Statistics for one stride-time series (NaN = removed artifact)."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    ok = ~np.isnan(x)
    xv, tv = x[ok], t[ok]
    out = {"stride_mean": xv.mean(), "stride_sd": xv.std(ddof=1)}
    out["stride_cv"] = out["stride_sd"] / out["stride_mean"]
    # independence: lag-1 autocorrelation using only truly consecutive valid pairs
    a, b = x[:-1], x[1:]
    m = ~np.isnan(a) & ~np.isnan(b)
    out["lag1"] = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 3 else np.nan
    # normality: skewness and excess kurtosis (tail weight)
    out["skew"] = stats.skew(xv)
    out["ex_kurtosis"] = stats.kurtosis(xv)
    # constant mean: linear drift, % of mean per minute
    slope = stats.linregress(tv, xv).slope
    out["drift_pct_per_min"] = 100 * slope * 60 / out["stride_mean"]
    # constant variance: log ratio of 2nd-half to 1st-half variance
    h = len(xv) // 2
    v1, v2 = xv[:h].var(ddof=1), xv[h:].var(ddof=1)
    out["log_var_ratio"] = np.log(v2 / v1)
    return out


def block_sum_var(x, k):
    """Variance of sums of k consecutive strides (non-overlapping, NaN-free windows)."""
    sums = [x[i:i + k].sum() for i in range(0, len(x) - k + 1, k)
            if not np.isnan(x[i:i + k]).any()]
    return np.var(sums, ddof=1) if len(sums) >= 3 else np.nan


def pair_sum_var(x):
    """Variance of X_i + X_{i+1} over all valid consecutive pairs (overlapping)."""
    s = x[:-1] + x[1:]
    s = s[~np.isnan(s)]
    return np.var(s, ddof=1) if len(s) >= 3 else np.nan


def subject_table(df, k_block=10):
    rows = []
    for subj, d in df.groupby("subject"):
        per_foot, sum_rows = [], []
        for foot in ("left", "right"):
            x = d[f"{foot}_stride_clean"].values
            per_foot.append(series_stats(x, d["time"].values))
            v = np.nanvar(x, ddof=1)
            sum_rows.append({
                "pair_obs": pair_sum_var(x), "pair_indep": 2 * v,
                "block_obs": block_sum_var(x, k_block), "block_indep": k_block * v,
                "rho": per_foot[-1]["lag1"], "var": v})
        row = {"subject": subj, "group": d["group"].iloc[0]}
        for f in FEATURES:  # average the two feet
            row[f] = np.nanmean([p[f] for p in per_foot])
        # Part 4 quantities (average of feet)
        sr = pd.DataFrame(sum_rows).mean()
        row["pair_var_obs"], row["pair_var_indep"] = sr.pair_obs, sr.pair_indep
        row["pair_var_formula"] = 2 * sr["var"] * (1 + sr.rho)   # V+V+2Cov with Cov = rho*V
        row["block_var_obs"], row["block_var_indep"] = sr.block_obs, sr.block_indep
        row["pair_ratio"] = sr.pair_obs / sr.pair_indep
        row["block_ratio"] = sr.block_obs / sr.block_indep
        rows.append(row)
    return pd.DataFrame(rows)
