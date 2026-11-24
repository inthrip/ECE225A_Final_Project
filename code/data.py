"""Part 1: loading data, assigning groups, flagging artifact strides, demo data."""
import numpy as np
import pandas as pd
from scipy import stats

from config import GROUP_ORDER, GROUP_PREFIXES, RNG


def subject_group(name):
    s = str(name).lower()
    for prefix, g in GROUP_PREFIXES.items():
        if s.startswith(prefix):
            return g
    raise ValueError(f"Cannot infer group from subject id '{name}'")


def load_data(path):
    with open(path) as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    df["group"] = df["subject"].map(subject_group)
    return df.sort_values(["subject", "time"]).reset_index(drop=True)


def flag_artifacts(df, k=5.0, lo=0.5, hi=2.5):
    """
    Documented artifact rule (applied separately to each subject and foot):
      a stride is an artifact if
        (i)  |stride - subject median| > k * (1.4826 * MAD)   [robust z-score, k=5], or
        (ii) stride < lo or stride > hi seconds               [non-physiological].
    Flagged strides are set to NaN (kept in place so that only truly consecutive
    strides are ever used for lag-1 / block-sum calculations).
    """
    df = df.copy()
    for foot in ("left", "right"):
        col = f"{foot}_stride"
        g = df.groupby("subject")[col]
        med = g.transform("median")
        mad = g.transform(lambda x: stats.median_abs_deviation(x, scale="normal"))
        z = (df[col] - med).abs() / mad.replace(0, np.nan)
        bad = (z > k) | (df[col] < lo) | (df[col] > hi)
        df[f"{col}_flag"] = bad
        df[f"{col}_clean"] = df[col].where(~bad)
    return df


def artifact_report(df):
    rows = []
    for g in GROUP_ORDER:
        d = df[df.group == g]
        rows.append({"group": g, "n_subjects": d.subject.nunique(), "n_strides": len(d) * 2,
                     "pct_flagged": 100 * (d.left_stride_flag.sum() + d.right_stride_flag.sum())
                                    / (2 * len(d))})
    return pd.DataFrame(rows)


def make_demo(n_strides=250):
    spec = {"control": (16, 1.10, .025, .1), "park": (15, 1.12, .035, .25),
            "hunt": (20, 1.20, .09, .5), "als": (13, 1.30, .07, .35)}
    rows = []
    for pre, (n, mu, sd, rho) in spec.items():
        for i in range(1, n + 1):
            m = mu * RNG.normal(1, .05)
            for foot in range(2):
                pass
            e = [RNG.normal()]
            for _ in range(n_strides - 1):
                e.append(rho * e[-1] + np.sqrt(1 - rho ** 2) * RNG.normal())
            L = m + sd * np.array(e)
            R = m + sd * np.roll(np.array(e), 1) + RNG.normal(0, .01, n_strides)
            t = 20 + np.cumsum(L)
            rows.append(pd.DataFrame({"time": t, "left_stride": L, "right_stride": R,
                                      "subject": f"{pre}{i}"}))
    df = pd.concat(rows, ignore_index=True)
    df.loc[RNG.choice(len(df), 40, replace=False), "left_stride"] = 4.0   # fake artifacts
    meta = pd.DataFrame({"subject": df.subject.unique()})
    meta["age"] = RNG.integers(25, 80, len(meta))
    meta["gait_speed"] = RNG.normal(1.1, .2, len(meta))
    return df, meta
