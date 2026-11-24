"""Part 3: multiple-comparison correction, permutation tests, group comparisons."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from config import ASSUMPTION_FEATURES, GROUP_ORDER, RNG




def multipletests(p, method="holm"):
    """Minimal Holm / Benjamini-Hochberg adjustment (returns (None, adjusted_p))."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    if method == "holm":
        vals = np.maximum.accumulate((n - np.arange(n)) * p[order])
    else:  # fdr_bh
        vals = np.minimum.accumulate((n / (np.arange(n) + 1) * p[order])[::-1])[::-1]
    adj[order] = np.minimum(vals, 1.0)
    return None, adj


def perm_test_median_diff(a, b, n_perm=10000):
    obs = abs(np.median(a) - np.median(b))
    pooled = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n_perm):
        RNG.shuffle(pooled)
        if abs(np.median(pooled[:na]) - np.median(pooled[na:])) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def group_comparisons(st, features, n_perm=10000):
    kw_rows, pw_rows = [], []
    for f in features:
        samples = [st.loc[st.group == g, f].dropna().values for g in GROUP_ORDER]
        kw_rows.append({"feature": f, "H": stats.kruskal(*samples).statistic,
                        "p_kw": stats.kruskal(*samples).pvalue})
        for (i, g1), (j, g2) in itertools.combinations(enumerate(GROUP_ORDER), 2):
            a, b = samples[i], samples[j]
            pw_rows.append({"feature": f, "pair": f"{g1} vs {g2}",
                            "median_1": np.median(a), "median_2": np.median(b),
                            "p_mwu": stats.mannwhitneyu(a, b).pvalue,
                            "p_perm": perm_test_median_diff(a, b, n_perm)})
    kw, pw = pd.DataFrame(kw_rows), pd.DataFrame(pw_rows)
    kw["q_bh"] = multipletests(kw.p_kw, method="fdr_bh")[1]            # across features
    pw["p_holm"] = multipletests(pw.p_perm, method="holm")[1]           # across ALL pairs x features
    return kw, pw


def assumption_failures(st):
    """Per group: is each diagnostic different from its iid/normal/stationary null of 0?"""
    rows = []
    for g in GROUP_ORDER:
        for f in ASSUMPTION_FEATURES:
            v = st.loc[st.group == g, f].dropna()
            rows.append({"group": g, "diagnostic": f, "median": v.median(),
                         "p_wilcoxon_vs_0": stats.wilcoxon(v).pvalue})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out.p_wilcoxon_vs_0, method="holm")[1]
    return out
