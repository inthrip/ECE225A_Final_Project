"""Parts 5 and 6: classifiers, leave-one-person-out evaluation, confounding analysis."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import FEATURES, GROUP_ORDER, RNG


class SingleFeatureRule(BaseEstimator, ClassifierMixin):
    """Picks (inside each training fold) the one feature whose depth-2 tree fits best."""
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def fit(self, X, y):
        X = np.asarray(X)
        best = -1
        for j in range(X.shape[1]):
            t = DecisionTreeClassifier(max_depth=self.max_depth, random_state=0).fit(X[:, [j]], y)
            acc = t.score(X[:, [j]], y)
            if acc > best:
                best, self.j_, self.tree_ = acc, j, t
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(np.asarray(X)[:, [self.j_]])


def make_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Single-feature rule": make_pipeline(imp(), SingleFeatureRule()),
        "Logistic regression": make_pipeline(imp(), StandardScaler(),
                                             LogisticRegression(C=1.0, max_iter=5000)),
        "Random forest": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=0)),
    }


def bootstrap_ci(y, pred, metric, n=2000):
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        vals.append(metric(y[idx], pred[idx]))
    return np.percentile(vals, [2.5, 97.5])


def evaluate(models, X, y):
    rows, preds = [], {}
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=LeaveOneOut())
        preds[name] = p
        acc = lambda a, b: (a == b).mean()
        bal = balanced_accuracy_score
        f1 = lambda a, b: f1_score(a, b, average="macro")
        row = {"model": name}
        for label, fn in [("accuracy", acc), ("balanced_acc", bal), ("macro_F1", f1)]:
            lo, hi = bootstrap_ci(y, p, fn)
            row[label] = fn(y, p)
            row[label + "_95CI"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(row)
    return pd.DataFrame(rows), preds


def confound_analysis(st, meta, models, y, outdir):
    st = st.merge(meta, on="subject", how="left")
    covs = [c for c in ("age", "gait_speed") if c in st and st[c].notna().sum() > 10]
    if not covs:
        print("No usable age / gait_speed columns in metadata; skipping confound analysis.")
        return
    print(f"\nCovariates available: {covs}")
    # (a) do covariates differ by group?  correlate with each feature?
    rows = []
    for c in covs:
        samples = [st.loc[st.group == g, c].dropna() for g in GROUP_ORDER]
        rows.append({"covariate": c, "test": "KW across groups",
                     "stat": stats.kruskal(*samples).statistic, "p": stats.kruskal(*samples).pvalue})
        for f in FEATURES:
            d = st[[c, f]].dropna()
            r, p = stats.spearmanr(d[c], d[f])
            rows.append({"covariate": c, "test": f"Spearman with {f}", "stat": r, "p": p})
    pd.DataFrame(rows).to_csv(outdir / "confound_associations.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # (b) feature group-differences before vs after regressing out the covariates
    d = st.dropna(subset=covs).copy()
    Z = np.column_stack([np.ones(len(d))] + [d[c].values for c in covs])
    rows = []
    for f in FEATURES:
        beta, *_ = np.linalg.lstsq(Z, d[f].values, rcond=None)
        d[f + "_resid"] = d[f].values - Z @ beta
        p0 = stats.kruskal(*[d.loc[d.group == g, f] for g in GROUP_ORDER]).pvalue
        p1 = stats.kruskal(*[d.loc[d.group == g, f + "_resid"] for g in GROUP_ORDER]).pvalue
        rows.append({"feature": f, "p_kw_raw": p0, "p_kw_adjusted": p1})
    adj = pd.DataFrame(rows)
    adj.to_csv(outdir / "confound_adjusted_kw.csv", index=False)
    print("\nKruskal-Wallis before vs after removing linear effect of", covs)
    print(adj.round(4).to_string(index=False))

    # (c) classification: covariates alone, and features + covariates
    d = d.reset_index(drop=True)
    yd = d["group"].values
    lr = models["Logistic regression"]
    res = []
    for label, cols in [("Covariates only", covs), ("Stride features", FEATURES),
                        ("Stride features + covariates", FEATURES + covs)]:
        p = cross_val_predict(lr, d[cols].values, yd, cv=LeaveOneOut())
        res.append({"inputs (logistic reg.)": label, "n": len(d),
                    "accuracy": (p == yd).mean(),
                    "balanced_acc": balanced_accuracy_score(yd, p)})
    res = pd.DataFrame(res)
    res.to_csv(outdir / "confound_models.csv", index=False)
    print("\nDoes adding covariates change performance?")
    print(res.round(3).to_string(index=False))
