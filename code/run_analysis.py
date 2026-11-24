"""
ECE 225A project: Can statistical diagnostics of stride timing distinguish
neurodegenerative diseases?   (PhysioNet gait-in-neurodegenerative-disease data)

Run this file; it calls the other modules in order:
  data.py         Part 1  load + artifact flagging
  diagnostics.py  Parts 2, 4  per-subject diagnostics, variance of sums
  stat_tests.py   Part 3  group comparisons, assumption tests
  models.py       Parts 5, 6  classification, confounding
  plotting.py     figures

Usage
  python run_analysis.py --data cleaned_gait.csv
  python run_analysis.py --data cleaned_gait.csv --meta subject_info.csv
  python run_analysis.py --demo            # synthetic data, to test the code

Input columns required: time, left_stride, right_stride, subject
Metadata CSV (optional): subject, age, gait_speed (m/s)
Requires: numpy pandas scipy scikit-learn matplotlib
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from config import FEATURES, GROUP_ORDER
from data import artifact_report, flag_artifacts, load_data, make_demo, subject_group
from diagnostics import subject_table
from models import confound_analysis, evaluate, make_models
from plotting import plot_confusions, plot_features, plot_variance_of_sums
from stat_tests import assumption_failures, group_comparisons

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="cleaned gait file (tsv/csv)")
    ap.add_argument("--meta", help="optional CSV: subject, age, gait_speed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

    # ---- Part 1
    if a.demo:
        df, meta = make_demo()
        df["group"] = df["subject"].map(subject_group)
    else:
        df = load_data(a.data)
        meta = pd.read_csv(a.meta) if a.meta else None
    df = flag_artifacts(df)
    rep = artifact_report(df)
    print("=== Artifact flagging ===\n", rep.round(2).to_string(index=False))
    rep.to_csv(out / "artifact_report.csv", index=False)

    # ---- Part 2
    st = subject_table(df)
    st.to_csv(out / "subject_features.csv", index=False)
    print("\n=== Median of diagnostics by group ===")
    print(st.groupby("group")[FEATURES].median().loc[GROUP_ORDER].round(3).to_string())
    plot_features(st, out / "fig_features_by_group.png")

    # ---- Part 3
    kw, pw = group_comparisons(st, FEATURES, a.perms)
    kw.to_csv(out / "kruskal_wallis.csv", index=False)
    pw.to_csv(out / "pairwise_permutation.csv", index=False)
    print("\n=== Kruskal-Wallis across 4 groups (BH-adjusted across features) ===")
    print(kw.round(4).to_string(index=False))
    print("\n=== Pairwise permutation tests, Holm-adjusted (significant only) ===")
    print(pw[pw.p_holm < 0.05].round(4).to_string(index=False) or "none")
    af = assumption_failures(st)
    af.to_csv(out / "assumption_failures.csv", index=False)
    print("\n=== Does each assumption fail within each group? (Wilcoxon vs 0) ===")
    print(af.round(4).to_string(index=False))

    # ---- Part 4
    print("\n=== Variance of sums: observed vs independent prediction (median by group) ===")
    cols = ["pair_var_obs", "pair_var_indep", "pair_var_formula", "pair_ratio",
            "block_var_obs", "block_var_indep", "block_ratio"]
    print(st.groupby("group")[cols].median().loc[GROUP_ORDER].round(5).to_string())
    plot_variance_of_sums(st, out / "fig_variance_of_sums.png")

    # ---- Part 5
    y = st["group"].values
    X = st[FEATURES].values
    models = make_models()
    res, preds = evaluate(models, X, y)
    res.to_csv(out / "model_comparison.csv", index=False)
    print("\n=== Leave-one-person-out results ===")
    print(res.round(3).to_string(index=False))
    print(f"(Huntington majority-class share = {(y == 'Huntington').mean():.3f})")
    plot_confusions(y, preds, out / "fig_confusion_matrices.png")
    lr = models["Logistic regression"].fit(X, y)
    coef = pd.DataFrame(lr[-1].coef_.T, index=FEATURES, columns=lr[-1].classes_)
    coef.to_csv(out / "logreg_coefficients.csv")

    # ---- Part 6
    if meta is not None:
        confound_analysis(st, meta, models, y, out)
    else:
        print("\n(No --meta file given: age / walking-speed confound check skipped.)")
    print(f"\nAll tables and figures written to {out.resolve()}")

if __name__ == "__main__":
    main()
