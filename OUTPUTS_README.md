# Output Files Guide

This document describes every file written by `run_analysis.py` into the results folder (`results/` by default, or whatever is passed with `--out`).

**Project question:** Does stride timing carry enough statistical signal to distinguish Parkinson's, Huntington's, ALS and healthy controls, and do the standard assumptions (independence, normality, constant variance) fail differently in each disease?

**Data:** PhysioNet *Gait in Neurodegenerative Disease* database (Hausdorff et al., 2000).

## How to reproduce

```
python run_analysis.py --data cleaned_data.csv                      # core outputs
python run_analysis.py --data cleaned_data.csv --meta subject_info.csv   # + confounding outputs
python run_analysis.py --demo                                       # synthetic data, to test the setup
```

- Required input columns: `time`, `left_stride`, `right_stride`, `subject`.
- Optional metadata CSV columns: `subject`, `age`, `gait_speed` (m/s).
- Random seeds are fixed, so permutation p-values and bootstrap CIs are reproducible for a given input file.
- Requires: numpy, pandas, scipy, scikit-learn, matplotlib.
- Other options: `--out DIR` (output folder), `--perms N` (number of permutations, default 10000).

## Code layout

Run only `run_analysis.py`; it imports the other modules. All files must sit in the same folder.

| File | Responsibility | Report section it supports |
|---|---|---|
| `config.py` | Group names, feature lists, shared seeded random generator | n/a |
| `data.py` | Loading, group assignment, artifact flagging, artifact report, synthetic demo data | Data cleaning |
| `diagnostics.py` | Per-subject statistics (lag-1, skew, kurtosis, drift, variance ratio) and variance-of-sums quantities | Assumption diagnostics; V(X+Y) demonstration |
| `stat_tests.py` | Holm/BH correction, permutation tests, Kruskal-Wallis, per-group assumption tests | Group comparisons |
| `models.py` | Baseline, single-feature, logistic regression and random forest models; leave-one-person-out evaluation; confounding analysis | Classification; confounding |
| `plotting.py` | All figures | Figures |
| `run_analysis.py` | Command-line entry point that runs the steps in order and writes the outputs | n/a |

Split of work: one partner owns `data.py`, `diagnostics.py` and `stat_tests.py`; the other owns `models.py`. Each reviews the other's files.

## Conventions used in all tables

- **Groups:** `Control`, `Parkinson`, `Huntington`, `ALS`.
- **Unit of analysis:** one row per subject. Left and right foot statistics are averaged to a single value per subject.
- **Multiple comparisons:** Where a table has both a raw and a corrected p-value, use the corrected one (`p_holm` or `q_bh`).
- **Stride units:** seconds.

---

## Tables

### `artifact_report.csv`: cleaning summary
One row per group.

| Column | Meaning |
|---|---|
| `group` | Disease group |
| `n_subjects` | Number of subjects |
| `n_strides` | Total strides examined (left + right) |
| `pct_flagged` | Percent of strides removed by the artifact rule |

**Artifact rule.** Applied per subject and per foot. A stride is flagged if it is more than 5 robust SDs (1.4826 x MAD) from the subject's median, or is shorter than 0.5 s or longer than 2.5 s. Flagged strides are set to missing (not deleted), so lag-1 correlation and block sums use only truly consecutive strides. The thresholds are our own choices.

### `subject_features.csv`: per-subject summary statistics
One row per subject. This is the input to every downstream test and model.

| Column | Meaning |
|---|---|
| `subject`, `group` | Subject ID and disease group |
| `stride_mean` | Mean stride time |
| `stride_sd` | Standard deviation of stride time |
| `stride_cv` | `stride_sd / stride_mean` (relative variability) |
| `lag1` | Lag-1 autocorrelation of consecutive strides. Independence assumption: 0 means independent |
| `skew` | Skewness. Normality assumption: 0 for symmetric |
| `ex_kurtosis` | Excess kurtosis (tail weight). Normality assumption: 0 for normal |
| `drift_pct_per_min` | Linear trend of stride time, as % of the mean per minute. Constant-mean assumption: 0 means no drift |
| `log_var_ratio` | log(variance of 2nd half / variance of 1st half). Constant-variance assumption: 0 means stable |
| `pair_var_obs` | Observed variance of X_i + X_{i+1} over consecutive stride pairs |
| `pair_var_indep` | Independent prediction, 2 x Var(X) |
| `pair_var_formula` | V(X)+V(Y)+2Cov(X,Y) with Cov estimated from lag-1 correlation, i.e. 2 x Var(X) x (1 + lag1) |
| `block_var_obs` | Observed variance of sums of 10 consecutive strides (non-overlapping blocks) |
| `block_var_indep` | Independent prediction, 10 x Var(X) |
| `pair_ratio` | `pair_var_obs / pair_var_indep` |
| `block_ratio` | `block_var_obs / block_var_indep` |

**Reading the ratios.** A ratio of 1 means variances add as if strides were independent. A ratio above 1 means correlated strides make total walking time more variable than the independent prediction.

### `assumption_failures.csv`: does each assumption fail within each group?
One row per group x diagnostic (4 x 5 = 20 rows).

| Column | Meaning |
|---|---|
| `group` | Disease group |
| `diagnostic` | One of `lag1`, `skew`, `ex_kurtosis`, `drift_pct_per_min`, `log_var_ratio` |
| `median` | Median of that diagnostic across the group's subjects |
| `p_wilcoxon_vs_0` | Wilcoxon signed-rank test of the group's values against 0 |
| `p_holm` | Holm-corrected p-value across all 20 tests (use this one) |

A small `p_holm` means the corresponding assumption is violated in that group. The sign of `median` shows the direction (e.g. positive `lag1` means long strides tend to follow long strides).

### `kruskal_wallis.csv`: do the four groups differ at all?
One row per feature (8 rows).

| Column | Meaning |
|---|---|
| `feature` | Feature tested |
| `H` | Kruskal-Wallis statistic |
| `p_kw` | Raw p-value |
| `q_bh` | Benjamini-Hochberg corrected p-value across the 8 features (use this one) |

This is an omnibus test. It says at least one group differs, not which.

### `pairwise_permutation.csv`: which groups differ?
One row per feature x group pair (8 x 6 = 48 rows).

| Column | Meaning |
|---|---|
| `feature` | Feature tested |
| `pair` | Groups compared, e.g. `Control vs ALS` |
| `median_1`, `median_2` | Median of the first and second group |
| `p_mwu` | Mann-Whitney U p-value (uncorrected, for reference) |
| `p_perm` | Permutation-test p-value on the difference in medians (10,000 permutations by default) |
| `p_holm` | Holm-corrected `p_perm` across all 48 tests (use this one) |

### `model_comparison.csv`: classification performance
One row per model, evaluated with leave-one-person-out cross-validation (one row per subject, so this equals leave-one-out).

| Column | Meaning |
|---|---|
| `model` | `Majority baseline`, `Single-feature rule`, `Logistic regression`, `Random forest` |
| `accuracy`, `accuracy_95CI` | Fraction correct, with bootstrap 95% CI |
| `balanced_acc`, `balanced_acc_95CI` | Mean per-class recall, with CI |
| `macro_F1`, `macro_F1_95CI` | Unweighted mean F1 across the 4 classes, with CI |

- **Majority baseline:** always predicts the largest group (Huntington's, about 31% of subjects). A model shows signal only if its CI is clearly above this.
- **Single-feature rule:** a depth-2 decision tree on one feature. The feature is chosen inside each training fold, so there is no leakage.
- **Logistic regression:** multinomial, standardized features.
- **Random forest:** 500 trees, the more flexible comparison model.
- CIs come from resampling subjects' predictions 2,000 times. With roughly 60 subjects they will be wide.

### `logreg_coefficients.csv`: logistic regression weights
Rows are the 8 features, columns are the 4 classes (`ALS`, `Control`, `Huntington`, `Parkinson`). The first (unnamed) column holds the feature names. The model is fit on all subjects with standardized features, so magnitudes are comparable across features. A larger positive value pushes predictions toward that class, and a larger negative value pushes them away. These are descriptive, not significance tests.

---

## Confounding outputs (only written when `--meta` is supplied)

### `confound_associations.csv`
| Column | Meaning |
|---|---|
| `covariate` | `age` or `gait_speed` |
| `test` | Either `KW across groups` (does the covariate differ between groups?) or `Spearman with <feature>` |
| `stat` | Kruskal-Wallis H or Spearman rho |
| `p` | Uncorrected p-value |

### `confound_adjusted_kw.csv`
| Column | Meaning |
|---|---|
| `feature` | Feature tested |
| `p_kw_raw` | Kruskal-Wallis p-value on the original feature |
| `p_kw_adjusted` | Kruskal-Wallis p-value after regressing out the covariates linearly |

If a feature's group difference disappears after adjustment, it may be explained by age or walking speed instead of disease.

### `confound_models.csv`
Logistic regression, leave-one-person-out, on subjects with complete metadata.

| Column | Meaning |
|---|---|
| `inputs (logistic reg.)` | `Covariates only`, `Stride features`, or `Stride features + covariates` |
| `n` | Subjects used |
| `accuracy`, `balanced_acc` | Performance of that input set |

If covariates alone predict group well, or adding them changes performance a lot, confounding is a concern.

---

## Figures

| File | Contents |
|---|---|
| `fig_features_by_group.png` | Boxplots with individual subjects overlaid, one panel per feature, by group. Dashed line at 0 marks the null value of each assumption diagnostic |
| `fig_variance_of_sums.png` | `pair_ratio` and `block_ratio` by group (log scale). Red dashed line at 1 is the independence prediction |
| `fig_confusion_matrices.png` | Leave-one-person-out confusion matrix for each of the four models (rows are true class, columns are predicted class) |

## Caveats to keep in mind

- Roughly 60 subjects means low power and wide confidence intervals, especially for per-class results.
- The artifact thresholds are our own choice. Check `pct_flagged` for group imbalance, because the rule may remove real irregular strides more often in highly variable groups.
- Confound analysis is only as good as the metadata supplied. Without `--meta` it is skipped entirely.
