"""Friedman test + exact post-hoc Wilcoxon signed-rank tests on a struct_sim-style CSV.

Expected structure: first column = statistic ("mean", "std", ...), then a
"qbp_name" column, then one column per prompt condition. Only the "mean" rows
are used and the "overall" row is dropped. By default, every condition column
with a value for all QBPs is compared (e.g. a column that only covers some
QBPs is skipped); use --conditions to pick columns explicitly.

Usage: python stat_sign.py [csv] [--conditions COL [COL ...]] [--name-col qbp_name]
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def analyse(csv_path, conditions=None, name_col="cpw_name", stat="mean", drop_rows=("overall",), verbose=True):
    """Friedman test across conditions + exact post-hoc Wilcoxon signed-rank tests.

    csv_path:   CSV whose first column selects the row type (`stat`) and which has a `name_col` column.
    conditions: condition columns to compare; default = all columns with a value for every QBP.
    drop_rows:  QBP names to exclude (e.g. aggregate rows).
    verbose:    print a report.

    Returns (friedman, posthoc): a dict (chi2, df, p, kendall_w, mean_ranks)
    and a DataFrame with one row per pair (W, median_diff, p, p_holm, p_bonferroni).
    """
    df = pd.read_csv(csv_path, index_col=0)
    qbps = df.loc[stat].set_index(name_col).drop(index=list(drop_rows), errors="ignore")
    if conditions is None:
        conditions = [c for c in qbps.columns if qbps[c].notna().all()]
    if len(conditions) < 3:
        raise ValueError(f"Friedman test needs at least 3 conditions, got {list(conditions)}")
    data = qbps[list(conditions)].astype(float)
    n, k = data.shape

    chi2, p = stats.friedmanchisquare(*(data[c] for c in conditions))
    friedman = {
        "chi2": chi2, "df": k - 1, "p": p,
        "kendall_w": chi2 / (n * (k - 1)),
        "mean_ranks": dict(zip(conditions, stats.rankdata(data.values, axis=1).mean(axis=0))),
    }

    rows = []
    for a, b in combinations(conditions, 2):
        w = stats.wilcoxon(data[b], data[a], method="auto")  # exact if n <= 50 and no zeros, else normal approx. (SciPy 1.16 still uses exact with ties)
        # same condition SciPy uses, only to report which was applied (round away float noise first)
        diff = (data[b] - data[a]).round(10)
        exact = n <= 50 and (diff != 0).all()
        ties = not diff.abs().is_unique  # exact p-value is then only approximately exact
        rows.append({"comparison": f"{a} vs {b}", "W": w.statistic, "median_diff": (data[b] - data[a]).median(),
                     "p": w.pvalue, "method": "exact" if exact else "approx", "ties": ties})
    posthoc = pd.DataFrame(rows).set_index("comparison")
    posthoc["p_holm"] = multipletests(posthoc["p"], method="holm")[1]
    posthoc["p_bonferroni"] = multipletests(posthoc["p"], method="bonferroni")[1]

    if verbose:
        print(f"{n} QBPs x {k} conditions: {', '.join(conditions)}\n")
        print(f"Friedman: chi2({k - 1}) = {chi2:.3f}, p = {p:.5f}, Kendall's W = {friedman['kendall_w']:.3f}")
        print("Mean ranks: " + ", ".join(f"{c} = {r:.3f}" for c, r in friedman["mean_ranks"].items()) + "\n")
        print(f"{'Comparison':<58}{'W':>5}{'med.Δ':>8}{'p':>9}{'p(Holm)':>9}{'p(Bonf)':>9}")
        for name, r in posthoc.iterrows():
            print(f"{name:<58}{r.W:>5.1f}{r.median_diff:>+8.3f}{r.p:>9.4f}{r.p_holm:>9.4f}{r.p_bonferroni:>9.4f}")
    return friedman, posthoc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=Path(__file__).with_name("struct_sim.csv"), type=Path)
    ap.add_argument("--conditions", nargs="+", help="condition columns to compare (default: all complete columns)")
    ap.add_argument("--name-col", default="qbp_name", help="column holding the QBP names")
    ap.add_argument("--stat", default="mean", help="value of the first column selecting the rows to use")
    args = ap.parse_args()
    try:
        analyse(args.csv, args.conditions, args.name_col, args.stat)
    except ValueError as e:
        ap.error(str(e))
