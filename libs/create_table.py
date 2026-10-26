import pandas as pd


def create_table(df, incl_optim_prmt, merge_std=False, decimals=3, std_decimals=2):
    """Clean a result table: readable row labels, optional optimised-prompt row.
    The result is inverted (qbp names as rows, prompt variants as columns) and ends with an
    "overall" row: the average per prompt condition (column) over all qbp rows.

    df : columns are qbp names under a "mean"/"std" level, optionally under a
         leading metric level, e.g. "struct_sim" (dropped); index is either
         (incl_model_guide, incl_knowl, optim_prmt) or already the row labels.
    incl_optim_prmt : keep the "+ optimised prompt" row if True, else drop it.
    merge_std : if True, combine mean and std into one "mean (std)" string per cell
         (trailing zeros dropped); otherwise keep separate mean and std columns.
    decimals : decimals the mean values are rounded to.
    std_decimals : decimals the "std" values are rounded to.
    """
    labels = {
        (0, 0, 0): "baseline",
        (1, 0, 0): "+ modelling guidelines",
        (1, 1, 0): "+ knowledge constraints",
        (1, 1, 1): "+ optimised prompt",
    }

    table = df.copy()
    if table.columns.nlevels == 3:                       # drop e.g. "struct_sim"
        table.columns = table.columns.droplevel(0)
    if isinstance(table.index, pd.MultiIndex):           # flags -> row labels
        table.index = [labels[idx] for idx in table.index]
    table.index.name = None

    if not incl_optim_prmt and "+ optimised prompt" in table.index:
        table = table.drop("+ optimised prompt")

    # inverted: qbp names as rows, prompt variants as columns, plus an "overall" row
    # (per prompt condition, the average over the qbp rows; for std, the average of the stds)
    mean, std = table["mean"].T, table["std"].T
    mean.loc["overall"], std.loc["overall"] = mean.mean(axis=0), std.mean(axis=0)

    if merge_std:
        def num(x, d):                                   # at most `d` decimals, no trailing zeros
            return f"{x:.{d}f}".rstrip("0").rstrip(".")

        def fmt(m, s):
            if pd.isna(m):
                return ""
            return num(m, decimals) if pd.isna(s) else f"{num(m, decimals)} ({num(s, std_decimals)})"

        return pd.DataFrame({c: [fmt(m, s) for m, s in zip(mean[c], std[c])]
                             for c in mean.columns}, index=mean.index)

    return pd.concat({"mean": mean.round(decimals), "std": std.round(std_decimals)})
