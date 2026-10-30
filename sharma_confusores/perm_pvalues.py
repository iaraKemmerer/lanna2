"""p-values of the full permutation test with 1000 permutations x 5 runs (perm_null.py): observed mean AUC of the 5 runs
vs the null of the same statistic. p = (1 + #{null >= observed}) / (1 + N). Also the 95th percentile of the null (the
AUC that chance reaches 5% of the time in that fold and condition). Output: results/analysis/perm_pvalues.csv"""
import pandas as pd

ALL = {"sharma": "seg+dur+gain+dc+pol+noise+lp4k", "sharmaW": "seg+gain+dc+pol+noise+lp4k"}
C = pd.read_csv("results/analysis/conditions_ci.csv"); rows = []
for net in ("sharma", "sharmaW"):
    for cond in ("none", "all"):
        for k in range(3):
            null = pd.read_csv(f"results/permnull/{net}_f{k}_{cond}.csv")["mean"].dropna()
            c = C[(C.net == net) & (C.fold == k) & (C.condition == ("none" if cond == "none" else ALL[net]))].iloc[0]
            rows.append(dict(net=net, condition=cond, fold=k, auc=c.auc, ci_low=c.ci_low, ci_high=c.ci_high, n_perm=len(null),
                             null_median=round(null.median(), 3), null_p95=round(null.quantile(.95), 3),
                             null_max=round(null.max(), 3), p=round((1 + (null >= c.auc).sum()) / (1 + len(null)), 4)))
R = pd.DataFrame(rows)
# Holm correction within each condition (6 tests: 2 networks x 3 folds)
for cond, g in R.groupby("condition"):
    order = g.p.sort_values().index; m = len(order); running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * R.at[idx, "p"])); R.at[idx, "p_holm"] = round(running, 4)
R.to_csv("results/analysis/perm_pvalues.csv", index=False); print(R.to_string(index=False))
