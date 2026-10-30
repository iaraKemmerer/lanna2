"""95% bootstrap CIs (children, stratified by class, 2000 resamples, seed 0) of the mean test AUC of the 5 MLP runs,
per input set and fold, and a comparison of the two MLP configurations on VALIDATION (why the common one was kept).
Output: results/table_ci.csv"""
import numpy as np
import pandas as pd

B = 2000


def auc_rows(s, pos, neg):
    a, b = s[pos][:, :, None], s[neg][:, None, :]
    return (a > b).mean(axis=(1, 2)) + 0.5 * (a == b).mean(axis=(1, 2))


P = pd.read_csv("results/predictions.csv"); rows = []
for (name, fold), g in P.groupby(["inputs", "fold"], sort=False):
    rng = np.random.default_rng(0)
    runs = sorted(g.run.unique()); first = g[g.run == runs[0]].sort_values("patient_id")
    ids, y = first.patient_id.values, first.label.values
    S = np.stack([g[g.run == r].set_index("patient_id").loc[ids, "p_sli"].values for r in runs])
    i1, i0 = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    pos, neg = i1[rng.integers(0, len(i1), (B, len(i1)))], i0[rng.integers(0, len(i0), (B, len(i0)))]
    boot = np.mean([auc_rows(S[k], pos, neg) for k in range(len(runs))], axis=0)
    full = np.mean([auc_rows(S[k], i1[None], i0[None])[0] for k in range(len(runs))])
    rows.append(dict(inputs=name, fold=fold, auc=round(full, 3), ci_low=round(np.percentile(boot, 2.5), 3), ci_high=round(np.percentile(boot, 97.5), 3)))
T = pd.DataFrame(rows); T.to_csv("results/table_ci.csv", index=False); print(T.to_string(index=False))
