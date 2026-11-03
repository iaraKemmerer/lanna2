"""
95% confidence intervals of the child-level test AUC of every condition of the factorial design.

Statistic: the mean AUC of the 5 runs (as reported). Bootstrap over CHILDREN, stratified by class: each resample draws
as many TD and SLI test children as the fold has (12 + 12; 6 + 18 on the held-out fold 3) with replacement (the same children for the 5 runs), the AUC of each run is computed on
the resample and averaged; the interval is the 2.5-97.5 percentile of 2000 resamples (seed 0). The interval describes
the uncertainty due to the 24 test children of the fold; it is not the chance level (that is the permutation test).

Output: results/analysis/conditions_ci.csv (net, fold, condition, auc, ci_low, ci_high)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES, OUT = HERE / "results" / "sharma", HERE / "results" / "analysis"
B = 2000


def auc_matrix(s, pos, neg):
    """AUC for each row of the index matrices pos (B, n1) and neg (B, n0); ties count 1/2."""
    a, b = s[pos][:, :, None], s[neg][:, None, :]
    return ((a > b).mean(axis=(1, 2)) + 0.5 * (a == b).mean(axis=(1, 2)))


def ci_of(pred, rng):
    runs = sorted(pred.run.unique())
    kids = pred[pred.run == runs[0]].sort_values("patient_id")
    ids, y = kids.patient_id.values, kids.label.values
    P = np.stack([pred[pred.run == r].set_index("patient_id").loc[ids, "p_sli"].values for r in runs])  # (runs, n)
    i1, i0 = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    pos = i1[rng.integers(0, len(i1), (B, len(i1)))]; neg = i0[rng.integers(0, len(i0), (B, len(i0)))]
    boot = np.mean([auc_matrix(P[k], pos, neg) for k in range(len(runs))], axis=0)
    full = np.mean([auc_matrix(P[k], i1[None, :], i0[None, :])[0] for k in range(len(runs))])
    return full, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    rows = []
    for d in sorted(RES.iterdir()):
        if "_perm" in d.name or d.name.startswith("permnull") or not (d / "summary.json").exists():
            continue
        f = next(d.glob("predictions_*.csv"), None)
        if f is None:
            continue
        net, fold = d.name.split("_")[0], int(d.name.split("_")[1][1:])
        cond = "_".join(d.name.split("_")[2:])
        auc, lo, hi = ci_of(pd.read_csv(f), np.random.default_rng(0))
        rows.append(dict(net=net, fold=fold, condition=cond, auc=round(auc, 4), ci_low=round(lo, 4), ci_high=round(hi, 4)))
    R = pd.DataFrame(rows); R.to_csv(OUT / "conditions_ci.csv", index=False)
    print(len(R), "conditions;", R[R.condition.isin(["none", "seg+dur+gain+dc+pol+noise+lp4k", "seg+gain+dc+pol+noise+lp4k", "gain"])].to_string(index=False))


if __name__ == "__main__":
    main()
