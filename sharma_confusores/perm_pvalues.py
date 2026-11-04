"""p-values of the two permutation tests (perm_null.py, 1000 permutations x 5 runs each) for the replication and for
every correction, both networks, folds 0-2 and the held-out fold 3.

  full        (files *_pval.csv) training AND validation labels shuffled, so the epoch of each null network is chosen
              with shuffled labels: the full permutation test (main result).
  supervised  (files without suffix) only the training labels shuffled, the epoch chosen with the real validation
              labels: training-label permutation with supervised validation (reported as a secondary analysis).

Observed statistic: mean test AUC of the 5 runs; p = (1 + #{null >= observed}) / (1 + N). Also: the spread of the 5
runs (SD, min, max of their AUC) and the null median, 95th percentile and maximum; null_seed_spread is the mean
|AUC - 0.5| of each null training run (each seed of each permutation, not their mean), and chance_seed_spread the same
for random scores with the same numbers of TD and SLI test children (text, "networks that learn random labels"). Holm correction within each test and
condition over the six tests of folds 0-2 (two networks x three folds); fold 3 is a separate confirmatory analysis,
with Holm over its two networks. On fold 3 the rows "f3-early" exclude the children of CFG.HOLDOUT_EARLY_TEST.
Output: results/analysis/perm_pvalues.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ci_bootstrap import ci_of
from config_lanna2 import CFG

HERE = Path(__file__).resolve().parent
ALL = {"sharma": "seg+dur+gain+dc+pol+noise+lp4k", "sharmaW": "seg+gain+dc+pol+noise+lp4k"}
TESTS = {"full": "_pval", "supervised": ""}


def predictions(net, fold, spec, drop=()):
    d = HERE / "results" / "sharma" / f"{net}_f{fold}_{spec}"
    pred = pd.read_csv(next(d.glob("predictions_*.csv")))
    return pred[~pred.patient_id.isin(drop)]


def holm(R, mask):
    idx = R[mask].p.sort_values().index; m = len(idx); running = 0.0
    for i, j in enumerate(idx):
        running = max(running, min(1.0, (m - i) * R.at[j, "p"])); R.at[j, "p_holm"] = round(running, 4)


def chance_spread(n_sli, n_td, n=20000):
    rng = np.random.default_rng(0); y = np.r_[np.ones(n_sli), np.zeros(n_td)]
    return float(np.mean([abs(roc_auc_score(y, rng.random(len(y))) - 0.5) for _ in range(n)]))


def main():
    chance = {}
    rows = []
    for net in ("sharma", "sharmaW"):
        for cond in ("none", "all"):
            spec = "none" if cond == "none" else ALL[net]
            for fold, label, drop in [(0, "0", ()), (1, "1", ()), (2, "2", ()), (3, "3", ()),
                                      (3, "3-early", CFG.HOLDOUT_EARLY_TEST)]:
                if not (HERE / "results" / "sharma" / f"{net}_f{fold}_{spec}" / "summary.json").exists():
                    continue
                pred = predictions(net, fold, spec, drop)
                s = np.array([roc_auc_score(g.label, g.p_sli) for _, g in pred.groupby("run")])
                _, lo, hi = ci_of(pred, np.random.default_rng(0))       # the same bootstrap as conditions_ci.csv
                base = dict(net=net, condition=cond, fold=label, auc=round(s.mean(), 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                            seed_sd=round(s.std(ddof=1), 4), seed_min=round(s.min(), 4), seed_max=round(s.max(), 4))
                for test, suf in TESTS.items():
                    f = HERE / "results" / "permnull" / f"{net}_f{fold}_{cond}{suf}.csv"
                    if not f.exists():
                        continue
                    N = pd.read_csv(f)
                    null = N["mean_without_early_test" if drop else "mean"].dropna()
                    seeds = N[[c for c in N if c.startswith("auc_seed")]].values if not drop else None
                    k = (int(pred[pred.run == pred.run.min()].label.sum()), int((pred[pred.run == pred.run.min()].label == 0).sum()))
                    chance.setdefault(k, chance_spread(*k))
                    rows.append(dict(base, test=test, n_perm=len(null), null_median=round(null.median(), 3),
                                     null_seed_spread=round(float(np.nanmean(np.abs(seeds - 0.5))), 3) if seeds is not None else np.nan,
                                     chance_seed_spread=round(chance[k], 3),
                                     null_p95=round(null.quantile(.95), 3), null_max=round(null.max(), 3),
                                     p=round((1 + (null >= base["auc"]).sum()) / (1 + len(null)), 4)))
    R = pd.DataFrame(rows); R["p_holm"] = np.nan
    for (test, cond), g in R.groupby(["test", "condition"]):
        holm(R, R.index.isin(g.index) & R.fold.isin(["0", "1", "2"]))
        for f3 in ("3", "3-early"):
            holm(R, R.index.isin(g.index) & (R.fold == f3))
    R.to_csv(HERE / "results" / "analysis" / "perm_pvalues.csv", index=False); print(R.to_string(index=False))


if __name__ == "__main__":
    main()
