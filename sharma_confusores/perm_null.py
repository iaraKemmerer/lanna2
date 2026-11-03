"""
Null distribution of the full permutation test, with the SAME statistic as the observed result.

Observed statistic: mean test AUC (per child) of the 5 runs (seeds 7000-7004) of one network, fold and condition.
Null statistic: the same mean of 5 runs, after shuffling the labels of the TRAINING children among them (every copy of
a child keeps one label). Everything else is repeated exactly as in the observed result: the circular-shift copies,
the balanced batches, the training from scratch and the choice of the epoch with the validation children (whose labels
are NOT shuffled), and the evaluation on the test children with their real labels.
p = (1 + #{null >= observed}) / (1 + N).

With "pval" the validation labels are shuffled too (among the validation children, with their own seed), so the
epoch of each null network is chosen with shuffled labels and no step of the null procedure sees a real label except
the final evaluation on the test children: this is the full permutation test of the paper. Without it, only the
training labels are shuffled and the epoch is chosen with the real validation labels (training-label permutation with
supervised validation).

On the held-out fold (CFG.HOLDOUT_FOLD) each row also has the AUC without the children of CFG.HOLDOUT_EARLY_TEST.

Writes results/permnull/<net>_f<fold>_<condition>[_pval].csv (permutation, auc of each seed, mean); resumable.
    python perm_null.py <fold> <sharma|sharmaW> <none|all> [N=1000] [pval]
"""
import csv
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from config_lanna2 import CFG
from main_sharma import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SharmaRunner
from sharma_lanna2 import FLAGS
from trainer_lanna2 import Trainer

WINDOW = 2048


def main():
    fold, net, cond = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    pval = len(sys.argv) > 5 and sys.argv[5] == "pval"
    holdout = fold == CFG.HOLDOUT_FOLD
    window = WINDOW if net == "sharmaW" else None
    spec = "none" if cond == "none" else "+".join(f for f in FLAGS if not (window and f == "dur"))
    CFG.MAX_EPOCHS, CFG.WEIGHT_DECAY = EPOCHS, WEIGHT_DECAY
    out = CFG.CODE_DIR / "results" / "permnull"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{net}_f{fold}_{cond}{'_pval' if pval else ''}.csv"
    done = set()
    if path.exists():
        done = {int(r["perm"]) for r in csv.DictReader(open(path))}
    f = open(path, "a", newline=""); w = csv.writer(f)
    if not done:
        w.writerow(["perm"] + [f"auc_seed{CFG.RUN_SEED_BASE + k}" for k in range(CFG.RUNS)] + ["mean"]
                   + (["mean_without_early_test"] if holdout else []))
    t0 = time.time()
    for perm in range(1, n + 1):
        if perm in done:
            continue
        r = SharmaRunner(fold, spec, permute_labels=perm, runs=CFG.RUNS, window=window, device="cpu", name="permnull_tmp",
                         permute_val=pval)
        trainer = Trainer("cpu", r.dl_train, r.dl_val)
        dl_test = r.dl_tests[r.condition]
        aucs, aucs_sub = [], []
        for k in range(CFG.RUNS):
            torch.manual_seed(CFG.RUN_SEED_BASE + k); np.random.seed(CFG.RUN_SEED_BASE + k)
            trainer.set_learning_rate(LEARNING_RATE)
            trainer.evaluate_model(r.build_model())
            recs, p, y = trainer.predict_children(dl_test)
            ok = np.all(np.isfinite(p))
            aucs.append(float(roc_auc_score(y, p[:, 1])) if ok else float("nan"))
            if holdout:
                keep = np.array([r.test_ids[i] not in CFG.HOLDOUT_EARLY_TEST for i in recs])
                aucs_sub.append(float(roc_auc_score(y[keep], p[keep, 1])) if ok else float("nan"))
        w.writerow([perm] + [f"{a:.4f}" for a in aucs] + [f"{np.nanmean(aucs):.4f}"]
                   + ([f"{np.nanmean(aucs_sub):.4f}"] if holdout else [])); f.flush()
        if perm % 25 == 0:
            print(f"{net} f{fold} {cond}: {perm}/{n} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"{net} f{fold} {cond}: terminado", flush=True)


if __name__ == "__main__":
    main()
