"""
Null distribution of the full permutation test, with the SAME statistic as the observed result.

Observed statistic: mean test AUC (per child) of the 5 runs (seeds 7000-7004) of one network, fold and condition.
Null statistic: the same mean of 5 runs, after shuffling the labels of the TRAINING children among them (every copy of
a child keeps one label). Everything else is repeated exactly as in the observed result: the circular-shift copies,
the balanced batches, the training from scratch and the choice of the epoch with the validation children (whose labels
are NOT shuffled), and the evaluation on the test children with their real labels.
p = (1 + #{null >= observed}) / (1 + N).

Writes results/permnull/<net>_f<fold>_<condition>.csv (permutation, auc of each seed, mean); resumable.
    python perm_null.py <fold> <sharma|sharmaW> <none|all> [N=1000]
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
    window = WINDOW if net == "sharmaW" else None
    spec = "none" if cond == "none" else "+".join(f for f in FLAGS if not (window and f == "dur"))
    CFG.MAX_EPOCHS, CFG.WEIGHT_DECAY = EPOCHS, WEIGHT_DECAY
    out = CFG.CODE_DIR / "results" / "permnull"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{net}_f{fold}_{cond}.csv"
    done = set()
    if path.exists():
        done = {int(r["perm"]) for r in csv.DictReader(open(path))}
    f = open(path, "a", newline=""); w = csv.writer(f)
    if not done:
        w.writerow(["perm"] + [f"auc_seed{CFG.RUN_SEED_BASE + k}" for k in range(CFG.RUNS)] + ["mean"])
    t0 = time.time()
    for perm in range(1, n + 1):
        if perm in done:
            continue
        r = SharmaRunner(fold, spec, permute_labels=perm, runs=CFG.RUNS, window=window, device="cpu", name="permnull_tmp")
        trainer = Trainer("cpu", r.dl_train, r.dl_val)
        dl_test = r.dl_tests[r.condition]
        aucs = []
        for k in range(CFG.RUNS):
            torch.manual_seed(CFG.RUN_SEED_BASE + k); np.random.seed(CFG.RUN_SEED_BASE + k)
            trainer.set_learning_rate(LEARNING_RATE)
            trainer.evaluate_model(r.build_model())
            _, p, y = trainer.predict_children(dl_test)
            aucs.append(float(roc_auc_score(y, p[:, 1])) if np.all(np.isfinite(p)) else float("nan"))
        w.writerow([perm] + [f"{a:.4f}" for a in aucs] + [f"{np.nanmean(aucs):.4f}"]); f.flush()
        if perm % 25 == 0:
            print(f"{net} f{fold} {cond}: {perm}/{n} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"{net} f{fold} {cond}: terminado", flush=True)


if __name__ == "__main__":
    main()
