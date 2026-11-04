"""
Held-out fold (fold 3): its test set is the 24 children that are in the test set of NO fold (6 TD, 18 SLI).

The corrections (in particular how the noise is equalized and its 5-dB level, the common polarity and the 4-kHz
low-pass) were chosen in exploratory work in which networks and channel detectors were evaluated on the test children
of folds 0-2. The children of this test set were never test children there (they were only used for training and
validation), so the evaluation on fold 3 is independent of how the corrections were chosen. Caveat: four of them
(TD 19 and 37, SLI 1002 and 1027) were test children of an earlier, non-disjoint fold 0 used at the very start of the
exploratory work (14/10/2026, first noise equalization, later discarded); results are also reported without them
(HOLDOUT_EARLY_TEST in config_lanna2.py).

Validation: 12 TD + 12 SLI drawn (fixed seed) from the other 72 children; training: the remaining 48. The order of
the training ids is shuffled with the same seed (it matters for the augmentation). /a/ uses the same children (plus
child 1, whose /o/ is excluded, in training).
    python make_holdout_fold.py        -> partitions/fold_3.json (with the .lbl corrections only)
"""
import json

import numpy as np

from config_lanna2 import CFG

SEED = 2026


def main():
    folds = [json.load(open(CFG.PARTITIONS_DIR / f"fold_{k}.json")) for k in range(CFG.N_FOLDS)]
    out = {"fold": CFG.HOLDOUT_FOLD,
           "note": "held-out fold: the test children are the 24 children that are in the test set of no other fold "
                   "(not test children of folds 0-2, on which the corrections were chosen; four were test children of an early exploratory fold); validation 12 TD + 12 SLI drawn "
                   f"from the other children (seed {SEED}), training the rest. See make_holdout_fold.py."}
    rng = np.random.default_rng(SEED)
    test_o = val_o = None
    for v in sorted(CFG.VOWELS, key=lambda v: v != "o"):   # /o/ first: it defines the test and validation children
        kids = sorted({p for f in folds for s in ("train", "val", "test") for p in f[v][s]})
        tested = {p for f in folds for p in f["o"]["test"]}
        if test_o is None:
            test_o = sorted(p for p in kids if p not in tested)
            rest = [p for p in kids if p in tested]
            td = [p for p in rest if p < CFG.SLI_ID_OFFSET]; sli = [p for p in rest if p >= CFG.SLI_ID_OFFSET]
            val_o = sorted(rng.choice(td, 12, replace=False).tolist() + rng.choice(sli, 12, replace=False).tolist())
        test = [p for p in test_o if p in kids]
        val = [p for p in val_o if p in kids]
        train = [p for p in kids if p not in test and p not in val]
        train = [int(p) for p in rng.permutation(train)]
        out[v] = {"train": train, "val": val, "test": test}
        n_td = lambda xs: sum(p < CFG.SLI_ID_OFFSET for p in xs)
        print(f"/{v}/: train {len(train)} ({n_td(train)} TD), val {len(val)} ({n_td(val)} TD), test {len(test)} ({n_td(test)} TD)")
    json.dump(out, open(CFG.PARTITIONS_DIR / f"fold_{CFG.HOLDOUT_FOLD}.json", "w"), indent=1)
    print("test:", out["o"]["test"])


if __name__ == "__main__":
    main()
