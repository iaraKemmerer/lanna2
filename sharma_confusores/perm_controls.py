"""
Controls of the full permutation test, on one fold: why do networks trained with shuffled labels separate the
classes in the replication? For 'none' (Sharma's input), 'gain' (peak normalization only) and every correction, in
both networks, 20 label shuffles each with three variants:
  val    epoch chosen with the real validation labels (as in sharma_factorial.py)
  pval   validation labels shuffled too
  last   no epoch selection: the last epoch
    python perm_controls.py 0
"""
import sys

from sharma_factorial import FLAGS, WINDOW, run

ALL_1S = "+".join(FLAGS)
ALL_W = "+".join(f for f in FLAGS if f != "dur")

if __name__ == "__main__":
    fold = int(sys.argv[1])
    for perm in range(1, 21):
        for spec, window in (("none", None), ("gain", None), (ALL_1S, None), ("none", WINDOW), ("gain", WINDOW), (ALL_W, WINDOW)):
            for variant in ("val", "pval", "last"):
                extra = {"pval": ["--permute-val"], "last": ["--epoch-choice", "last"]}.get(variant, [])
                run(fold, spec, window, perm, extra=extra, suffix={"pval": "pval", "last": "last"}.get(variant, ""))
    print(f"fold {fold} terminado", flush=True)
