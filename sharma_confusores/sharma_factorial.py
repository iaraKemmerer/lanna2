"""
Factorial design of the confound corrections with the CNN of Sharma and Singh (2022), on one fold.

  1-s network (SharmaCNN): every subset of seg, dur, gain, dc, pol, noise, lp4k (dur implies seg: 96 distinct).
  Window network (SharmaWindowCNN, 128-ms windows): every subset of seg, gain, dc, pol, noise, lp4k (64).
  Full permutation test: 20 label shuffles for 'none' and for every correction, in both networks.
Each condition: 5 runs (main_sharma.py). A condition that already has its summary.json is skipped (resumable).

    python sharma_factorial.py 0      # fold 0
"""
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
PY = os.environ.get("PYTHON", sys.executable)
FLAGS = ("seg", "dur", "gain", "dc", "pol", "noise", "lp4k")
WINDOW = 2048


def subsets(flags):
    out = []
    for k in range(len(flags) + 1):
        for c in itertools.combinations(flags, k):
            if "dur" in c and "seg" not in c:          # same input as with seg: skipped
                continue
            out.append("+".join(c) or "none")
    return out


def run(fold, spec, window=None, perm=None, extra=(), suffix=""):
    name = "_".join(p for p in ["sharmaW" if window else "sharma", f"f{fold}", spec,
                                f"perm{perm}" if perm is not None else "", suffix] if p)
    if (CODE / "results" / "sharma" / name / "summary.json").exists():
        return
    cmd = [PY, "main_sharma.py", "-f", str(fold), "-c", spec, "--device", "cpu"]
    if window:
        cmd += ["--window", str(window)]
    if perm is not None:
        cmd += ["--permute-labels", str(perm)]
    cmd += list(extra)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if l.startswith("mean")]
    print(f"{time.strftime('%d/%m %H:%M')} f{fold} {name}: {last[-1] if last else 'ERROR ' + r.stderr[-300:]} ({time.time() - t0:.0f} s)", flush=True)


def main():
    fold = int(sys.argv[1])
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    all_1s, all_w = subsets(FLAGS), subsets(tuple(f for f in FLAGS if f != "dur"))
    for spec in all_1s:
        run(fold, spec)
    for spec in all_w:
        run(fold, spec, WINDOW)
    for perm in range(1, 21):
        for spec, window in (("none", None), (all_1s[-1], None), ("none", WINDOW), (all_w[-1], WINDOW)):
            run(fold, spec, window, perm)
    print(f"{time.strftime('%d/%m %H:%M')} fold {fold} terminado", flush=True)


if __name__ == "__main__":
    main()
