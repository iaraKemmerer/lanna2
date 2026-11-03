#!/bin/bash
# Second review: (1) the held-out fold 3 (make_holdout_fold.py), replication and every correction, both networks,
# 5 runs; (2) the full permutation test, with the validation labels shuffled too ("pval"), 1000 permutations x 5 runs,
# on folds 0-3; (3) on fold 3 also the training-label permutation with supervised validation, as on folds 0-2.
# CPU only (the GPU is busy), 6 jobs at a time with 3 threads each, low priority. Resumable. Mark REVISION2_TERMINADO.
cd "$(dirname "$0")"
PY=${PYTHON:-$(command -v python3)}; [ -x ../../venv/bin/python3 ] && PY=${PYTHON:-../../venv/bin/python3}
export PYTHON=$PY OMP_NUM_THREADS=3; mkdir -p logs; L=logs/run_all.log
echo "$(date '+%d/%m %H:%M') revision2: start" >> $L
$PY make_holdout_fold.py >> $L 2>&1
$PY - <<'P' > logs/holdout_f3.log 2>&1
from sharma_factorial import FLAGS, WINDOW, run
for spec, window in (("none", None), ("+".join(FLAGS), None), ("none", WINDOW), ("+".join(f for f in FLAGS if f != "dur"), WINDOW)):
    run(3, spec, window)
P
{ for net in sharma sharmaW; do for c in none all; do
    for k in 0 1 2 3; do echo "$k $net $c pval"; done
    echo "3 $net $c val"
  done; done; } |
  xargs -P 6 -L 1 sh -c 'nice -n 10 '"$PY"' perm_null.py $0 $1 $2 1000 $3 2>&1 | grep -v "training inputs\|RuntimeWarning" > logs/permnull_$1_f$0_$2_$3.log'
$PY ci_bootstrap.py >> $L 2>&1
$PY perm_pvalues.py >> $L 2>&1
echo "$(date '+%d/%m %H:%M') REVISION2_TERMINADO" >> $L
