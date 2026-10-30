#!/bin/bash
# Null distribution with 1000 permutations x 5 runs (perm_null.py) for the replication and for all the corrections,
# both networks, three folds: 12 jobs, 8 at a time on CPU (3 threads each). Resumable. Mark PERM_NULL_TERMINADO.
cd "$(dirname "$0")"
PY=${PYTHON:-$(command -v python3)}; [ -x ../../venv/bin/python3 ] && PY=${PYTHON:-../../venv/bin/python3}
export OMP_NUM_THREADS=3; mkdir -p logs
for net in sharma sharmaW; do for c in none all; do for k in 0 1 2; do echo "$k $net $c"; done; done; done |
  xargs -P 8 -L 1 sh -c 'nice -n 5 '"$PY"' perm_null.py $0 $1 $2 1000 2>&1 | grep -v "training inputs" > logs/permnull_$1_f$0_$2.log'
echo "$(date '+%d/%m %H:%M') PERM_NULL_TERMINADO" >> logs/run_all.log
