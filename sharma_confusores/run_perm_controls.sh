#!/bin/bash
# Controls of the permutation test (perm_controls.py), three folds in parallel on CPU. Mark PERM_CONTROLS_TERMINADO.
cd "$(dirname "$0")"
PY=${PYTHON:-$(command -v python3)}; [ -x ../../venv/bin/python3 ] && PY=${PYTHON:-../../venv/bin/python3}
export PYTHON=$PY OMP_NUM_THREADS=4; mkdir -p logs
for k in 0 1 2; do $PY perm_controls.py $k > logs/perm_controls_f$k.log 2>&1 & done
wait
echo "$(date '+%d/%m %H:%M') PERM_CONTROLS_TERMINADO" >> logs/run_all.log
