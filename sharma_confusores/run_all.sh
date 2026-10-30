#!/bin/bash
# Reproduces the whole confound ablation with the CNN of Sharma and Singh (2022): the factorial design on the three
# folds, in parallel on CPU (4 threads each), then the analysis. Resumable: finished conditions are skipped.
#   export LANNA_DIR=/path/to/LANNA_Vocales_a_y_o/LANNA_Vocales_a_y_o     (optional, see README.md)
#   bash run_all.sh
# Logs in logs/; final mark SHARMA_FACTORIAL_TERMINADO in logs/run_all.log.
cd "$(dirname "$0")"
PY=${PYTHON:-$(command -v python3)}; [ -x ../../venv/bin/python3 ] && PY=${PYTHON:-../../venv/bin/python3}
export PYTHON=$PY OMP_NUM_THREADS=4
mkdir -p logs; L=logs/run_all.log
echo "$(date '+%d/%m %H:%M') start (folds 0 1 2)" >> $L
for k in 0 1 2; do $PY sharma_factorial.py $k > logs/factorial_f$k.log 2>&1 & done
wait
for k in 0 1 2; do $PY perm_controls.py $k > logs/perm_controls_f$k.log 2>&1 & done   # controls of the permutation test
wait
bash run_perm_null.sh                       # null with 1000 permutations x 5 runs (~10 h on 8 CPU processes)
$PY ci_bootstrap.py >> $L 2>&1              # 95% bootstrap CIs over children, every condition
$PY perm_pvalues.py >> $L 2>&1              # p-values and Holm correction (Table "permutation test")
$PY analysis_factorial.py >> $L 2>&1        # factorial figures and effects
$PY controls.py >> $L 2>&1                  # detectors and /a/-/o/, F0 per correction
$PY voice_checks.py >> $L 2>&1              # F1, F2, F0, HNR, jitter, shimmer per correction
echo "$(date '+%d/%m %H:%M') SHARMA_FACTORIAL_TERMINADO" >> $L
