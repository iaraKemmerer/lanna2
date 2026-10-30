#!/bin/bash
# Reproduces the table and figure of the models that receive only confound measures (no audio).
#   export LANNA_DIR=/path/to/LANNA_Vocales_a_y_o/LANNA_Vocales_a_y_o     (optional, see README.md)
#   bash run_all.sh            -> results/ and figures/es, figures/en; final mark MLP_CONFUSORES_TERMINADO in logs/run_all.log
cd "$(dirname "$0")"
PY=${PYTHON:-$(command -v python3)}; [ -x ../../venv/bin/python3 ] && PY=${PYTHON:-../../venv/bin/python3}
export OMP_NUM_THREADS=2; mkdir -p logs
$PY mlp_confusores.py > logs/run_all.log 2>&1
$PY ci_mlp.py >> logs/run_all.log 2>&1                                   # 95% bootstrap CIs (Table of the paper)
MLP_MODE=networks $PY mlp_confusores.py >> logs/run_all.log 2>&1       # first configuration, kept to justify the change on validation
echo "$(date '+%d/%m %H:%M') MLP_CONFUSORES_TERMINADO" >> logs/run_all.log
