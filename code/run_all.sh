#!/bin/bash
# Reproduces every number and figure of the paper, in order. See README.md.
#   bash run_all.sh            everything (about 2 days on one GPU, mostly the full permutation test)
#   bash run_all.sh analyses   only the CPU parts that need no training (about 30 minutes)
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
export FIG_LANG=${FIG_LANG:-all}   # figures in English (figures/en/) and Spanish (figures/es/), computed once

# ---------------------------------------------------------------- Part I and II (CPU, no training)
$PY audit.py             # Figs. 1-6, E6 audit of /a/, Tables "channel measures" and "classifiers without the voice"
$PY signal_figures.py    # Figs. 7-8 (noise for all, SNR compression)
$PY praat.py             # Fig. 9 and E8 (jitter, shimmer, HNR)
$PY detectors.py         # Table "channel detectors", Fig. 10
[ "$1" = "analyses" ] && { $PY stats.py; exit 0; }

# ---------------------------------------------------------------- Part III: networks (GPU)
# Each fold uses its own fixed network (A0 for fold 0, A1 for fold 1, A2 for fold 2), trained and tested on that fold only.
for pair in "A0 0" "A1 1" "A2 2"; do
  set -- $pair; A=$1; K=$2
  for FORM in original dc pol dcpol noise_for_all donor_v3 adaptive_v4; do     # Table "child-level AUC", Figs. 11-12
    $PY main_lanna2.py -a $A -f $K -c $FORM
  done
  for FORM in original noise_for_all; do                                       # controls
    $PY main_lanna2.py -a $A -f $K -c $FORM --region silence                 # E2: the network only hears silence
    $PY main_lanna2.py -a $A -f $K -c $FORM --truncate-ms 150               # E3b: every vowel cut to 150 ms
    $PY main_lanna2.py -a $A -f $K -c $FORM -v a                                # E6: vowel /a/
  done
  for S in $(seq 1 20); do                                                     # full permutation test, Fig. 13
    $PY main_lanna2.py -a $A -f $K -c noise_for_all --permute-labels $S
  done
done

# ---------------------------------------------------------------- statistics and remaining figures (CPU)
$PY stats.py             # CIs, DeLong/Holm, cheap and full permutation tests, E3a, Figs. 11-13, E3a, E9
