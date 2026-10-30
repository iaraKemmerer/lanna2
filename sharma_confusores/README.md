# Confound ablation with the CNN of Sharma and Singh (2022)

Self-contained folder: everything needed to reproduce this part of the LANNA2 paper is here.

```bash
pip install -r requirements.txt
# data: see ../README.md (python ../prepare_data.py ...); or export LANNA_DIR=/path/to/data/LANNA
bash run_all.sh                                                    # ~1.5 h on CPU, the three folds in parallel
```

## What it does

**Network.** `sharma_lanna2.py`:
- `SharmaCNN` is the 1-D CNN of Sharma and Singh, *Comput. Methods Programs Biomed.* 213 (2022) 106487, Fig. 2. It has a BatchNorm on the input, then five convolutional layers with 10 filters each:
  - layer 1: kernel 160, stride 80;
  - layers 2–4: kernel 5, max-pooling 3/3;
  - layer 5: kernel 5.
  Each convolution is followed by BatchNorm and ReLU, and a Linear(10→2) gives the output. It has 3,774 parameters and needs at least 15,040 input samples.
- `SharmaWindowCNN` is a **modified** version for 128-ms windows, so it is no longer their network as published. It keeps the same layers, but pads layers 2–5, uses ceil-mode pooling, and adds a global average over time before the Linear layer.

Points the paper leaves open, fixed here:
- zero padding goes after the audio, and a longer file keeps its first 16,000 samples;
- the order is Conv → BN → ReLU;
- the convolutions have no padding.
Only the CNN is reimplemented; the CNN-LSTM hybrid is not.

**Inputs and corrections.** The replication (`none`) is Sharma's input: the whole file at 16 kHz, not normalized, cut or zero-padded to 16,000 samples. Each correction removes one confound:

| flag | removes | how |
|---|---|---|
| `seg` | silence (background noise seen alone) | only the vowel segment of the `.lbl` file |
| `dur` | duration | the vowel repeated to fill 1 s, no zeros (implies `seg`) |
| `gain` | recording level | peak normalization |
| `dc` | DC offset | mean removed |
| `pol` | polarity | canonical polarity (vowel skewness ≥ 0) |
| `noise` | background noise / SNR | noise from the training TD silences, 5 dB below the vowel |
| `lp4k` | energy above 4 kHz | 8th-order Butterworth low-pass, zero phase |

**Training (`main_sharma.py`).** These settings follow Sharma:
- SGD with momentum 0.9, learning rate 0.01, L2 1e-4;
- 30 epochs;
- Glorot initialization;
- circular-shift augmentation in 10-ms steps within ±300 ms, with 10 copies per child.

These settings change so the results compare with the rest of the paper:
- the subject-disjoint folds in `partitions/` (48 train / 24 validation / 24 test children);
- the epoch is chosen on validation;
- balanced batches;
- 5 runs per condition;
- the result is the child-level AUC.

With `--window 2048` (128 ms), the window network receives 50%-overlap windows and averages them per child; it uses no `dur` and no copies.

**Design (`sharma_factorial.py`).** It runs every combination of corrections ("factorial design"):
- for the 1-s network, all subsets of the 7 flags (96 distinct, since `dur` implies `seg`);
- for the window network, all subsets of the 6 flags without `dur` (64);
- a full permutation test, retraining with the training labels shuffled 20 times, for `none` and for all corrections, in both networks.

The design therefore contains each correction alone, every pair, "all but one" and all together.

## Output

`results/sharma/<name>/` follows the same format as the rest of the paper:
- `summary.txt` / `summary.json`;
- `predictions_<condition>.csv`;
- `historial_epocas_I0.json`;
- `robustez_final_I0.png`.

The audio-derived inputs go to `work/`. Neither folder may be shared, because both derive from LANNA.

## Files

| file | contents |
|---|---|
| `sharma_lanna2.py` | `SharmaCNN`, `SharmaWindowCNN`, input corrections, circular-shift augmentation |
| `main_sharma.py` | one fold × one set of corrections × 5 runs |
| `sharma_factorial.py` | the factorial design on one fold |
| `run_all.sh` | the three folds in parallel, the permutation controls, the 1000-permutation null, the analysis and the controls of each correction |
| `perm_null.py`, `run_perm_null.sh` | full permutation test with 1000 permutations x 5 runs (same statistic as the observed result) |
| `ci_bootstrap.py` | 95% bootstrap CIs over test children for every condition |
| `perm_pvalues.py` | p-values, Holm correction and 95th percentile of the null |
| `voice_checks.py` | F1, F2, F0, HNR, jitter and shimmer before/after each correction |
| `intelligibility_checks.py` | intelligibility per correction: five-vowel recognition (trained on original audio) and SQUIM |
| `perm_controls.py` | permutation test with shuffled validation labels or the last epoch |
| `analysis_factorial.py` | tables and figures of the factorial design |
| `controls.py` | detectors (TD vs SLI, equipment within TD) and voice checks (/a/ vs /o/, F0) per correction |
| `config_lanna2.py`, `dataset_lanna2.py`, `channel_conditions_lanna2.py`, `trainer_lanna2.py`, `experiment_runner_lanna2.py`, `model_lanna2.py` | copied from `../code` (reading LANNA, `.lbl` segment, noise bank, trainer); `ExperimentRunner` gains `build_model()` |
| `partitions/` | the three folds (with the `.lbl` corrections) |
