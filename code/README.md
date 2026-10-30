# LANNA2 — code to reproduce the paper

*Shortcuts in LANNA: duration and recording-channel confounds in the detection of specific language impairment from
sustained vowels, and a confound-controlled evaluation protocol.*

This folder contains everything needed to reproduce every number, table and figure of the paper, starting from the
LANNA recordings. It is self-contained: the two networks are fixed architectures (`architectures/A0.json`,
`architectures/A1.json`); no architecture search is run here.

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# data: python ../prepare_data.py /path/to/Lanna_speech_database_children.zip   (see ../README.md)
# or: export LANNA_DIR=/path/to/data/LANNA                                       # folder with TD/ and SLI/ subject folders
```

The LANNA corpus (Grill and Tučková, 2016) is not distributed with this code. `LANNA_DIR` must contain one folder per
child under `TD/` and `SLI/`, each with `01SAMOHL/*.wav` and the `.lbl` label files. Only the sustained vowels
`/a/` and `/o/` are used (43 TD and 54 SLI children).

## 2. Run

```bash
bash run_all.sh analyses   # CPU only, ~30 min: Parts I-II and the channel detectors
bash run_all.sh            # everything, ~3 days on one GPU (most of it is the full permutation test)
```

Or step by step:

| Step | Command | What it produces | Paper |
|---|---|---|---|
| 1 | `python audit.py` | channel measures and duration per child, classifiers that never hear the voice, /a/ vs /o/ | Figs. 1-6, E6 audit |
| 2 | `python signal_figures.py` | a recording before/after *noise for all*; SNR compression | Figs. 7-8 |
| 3 | `python praat.py` | jitter, shimmer and HNR per condition; HNR gap vs added noise | Fig. 9, E8 |
| 4 | `python detectors.py` | channel and spectrum detectors, Q1 (TD vs SLI) and Q2 (equipment within TD) | Table 3, Fig. 10 |
| 5 | `python main_lanna2.py -a A0 -f 0 -c noise_for_all` | one network, one fold, one condition, 5 runs | Table 4 (one cell) |
| 6 | `python stats.py` | AUC table, bootstrap CIs, DeLong/Holm, permutation tests, duration check | Table 4, Figs. 11-13, E3, E9 |

**Figures and language.** Every figure is drawn in English and/or Spanish from the same computation, chosen with
the environment variable `FIG_LANG`: `en` (default when running a script by hand) writes `figures/en/`, `es` writes
`figures/es/`, and `all` (default in `run_all.sh`) computes once and writes both. Only texts change between languages
(the Spanish figures use decimal commas); every number is the same. `../paper_ieee/sync_figures.sh` copies
`figures/en/` into the English paper (`paper_ieee/figures/`) and `figures/es/` into the Spanish one (`paper_es/figures/`).

```bash
FIG_LANG=all python audit.py        # figures/en/f1_... and figures/es/f1_...
```

`run_all.sh` lists every `main_lanna2.py` call used in the paper. Options:

```
-a A0|A1|A2           fixed network (A0 is used on fold 0, A1 on fold 1, A2 on fold 2)
-f 0|1|2              subject-disjoint fold (partitions/fold_k.json)
-c CONDITION          original | dc | pol | dcpol | noise_for_all | donor_v3 | adaptive_v4
-v o|a                vowel (E6: /a/)
-r N                  runs (default 5, seeds 7000..)
--region silence      E2: the network receives only the silence around the vowel
--truncate-ms 150     E3b: every vowel is cut to its first 150 ms
--permute-labels S    full permutation test: training labels shuffled among children with seed S (1 run)
--snr-db 5            noise level of noise_for_all (dB below the vowel)
```

## 3. Code structure

Every core module carries the suffix `_lanna2`, so the package can live next to other projects that use the
same module names (config, dataset, model...).

| Module | Contents |
|---|---|
| `config_lanna2.py` | `Config` / `CFG`: every parameter of the paper, in numbered sections (paths, seeds, data, augmentation, training, optimizer, model, audio conditions, statistics) |
| `dataset_lanna2.py` | `AudioDataProcessor` (reading LANNA, `.lbl` segment, network inputs, partitions), `AudioAugmenter` (pitch-shift augmentation), `WindowDataset` (50%-overlap windows), `BalancedBatchSampler` (16 TD + 16 SLI) |
| `channel_conditions_lanna2.py` | `ChannelConditions`: the audio conditions (original, dc, pol, dcpol, noise_for_all, donor_v3, adaptive_v4) and the noise banks of each fold |
| `model_lanna2.py` | `CauchyActivation`, `MLP`, `CNN` built from `architectures/<name>.json` |
| `trainer_lanna2.py` | `Trainer`: SGD, training epoch on windows, child-level evaluation, choice of the best validation epoch |
| `experiment_runner_lanna2.py` | `ExperimentRunner`: one network × fold × condition × runs; writes predictions, `summary.txt`, `historial_epocas_I0.json` and `robustez_final_I0.png` |
| `main_lanna2.py` | command line |
| `audit.py`, `signal_figures.py`, `praat.py`, `detectors.py`, `stats.py` | the analyses and figures of the paper |

**What the pipeline does.**
- **Audio.** Each file is resampled to 16 kHz (torchaudio). The vowel is the last segment of its `.lbl` file. The
  network receives only the vowel, peak-normalized and stored as a 1-s array (vowel, then zeros).
- **noise_for_all.** Remove the DC offset; add to every recording a noise 5 dB below its vowel, synthesized by
  overlap-add of 32-ms Hann fragments taken from silences of the training TD children of the fold; apply the
  canonical polarity.
- **Partitions.** `partitions/fold_k.json` holds the patient ids of each split, in the order used in the paper.
- **Training.**
  - Augmentation: pitch-shifted copies balance the classes (360 training recordings).
  - Windows and batches: 16-ms windows with 50% overlap; batches of 16 TD + 16 SLI windows.
  - Optimization: SGD (momentum 0.9, weight decay 0.01) for 150 epochs.
  - Model selection: the weights of the first epoch with the best child-level validation F1.
  - Child output: the mean softmax probability over the child's windows.
  - Runs: seeds 7000-7004.

## 4. The .lbl corrections

`CFG.LBL_CORRECTIONS = True` (default) applies two rules to the LANNA label files:
- The vowel segment is the labelled segment that matches the vowel of the file (the longest one, if several).
- A file whose label is another vowel is excluded. In LANNA this is the /o/ file of TD child 1, which contains an
  /a/.

With the corrections:
- **Fold 0:** child 1 was a test child, so the test set has 11 TD + 12 SLI.
- **Folds 1 and 2:** child 1 was a validation child; a TD child of training (fixed seed) takes its place, so training
  has 48 children (`partitions/`).

The first version of the paper's results was obtained **without** the corrections. To reproduce it, set
`CFG.LBL_CORRECTIONS = False`, which uses `partitions_uncorrected/` and the last segment of every `.lbl`.

## 5. Reproducibility

- **CPU parts are deterministic.** `audit.py`, `signal_figures.py`, `praat.py`, `detectors.py` and `stats.py` give the
  numbers of the paper exactly.
- **This code matches the paper's pipeline** (checked with `LANNA2/scripts/verificar_paquete.py`).
  - The network inputs of every condition are identical, sample by sample, to the ones used in the paper.
  - So are the initial weights for each seed and the augmented training set.
  - On CPU, the whole training (losses, validation curves and test probabilities) is bit-identical to the original
    implementation.
- **GPU training is not bit-reproducible.** cuDNN convolutions are non-deterministic, so repeated runs with the same
  seed differ.
  - On LANNA, identical runs differ by up to ~0.1 AUC. This is why the paper reports the mean of 5 runs and
    permutation tests.
  - Expect the same means within that spread, not identical digits.
- **The paper's own runs.** `stats.py` reads the predictions in `results/train/`. To recompute the statistics on the
  exact runs of the paper, put their prediction files there with the same names.

## 6. Data protection

`work/` (network inputs) and any figure that shows a waveform are derived from the children's recordings. Do not share
them. Partitions contain only patient ids.
