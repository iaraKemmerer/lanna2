# LANNA2 — shortcuts in the LANNA corpus

Code for the paper:

> I. Kemmerer, M. F. Gerard, L. D. Vignolo. *Shortcuts in LANNA: duration and recording-channel confounds in the
> detection of specific language impairment from sustained vowels.* Submitted to Computer Methods and Programs in
> Biomedicine.

In the LANNA corpus the diagnosis (typically developing children, TD, vs. children with specific language
impairment, SLI) is confounded with how each group was recorded. The confounds are:
- recording equipment, site and session;
- recording level, background noise, DC offset, waveform polarity and high-frequency energy;
- vowel duration.

This repository contains everything needed to reproduce the paper from the public corpus:
- the audit of the corpus;
- models that receive only the confounds;
- the reimplementation of the CNN of Sharma and Singh (2022), with a factorial ablation of every confound correction;
- the channel detectors and voice checks;
- the proposed protocol.

## Data

The LANNA corpus is **not** included. It is public under CC BY 4.0:

- Grill P., Tučková J. *Speech Databases of Typical Children and Children with SLI.* PLoS ONE 11(3): e0150365 (2016).
- Download `Lanna_speech_database_children.zip` from https://doi.org/10.6084/m9.figshare.2360626

Then run:

```bash
python prepare_data.py /path/to/Lanna_speech_database_children.zip
```

This copies the sustained vowels (folder `01SAMOHL`) of the 97 children used in the paper into `data/LANNA/`:
- 43 TD and 54 SLI, the same children as Sharma and Singh (2022);
- the layout is `data/LANNA/TD/H<k>/01SAMOHL/` and `data/LANNA/SLI/P<k>/01SAMOHL/`;
- `data_mapping.csv` gives the archive folder of each child; the files are byte-identical.

**Child ids.** A child's id is the number of its folder, plus 1000 for the SLI children: `TD/H1` is child 1, `SLI/P1` is child 1001. This keeps TD and SLI children with the same folder number apart (`Config.SLI_ID_OFFSET`). The partitions (`partitions/fold_<k>.json`) use these ids.

No audio, and nothing computed from the audio, is stored in this repository. The `.gitignore` excludes `data/`, `work/`, `results/` and `figures/`.

## Label corrections

`lbl_corrections.csv` lists the `.lbl` files of /a/ and /o/ that need a decision:
- **One file is excluded:** the /o/ file of TD child H1 (archive folder `Healthy/H26`) contains an /a/.
- **Four files are documented and kept:** three files have several labels, and the code uses the longest segment labelled with the vowel. One file has a single label that is not the vowel. All four end up with the same segment as the plain "last segment" rule.

The rule is implemented in `AudioDataProcessor.lbl_segment` (`code/dataset_lanna2.py`). It can be switched off with `Config.LBL_CORRECTIONS = False`.

## Contents

| Folder | What it reproduces | Run |
|---|---|---|
| `code/` | audit of the corpus, signal figures, Praat measures, channel detectors, fixed CNNs, statistics | `bash code/run_all.sh` |
| `mlp_confusores/` | models that receive only confound measures, no audio | `bash mlp_confusores/run_all.sh` |
| `sharma_confusores/` | CNN of Sharma and Singh (2022), factorial ablation of the corrections, permutation tests, controls of each correction | `bash sharma_confusores/run_all.sh` |

Each folder is self-contained: it has its own README, configuration and requirements, and writes `results/` and `figures/<es|en>/` inside itself.

Every condition uses:
- three subject-disjoint folds (`partitions/fold_<k>.json`, patient ids only);
- five runs per condition;
- the child-level AUC, reported per fold.

Folds are never averaged into a single number.

## Which script reproduces each table and figure of the paper

| Paper | Contents | Script (output) |
|---|---|---|
| Fig. 1, 2, 3, 4, 5, 6, 8 · Tables 2, 3 | sampling rate, durations, channel measures, /a/ vs /o/, equipment within TD, classifiers without voice, zero padding | `code/audit.py` (`code/results/audit/`, `code/figures/<es|en>/`) |
| Fig. 9 | SNR compression by noise for all | `code/signal_figures.py` |
| Table 4 · Fig. 7 | MLP with only confound measures, with 95% CIs | `mlp_confusores/mlp_confusores.py`, `ci_mlp.py` (`mlp_confusores/results/`) |
| Table 5 | what was reproduced, reconstructed and modified in the replication | described in the paper; implemented in `sharma_confusores/sharma_lanna2.py`, `main_sharma.py` |
| Table 6 | replication of the CNN of Sharma and Singh (accuracy, sensitivity, specificity, AUC) | `sharma_confusores/main_sharma.py` via `sharma_factorial.py` (`results/sharma/sharma_f<k>_none/summary.json`) |
| Table 7 | the corrections | `sharma_confusores/sharma_lanna2.py` |
| Table 8 | detectors and /a/-/o/, F0 per correction | `sharma_confusores/controls.py` (`results/analysis/controls.csv`) |
| Table 9 | F1, F2, F0, HNR, jitter, shimmer per correction | `sharma_confusores/voice_checks.py` (`results/analysis/voice_checks.csv`) |
| Fig. 10, 11 | factorial design: each correction alone, all but one, all; main effects | `sharma_confusores/sharma_factorial.py`, `analysis_factorial.py` |
| Table 10 | full permutation test (1000 permutations x 5 runs), 95% CIs, Holm, quantiles of the null | `perm_null.py` (`run_perm_null.sh`), `ci_bootstrap.py`, `perm_pvalues.py` (`results/analysis/perm_pvalues.csv`) |
| Text, "networks trained with random labels" | shuffled validation labels / last epoch | `sharma_confusores/perm_controls.py` |

Every number in these tables can be regenerated from the public archive of the corpus with `prepare_data.py` followed by the `run_all.sh` of each folder. Runs use fixed seeds (7000-7004). Results are bit-identical on the same software versions (`code/requirements.txt`, CPU).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r code/requirements.txt
```

CPU is enough for `mlp_confusores` and `sharma_confusores`, which take about 1.5 h with 3 parallel processes. The fixed CNNs of `code/` are faster on a GPU.

## License

- **Code:** MIT (see `LICENSE`).
- **LANNA corpus:** CC BY 4.0, by its authors. Please cite both, as listed in `CITATION.cff`.
