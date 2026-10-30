# Models that never hear the voice: an MLP with only confound measures

Self-contained folder for the table and figure "an MLP that receives only confounds" of the LANNA2 paper.

```bash
pip install -r requirements.txt
# data: see ../README.md (python ../prepare_data.py ...); or export LANNA_DIR=/path/to/data/LANNA
bash run_all.sh                                                    # CPU, a few minutes
```

## Inputs

`mlp_confusores.py` measures, for each child and the vowel /o/, the following. The `.lbl` corrections are applied.

| measure | what it is |
|---|---|
| `duration` | duration of the vowel (`.lbl` segment) |
| `level` | peak of the whole file, in dBFS. This is what a network without normalization sees. |

The next five are computed **as the network sees them**, after peak-normalizing the vowel:

| measure | what it is |
|---|---|
| `dc` | DC offset of the file divided by the vowel peak |
| `polarity` | skewness of the vowel |
| `noise` | background noise relative to the vowel peak, in dB |
| `snr` | vowel level minus noise level, in dB |
| `above_4k` | fraction of the vowel energy above 4 kHz |

None of these measures describes the voice; they describe the recording. The script then classifies TD vs SLI with each measure alone and with several combinations.

## Models

- **MLP.** The same MLP in every fold: one hidden layer of 8 units, Linear → BatchNorm → Cauchy → Dropout(0.1). Training follows the networks, except learning rate 0.01 and 300 epochs (one example per child gives only 3 batches per epoch; the learning rates of the networks were tuned for hundreds of windows per child):
  - SGD with momentum 0.9 and weight decay 0.01, except on the Cauchy parameters;
  - balanced batches of 8 TD + 8 SLI children;
  - the epoch with the best validation F1;
  - 5 runs.
- **Reference:** logistic regression on the same inputs.

Both models use the same subject-disjoint folds as the networks (`partitions/`: 48 train / 24 validation / 24 test children).

## Output

- `results/table.csv`, `results/summary.txt`: test AUC and F1 per input set, model and fold, plus the chance level for 24 children.
- `results/results.csv`: one row per run.
- `results/measures_o.csv`: the per-child measures. They derive from LANNA, so do not share them.
- `figures/es/mlp_confounds.png` and `figures/en/mlp_confounds.png`: the figure in Spanish and in English.
