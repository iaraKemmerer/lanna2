import os
from pathlib import Path

import torch


class Config:
    """
    Central class with every parameter needed to reproduce the LANNA2 paper.

    This is the configuration of the method as described in the paper (Materials and Methods): the audio
    conditions, the fixed networks, the training protocol and the statistics. The networks are fixed
    architectures (architectures/A0.json, A1.json, A2.json); no architecture search is run here.

    Organization:
    1. GENERAL: name of the project.
    2. PATHS AND FILES: where the audio is, and where inputs, results and figures are written.
    3. HARDWARE AND SEEDS: device and the seeds that make the runs reproducible.
    4. DATA AND PREPROCESSING: sampling rate, vowel segment, silence, labels and folds.
    5. DATA AUGMENTATION: pitch-shifted copies of the training recordings.
    6. TRAINING (Trainer): windows, batches, epochs and runs.
    7. OPTIMIZER (SGD): optimizer parameters.
    8. MODEL: the parts of the CNN that are common to every architecture file.
    9. AUDIO CONDITIONS: the channel corrections compared in the paper.
    10. STATISTICS: bootstrap, permutation tests and detectors.

    -----------------------------------------------------------------------
    WHERE THE AUDIO IS: the LANNA corpus (Grill and Tučková, 2016) is not distributed with this code. Set the
    environment variable LANNA_DIR to the folder that contains the TD/ and SLI/ subject folders (each with
    01SAMOHL/*.wav and the .lbl label files). Nothing else needs to be edited to reproduce the paper.
    -----------------------------------------------------------------------
    """

    # =================================================================
    # 1. GENERAL
    # =================================================================
    PROJECT = "LANNA2"

    # =================================================================
    # 2. PATHS AND FILES
    # =================================================================
    CODE_DIR = Path(__file__).resolve().parent
    LANNA_DIR = Path(os.environ.get("LANNA_DIR", CODE_DIR.parent / "data" / "LANNA"))
    WORK_DIR = CODE_DIR / "work"                    # network inputs, derived from the audio: never share them
    RESULTS_DIR = CODE_DIR / "results"              # predictions, summaries and statistics
    FIGURES_DIR = CODE_DIR / "figures"              # figures of the paper
    PARTITIONS_DIR = CODE_DIR / "partitions"        # fold_<k>.json: patient ids of train / val / test (with the .lbl corrections)
    PARTITIONS_UNCORRECTED_DIR = CODE_DIR / "partitions_uncorrected"   # the same folds without the .lbl corrections
    ARCHITECTURES_DIR = CODE_DIR / "architectures"  # <name>.json: the fixed networks

    # =================================================================
    # 3. HARDWARE AND SEEDS
    # =================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN_SEED_BASE = 7000         # run k of a condition uses torch/numpy seed 7000 + k
    AUGMENTATION_SEED = 42       # numpy seed of the pitch-shift parameters
    NUM_WORKERS = 0              # DataLoader workers (0 keeps the random generators deterministic)

    # =================================================================
    # 4. DATA AND PREPROCESSING
    # =================================================================
    SAMPLE_RATE = 16000          # every recording is resampled to 16 kHz
    CLIP_LENGTH = 16000          # network inputs are stored as 1-s arrays: the vowel, then zeros
    VOWEL_FOLDER = "01SAMOHL"    # sustained vowels of LANNA
    VOWELS = ("a", "o")          # vowels used in the paper (/o/ is the main one)
    SILENCE_MARGIN = 160         # 10 ms around the vowel (.lbl) are not counted as silence
    SILENCE_MIN_LENGTH = 512     # shorter silence parts are not used to build noise
    SLI_ID_OFFSET = 1000         # patient id = digits of the subject folder (+1000 for SLI)
    LABEL_MAP = {"TD": 0, "SLI": 1}
    # .lbl corrections (on by default): the vowel segment is the labelled segment that matches the vowel of the file
    # (the longest one if several), and a file whose label is ANOTHER vowel is excluded (in LANNA: the /o/ file of
    # TD child 1 contains an /a/). Partitions: if the excluded child was in validation, a TD child of training (fixed
    # seed) completes it; the test set is never modified. Set to False to reproduce the uncorrected runs.
    LBL_CORRECTIONS = True
    N_FOLDS = 3                  # subject-disjoint folds: 49 train, 24 validation and 24 test children each

    # =================================================================
    # 5. DATA AUGMENTATION
    # =================================================================
    AUGMENTATION_FACTOR = 5                   # pitch-shifted copies per child of the majority class (SLI)
    PITCH_STEPS = (-3, 3)                     # semitones, uniform integer
    PITCH_N_FFT = (256, 512, 1024, 2048)      # STFT size of librosa's pitch shift, uniform
    AUGMENTATION_KEEPS_SILENCE = True         # after the shift, samples after the vowel are set back to zero

    # =================================================================
    # 6. TRAINING (Trainer)
    # =================================================================
    BATCH_SIZE = 32              # 16 TD + 16 SLI windows per training batch
    MAX_EPOCHS = 150             # no early stopping: the best validation epoch is kept
    RUNS = 5                     # runs per condition (seeds 7000..7004)

    # =================================================================
    # 7. OPTIMIZER (SGD)
    # =================================================================
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-2          # the learning rate is part of each architecture file

    # =================================================================
    # 8. MODEL
    # =================================================================
    FILTERS = 10                 # channels of every Conv1d layer
    NUM_CLASSES = 2

    # =================================================================
    # 9. AUDIO CONDITIONS
    # =================================================================
    CONDITIONS = ["original", "dc", "pol", "dcpol", "noise_for_all", "donor_v3", "adaptive_v4"]
    NOISE_SNR_DB = 5.0           # noise_for_all: noise power 5 dB below the power of the vowel
    NOISE_FRAGMENT = 512         # 32-ms Hann fragments, 50% overlap-add
    V4_BAND = (80.0, 7000.0)     # adaptive_v4 (discarded): band-pass
    V4_SNR_PERCENTILE = 5        # adaptive_v4: common SNR = 5th percentile of the training SNRs
    V4_RMS = 0.1                 # adaptive_v4: vowel RMS instead of peak normalization

    # =================================================================
    # 10. STATISTICS
    # =================================================================
    BOOTSTRAP_SAMPLES = 2000          # class-stratified resamples of the test children
    TEST_LABEL_PERMUTATIONS = 10000   # cheap permutation test
    FULL_PERMUTATIONS = 20            # full permutation test: retrainings with shuffled training labels
    DETECTOR_PERMUTATIONS = 500       # chance level of the equipment question (Q2)
    DETECTOR_C = 0.3                  # inverse regularization of the detectors (logistic regression)


CFG = Config()
