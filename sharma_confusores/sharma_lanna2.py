"""
The 1-D CNN of Sharma and Singh (2022) and the inputs of the confound ablation.

Sharma Y., Singh B.K. One-dimensional convolutional neural network and hybrid deep-learning paradigm for
classification of specific language impaired children using their speech. Comput. Methods Programs Biomed. 213
(2022) 106487. Only the CNN is reimplemented (not the CNN-LSTM hybrid).

Their preprocessing (p. 3): the whole file resampled to 16 kHz, "no additional alteration", cut or zero-padded to
16000 samples. Points the paper leaves open, fixed here (and stated in our paper):
  - zero padding goes AFTER the audio; a longer file keeps its first 16000 samples;
  - Conv -> BatchNorm -> ReLU, with a BatchNorm on the input (text, p. 4);
  - no padding in the convolutions (it is what makes the sizes of their Fig. 2 come out: 199, 65, 20, 5, 1).

The network needs at least 15040 samples (0.94 s) to produce an output, so it cannot take our short windows. Duration
is therefore removed by repeating the vowel until it fills the 16000 samples ("dur").

Ablation: an input is the Sharma input plus a set of corrections ("flags"), each removing one confound:
    seg     only the vowel segment of the .lbl file (removes the silence, where the background noise is seen alone)
    dur     the vowel repeated to fill 1 s, no zeros (removes duration; implies seg)
    gain    peak normalization of what the network receives (removes the recording level)
    dc      DC offset removed
    pol     canonical polarity (ChannelConditions.canonical_polarity)
    noise   noise from the training TD silences added 5 dB below the vowel (masks background noise and SNR)
    lp4k    8th-order Butterworth low-pass at 4 kHz, zero phase (removes the energy above 4 kHz)
"none" is the replication of Sharma. Order of operations on the whole 16-kHz file: dc, noise, pol, lp4k; then the
segment, the repetition, the normalization and the cut/padding to 16000 samples.
"""
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt

from channel_conditions_lanna2 import ChannelConditions
from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor

FLAGS = ("seg", "dur", "gain", "dc", "pol", "noise", "lp4k")


# =============================================================
# MODEL
# =============================================================
class SharmaCNN(nn.Module):
    """
        input (B, 1, 16000) -> BatchNorm
        CL1: Conv1d(1 -> 10, k=160, stride=80) -> BN -> ReLU                       (B, 10, 199)
        CL2-CL4: Conv1d(10 -> 10, k=5) -> BN -> ReLU -> MaxPool(3, stride 3)        (B, 10, 65 / 20 / 5)
        CL5: Conv1d(10 -> 10, k=5) -> BN -> ReLU                                   (B, 10, 1)
        Linear(10 -> 2): logits (softmax in the loss)
    Glorot initialization with zero biases, as in the paper. 3672 weights + 102 BatchNorm parameters.
    """

    def __init__(self, filters: int = 10):
        super().__init__()
        layers = [nn.BatchNorm1d(1), nn.Conv1d(1, filters, 160, 80), nn.BatchNorm1d(filters), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Conv1d(filters, filters, 5), nn.BatchNorm1d(filters), nn.ReLU(), nn.MaxPool1d(3, 3)]
        layers += [nn.Conv1d(filters, filters, 5), nn.BatchNorm1d(filters), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(filters, CFG.NUM_CLASSES)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class SharmaWindowCNN(nn.Module):
    """
    MODIFIED Sharma network for short windows (not their network as published): the same five convolutional layers,
    filters, kernels, strides, BatchNorm and ReLU, but CL2-CL5 pad 2 samples on each side, the max-pooling keeps the
    last incomplete block (ceil_mode) and a global average pooling over time comes before Linear(10 -> 2).
    With a 2048-sample window (128 ms): (B, 10, 24) -> 8 -> 3 -> 1 -> 1. Same number of parameters as SharmaCNN.
    """

    def __init__(self, filters: int = 10):
        super().__init__()
        layers = [nn.BatchNorm1d(1), nn.Conv1d(1, filters, 160, 80), nn.BatchNorm1d(filters), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Conv1d(filters, filters, 5, padding=2), nn.BatchNorm1d(filters), nn.ReLU(),
                       nn.MaxPool1d(3, 3, ceil_mode=True)]
        layers += [nn.Conv1d(filters, filters, 5, padding=2), nn.BatchNorm1d(filters), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(filters, CFG.NUM_CLASSES)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.features(x).mean(dim=2))


# =============================================================
# INPUTS
# =============================================================
def parse_flags(spec: str) -> Tuple[str, ...]:
    """'none' or flags joined by '+', e.g. 'seg+dur+gain'. dur implies seg. Returned in the canonical order."""
    flags = set() if spec in ("", "none") else set(spec.split("+"))
    unknown = flags - set(FLAGS)
    if unknown:
        raise ValueError(f"unknown flags {sorted(unknown)}; choose from {FLAGS}")
    if "dur" in flags:
        flags.add("seg")
    return tuple(f for f in FLAGS if f in flags)


def lowpass_4k(w):
    sos = butter(8, 4000.0, btype="low", fs=CFG.SAMPLE_RATE, output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, w))


def fit_length(x: np.ndarray, n: int = CFG.CLIP_LENGTH) -> np.ndarray:
    """First n samples, or zeros after the audio up to n."""
    return x[:n] if len(x) >= n else np.concatenate([x, np.zeros(n - len(x))])


def build_inputs(flags: Tuple[str, ...], fold: int, vowel: str = "o", snr_db: float = None,
                 use_cache: bool = True) -> Dict[int, Tuple[np.ndarray, int]]:
    """{patient_id: (float32 (16000,), label)} for every child, for one set of corrections and one fold."""
    snr_db = CFG.NOISE_SNR_DB if snr_db is None else snr_db
    tag = "sharma_" + ("+".join(flags) or "none") + f"{'' if snr_db == CFG.NOISE_SNR_DB else f'_snr{snr_db:g}'}_{vowel}_f{fold}"
    path = CFG.WORK_DIR / f"{tag}{'_lblfix' if CFG.LBL_CORRECTIONS else ''}.pkl"
    proc = AudioDataProcessor()
    if use_cache and path.exists():
        return pickle.load(open(path, "rb"))
    cc = ChannelConditions(proc, set(proc.partition(fold, "o")["train"]))
    out = {}
    for pid, f in sorted(proc.recordings(vowel).items()):
        w, s, e = proc.load_recording(f)
        rng = np.random.default_rng(1000 * pid + ord(vowel))          # same noise as noise_for_all
        if "dc" in flags:
            w = w - w.mean()
        if "noise" in flags:
            a = np.sqrt(cc.power(w[s:e]))
            w = w + a * 10 ** (-snr_db / 20) * cc.synth_noise(cc.bank(), len(w), rng)
            if "dc" in flags:
                w = w - w.mean()
        if "pol" in flags:
            w = cc.canonical_polarity(w, s, e)
        if "lp4k" in flags:
            w = lowpass_4k(w)
        x = w[s:e] if "seg" in flags else w
        if "dur" in flags:
            x = np.tile(x, int(np.ceil(CFG.CLIP_LENGTH / len(x))))
        if "gain" in flags:
            peak = np.abs(x).max()
            x = x / peak if peak > 0 else x
        out[pid] = (fit_length(x).astype(np.float32), CFG.LABEL_MAP[proc.group_of(f)])
    CFG.WORK_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


def circular_shift_copies(x: np.ndarray, copies: int, rng) -> list:
    """Sharma's augmentation: circular shifts in steps of 10 ms within +-300 ms (a nonzero step for each copy)."""
    steps = rng.choice([k for k in range(-30, 31) if k != 0], size=copies, replace=copies > 60)
    return [np.roll(x, int(k) * CFG.SAMPLE_RATE // 100) for k in steps]
