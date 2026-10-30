import glob
import json
import os
import pickle
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

from config_lanna2 import CFG


def voiced_length(x) -> int:
    """Samples up to the last non-zero value: the vowel ends there, then comes the zero padding to CLIP_LENGTH."""
    nz = np.nonzero(np.asarray(x))[0]
    return int(nz[-1]) + 1 if len(nz) else 0


# =============================================================
# LOADING AND PREPROCESSING
# =============================================================
class AudioDataProcessor:
    """
    Reads the LANNA recordings and builds the network inputs: one 1-s float32 array per child (the processed vowel,
    then zeros), for a given audio condition, fold and vowel.

    Steps for each recording:
      1. Load and resample to 16 kHz (torchaudio, default parameters).
      2. Vowel segment = the LAST label of the .lbl file (start and duration in samples of the original rate,
         rescaled with round()). If it falls outside the file, the whole file is used.
      3. Audio condition on the whole file (ChannelConditions).
      4. Network input: the vowel, peak-normalized (RMS = 0.1 for adaptive_v4), truncated or zero-padded to 1 s.

    Two variants used by the controls of the paper:
      region="silence"   the network receives the parts of the file OUTSIDE the vowel instead of the vowel
                         (E2: a network that never hears the voice). Files without silence get 512 zeros.
      truncate_ms=150    every vowel is cut to its first 150 ms (E3b: duration is removed).

    Inputs are cached in CFG.WORK_DIR (building a condition takes about a minute).
    """

    # =============================================================
    # FILES AND PATIENTS
    # =============================================================
    @staticmethod
    def list_files() -> List[str]:
        """Every sustained-vowel file (a/e/i/o/u), sorted by full path. The order matters: noise banks follow it."""
        files = sorted(glob.glob(f"{CFG.LANNA_DIR}/*/*/{CFG.VOWEL_FOLDER}/*.[wW][aA][vV]"))
        return [f for f in files if AudioDataProcessor.vowel_of(f) in "aeiou"]

    @staticmethod
    def vowel_of(path: str) -> str:
        return os.path.basename(path)[-5].lower()          # e.g. 4PK1O.wav -> "o"

    @staticmethod
    def group_of(path: str) -> str:
        return "SLI" if os.sep + "SLI" + os.sep in path else "TD"

    @staticmethod
    def patient_id(path: str) -> int:
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pid = int("".join(c for c in folder if c.isdigit()))
        return pid + (CFG.SLI_ID_OFFSET if AudioDataProcessor.group_of(path) == "SLI" else 0)

    def recordings(self, vowel: str) -> Dict[int, str]:
        """{patient_id: path} for one vowel: one file per child (without the excluded files, see lbl_segment)."""
        return {self.patient_id(f): f for f in self.list_files()
                if self.vowel_of(f) == vowel and self.lbl_segment(f)[0] != "exclude"}

    @staticmethod
    def partition(fold: int, vowel: str = "o") -> Dict[str, List[int]]:
        """{'train', 'val', 'test'}: patient ids in the exact order used for training (it matters for augmentation)."""
        folder = CFG.PARTITIONS_DIR if CFG.LBL_CORRECTIONS else CFG.PARTITIONS_UNCORRECTED_DIR
        return json.load(open(folder / f"fold_{fold}.json"))[vowel]

    # =============================================================
    # ONE RECORDING
    # =============================================================
    @staticmethod
    def lbl_segment(path: str):
        """
        (status, (label, start, duration)) of the .lbl file, in samples of the original rate.
        Without corrections: always the LAST segment. With CFG.LBL_CORRECTIONS: the segment whose label is the vowel of
        the file (the longest one if several); if no segment has that vowel but one has ANOTHER vowel, status
        'exclude' (the file does not contain the vowel it is named after); otherwise the last segment.
        """
        lbl = os.path.splitext(path)[0] + ".lbl"
        if not os.path.isfile(lbl):
            return "ok", None
        tok = open(lbl).read().split()
        if len(tok) < 3:
            return "ok", None
        if not CFG.LBL_CORRECTIONS:                       # original rule: the last two numbers of the file
            return "ok", ("last", int(tok[-2]), int(tok[-1]))
        segments = [(tok[i], int(tok[i + 1]), int(tok[i + 2])) for i in range(0, len(tok) - 2, 3)]
        vowel = AudioDataProcessor.vowel_of(path)
        same = [s for s in segments if s[0].lower() == vowel]
        if same:
            return "ok", max(same, key=lambda s: s[2])
        if any(s[0].lower() in "aeiou" for s in segments):
            return "exclude", None
        return "ok", segments[-1]

    @staticmethod
    def load_recording(path: str) -> Tuple[np.ndarray, int, int]:
        """Whole file at 16 kHz (float64, not normalized) and the vowel segment (start, end) in samples."""
        w, sr0 = torchaudio.load(path)
        w = torchaudio.functional.resample(w, sr0, CFG.SAMPLE_RATE)[0].double().numpy()
        start, end = 0, len(w)
        status, seg = AudioDataProcessor.lbl_segment(path)
        if seg is not None:
            i0 = int(round(seg[1] * CFG.SAMPLE_RATE / sr0))
            i1 = int(round((seg[1] + seg[2]) * CFG.SAMPLE_RATE / sr0))
            if 0 <= i0 < i1 <= len(w):
                start, end = i0, i1
        return w, start, end

    @staticmethod
    def silence_parts(w, start, end) -> List[np.ndarray]:
        """The parts of the file outside the vowel (CFG.SILENCE_MARGIN), only those of CFG.SILENCE_MIN_LENGTH or more."""
        parts = (w[:max(start - CFG.SILENCE_MARGIN, 0)], w[min(end + CFG.SILENCE_MARGIN, len(w)):])
        return [p for p in parts if len(p) >= CFG.SILENCE_MIN_LENGTH]

    @staticmethod
    def network_input(w, start, end, norm: str = "peak") -> np.ndarray:
        """The vowel segment, normalized (peak = 1, or RMS = CFG.V4_RMS), truncated or zero-padded to 1 s."""
        x = torch.tensor(w[start:end], dtype=torch.float32).view(1, -1)
        if norm == "rms":
            r = x.pow(2).mean().sqrt()
            if r > 0:
                x = x * (CFG.V4_RMS / r)
        else:
            peak = x.abs().max()
            if peak > 0:
                x = x / peak
        if x.size(1) > CFG.CLIP_LENGTH:
            x = x[:, :CFG.CLIP_LENGTH]
        elif x.size(1) < CFG.CLIP_LENGTH:
            x = torch.cat([x, torch.zeros(1, CFG.CLIP_LENGTH - x.size(1))], 1)
        return x.view(-1).numpy()

    # =============================================================
    # MAIN FUNCTION: NETWORK INPUTS OF A CONDITION
    # =============================================================
    def build_inputs(self, condition: str, fold: int, vowel: str = "o", region: str = "vowel",
                     truncate_ms: float = None, snr_db: float = None, use_cache: bool = True) -> Dict[int, Tuple[np.ndarray, int]]:
        """{patient_id: (input float32 (16000,), label)} for every child of the corpus."""
        from channel_conditions_lanna2 import ChannelConditions
        snr_db = CFG.NOISE_SNR_DB if snr_db is None else snr_db
        tag = (f"{condition}{'' if snr_db == CFG.NOISE_SNR_DB else f'_snr{snr_db:g}'}_{region}"
               f"{'' if not truncate_ms else f'_cut{truncate_ms:g}'}_{vowel}_f{fold}{'_lblfix' if CFG.LBL_CORRECTIONS else ''}")
        path = CFG.WORK_DIR / f"{tag}.pkl"
        if use_cache and path.exists():
            keep = self.recordings(vowel)                  # files excluded by the .lbl corrections are dropped
            return {p: v for p, v in pickle.load(open(path, "rb")).items() if p in keep}
        conditions = ChannelConditions(self, set(self.partition(fold, "o")["train"]))
        out = {}
        for pid, f in sorted(self.recordings(vowel).items()):
            w, s, e = self.load_recording(f)
            x, norm = conditions.apply(condition, w, s, e, pid, vowel, self.group_of(f), snr_db)
            if region == "silence":
                parts = self.silence_parts(x, s, e)
                x = np.concatenate(parts) if parts else np.zeros(0)
                if len(x) < CFG.SILENCE_MIN_LENGTH:
                    x = np.concatenate([x, np.zeros(CFG.SILENCE_MIN_LENGTH - len(x))])
                s, e = 0, len(x)
            if truncate_ms:
                e = min(e, s + int(round(truncate_ms * CFG.SAMPLE_RATE / 1000)))
            out[pid] = (self.network_input(x, s, e, norm), CFG.LABEL_MAP[self.group_of(f)])
        CFG.WORK_DIR.mkdir(parents=True, exist_ok=True)
        pickle.dump(out, open(path, "wb"))
        return out


# =============================================================
# DATA AUGMENTATION
# =============================================================
class AudioAugmenter:
    """
    Pitch-shifted copies of the training recordings, balancing the classes (applied to the training set only,
    once, before windowing; all runs of a condition share the same augmented set).
    """

    def __init__(self, sr: int = CFG.SAMPLE_RATE, random_state: int = CFG.AUGMENTATION_SEED):
        self.sr = sr
        np.random.seed(random_state)          # reproducible pitch-shift parameters

    def _pitch_shift_transform(self, x: np.ndarray) -> np.ndarray:
        """n_steps ~ U{-3..3} semitones and n_fft ~ U{256, 512, 1024, 2048}; librosa keeps the length."""
        n_steps = np.random.randint(CFG.PITCH_STEPS[0], CFG.PITCH_STEPS[1] + 1)
        n_fft = np.random.choice(list(CFG.PITCH_N_FFT))
        return librosa.effects.pitch_shift(x.astype(np.float32), sr=self.sr, n_steps=n_steps, n_fft=n_fft).astype(np.float32)

    def apply_pitch_augmentation(self, waves: List[np.ndarray], labels: List[int], ids: List[int],
                                 factor: int = CFG.AUGMENTATION_FACTOR):
        """
        ADAPTIVE augmentation: each child of the majority class gets `factor` copies; the minority class gets as
        many copies as needed to reach the same total, spread as evenly as possible (the first children in
        training order get one extra copy). Copies are inserted right after their original, in training order.
        With CFG.AUGMENTATION_KEEPS_SILENCE the samples after the original vowel are set back to zero (the shift
        spills sound over the zero padding), so copies keep the duration and the number of windows of the original.
        """
        labels = np.asarray(labels)
        classes, counts = np.unique(labels, return_counts=True)
        if counts[0] > counts[1]:
            major, minor, n_major, n_minor = classes[0], classes[1], counts[0], counts[1]
        else:
            major, minor, n_major, n_minor = classes[1], classes[0], counts[1], counts[0]
        needed = n_major * (factor + 1) - n_minor
        base, extra = needed // n_minor, needed % n_minor
        out_w, out_y, out_id = [], [], []
        for x, y, pid in zip(waves, labels, ids):
            out_w.append(x); out_y.append(int(y)); out_id.append(pid)
            if y == major:
                n_copies = factor
            else:
                n_copies = base + (1 if extra > 0 else 0)
                extra -= 1 if extra > 0 else 0
            n_voice = voiced_length(x)
            for _ in range(n_copies):
                z = self._pitch_shift_transform(x)
                if CFG.AUGMENTATION_KEEPS_SILENCE and 0 < n_voice < len(z):
                    z = z.copy(); z[n_voice:] = 0
                out_w.append(z); out_y.append(int(y)); out_id.append(pid)
        return out_w, out_y, out_id


# =============================================================
# WINDOWS AND BALANCED BATCHES
# =============================================================
class WindowDataset(Dataset):
    """
    Windows of W samples with hop W/2 (50% overlap), inside the vowel only (up to the last non-zero sample: the
    zero padding is never seen). Each window keeps the index of its recording, so the output can be averaged per
    child. Item: (window float32 (W,), label int64, recording index).
    """

    def __init__(self, waves: List[np.ndarray], labels: List[int], W: int):
        X, Y, I = [], [], []
        for i, (x, y) in enumerate(zip(waves, labels)):
            n = max(voiced_length(x), 1)
            v = x[:n]
            if n < W:                                   # never happens in LANNA (shortest vowel: 2060 samples)
                v = np.pad(v, (0, W - n), mode="reflect")
                n = W
            for s in range(0, n - W + 1, W // 2):
                X.append(v[s:s + W]); Y.append(y); I.append(i)
        self.x = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.int64)
        self.rec = torch.tensor(I, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.rec[i]

    def counts(self) -> Dict[str, int]:
        return {"TD": int((self.y == 0).sum()), "SLI": int((self.y == 1).sum())}


class BalancedBatchSampler(Sampler):
    """
    Training batches with half of the windows of each class. Each epoch: shuffle the TD windows and the SLI windows
    (numpy global generator), then make min(#TD, #SLI) // 16 batches of 16 + 16 and shuffle each batch. A different
    random subset of the majority class is seen every epoch.
    """

    def __init__(self, y, batch_size: int = CFG.BATCH_SIZE):
        y = np.asarray(y)
        self.half = batch_size // 2
        self.classes = [np.nonzero(y == c)[0] for c in (0, 1)]
        self.n = min(len(c) for c in self.classes) // self.half

    def __iter__(self):
        perm = [np.random.permutation(c) for c in self.classes]
        for b in range(self.n):
            batch = np.concatenate([p[b * self.half:(b + 1) * self.half] for p in perm])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.n
