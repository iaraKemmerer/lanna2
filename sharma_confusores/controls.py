"""
Controls of each correction of the ablation, measured on the input that Sharma's network actually receives
(sharma_lanna2.build_inputs, fold 0: the noise bank comes from its training children).

(i) Is the cue gone? Recording descriptors of each 1-s input, none of which describes the voice:
      level      peak, dBFS
      dc         mean of the input
      polarity   skewness of the nonzero part
      zeros      fraction of zero samples (duration under zero padding)
      floor      10th percentile of the 32-ms frame level, relative to the peak (background noise)
      above_4k   fraction of energy above 4 kHz
    Q1  TD vs SLI from all descriptors: logistic regression, leave-one-child-out AUC.
    Q2  within TD only, 16-kHz dictaphone vs 44.1-kHz recordings (the equipment without the diagnosis): same
        detector; chance = 95th percentile of 200 label permutations.
(ii) Is the voice still there?
    /a/ vs /o/ from 13 MFCC (mean and SD over the nonzero part), logistic regression, leave-one-child-out (both
    vowels of a child always on the same side): accuracy.
    F0: median F0 (pYIN, 75-600 Hz) of each /o/ input vs the original vowel: median |difference| in semitones.

Output: results/analysis/controls.csv, controls.txt (table "controls of each correction" of the paper)
Usage: python controls.py        (CPU, ~15 min)
"""
import warnings

import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor
from sharma_lanna2 import FLAGS, build_inputs

warnings.filterwarnings("ignore")
SR, FOLD = CFG.SAMPLE_RATE, 0
OUT = CFG.CODE_DIR / "results" / "analysis"
CONDITIONS = [()] + [(f,) for f in FLAGS] + [FLAGS]


def descriptors(x):
    nz = np.nonzero(x)[0]
    v = x[nz[0]:nz[-1] + 1] if len(nz) else x
    peak = np.abs(x).max() + 1e-12
    c = v - v.mean()
    frames = librosa.util.frame(v, frame_length=512, hop_length=256) if len(v) >= 512 else v[None, :, None]
    lev = 10 * np.log10(np.mean(frames ** 2, axis=0) + 1e-20)
    X = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2; f = np.fft.rfftfreq(len(v), 1 / SR)
    return dict(level=20 * np.log10(peak), dc=float(x.mean()), polarity=float(np.mean(c ** 3) / (np.mean(c ** 2) ** 1.5 + 1e-12)),
                zeros=float(np.mean(x == 0)), floor=float(np.percentile(lev, 10) - 20 * np.log10(peak)),
                above_4k=float(X[f >= 4000].sum() / (X[f > 20].sum() + 1e-20)))


def loo_auc(X, y):
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3)), X, y,
                          cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def mfcc_stats(x):
    nz = np.nonzero(x)[0]
    v = x[nz[0]:nz[-1] + 1] if len(nz) else x
    m = librosa.feature.mfcc(y=v.astype(np.float32), sr=SR, n_mfcc=13)
    return np.r_[m.mean(1), m.std(1)]


def f0(x):
    nz = np.nonzero(x)[0]
    v = x[nz[0]:nz[-1] + 1] if len(nz) else x
    f, _, _ = librosa.pyin(v.astype(np.float32), fmin=75, fmax=600, sr=SR, frame_length=1024)
    return float(np.nanmedian(f)) if np.any(np.isfinite(f)) else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AudioDataProcessor()
    fs = {pid: __import__("soundfile").info(f).samplerate for pid, f in proc.recordings("o").items()}
    # reference F0 of every child: the original vowel segment
    ref = {}
    for pid, f in proc.recordings("o").items():
        w, s, e = proc.load_recording(f)
        ref[pid] = f0(w[s:e] / (np.abs(w[s:e]).max() + 1e-12))
    rows = []
    for cond in CONDITIONS:
        name = "+".join(cond) or "none"
        flags = tuple(f for f in FLAGS if f in cond) if cond else ()
        if "dur" in flags and "seg" not in flags:
            flags = tuple(f for f in FLAGS if f in set(flags) | {"seg"})
        Do = build_inputs(flags, FOLD, "o"); Da = build_inputs(flags, FOLD, "a")
        ids = sorted(Do)
        F = pd.DataFrame([descriptors(Do[p][0]) for p in ids], index=ids)
        y = np.array([Do[p][1] for p in ids])
        q1 = loo_auc(F.values, y)
        single = {c: max(roc_auc_score(y, F[c]), 1 - roc_auc_score(y, F[c])) for c in F.columns}
        td = [p for p in ids if Do[p][1] == 0]
        ye = np.array([int(fs[p] == 16000) for p in td])
        q2 = loo_auc(F.loc[td].values, ye)
        rng = np.random.default_rng(0)
        q2_chance = float(np.quantile([loo_auc(F.loc[td].values, rng.permutation(ye)) for _ in range(200)], .95))
        both = [p for p in ids if p in Da]
        Xv = np.array([mfcc_stats(Do[p][0]) for p in both] + [mfcc_stats(Da[p][0]) for p in both])
        yv = np.r_[np.zeros(len(both)), np.ones(len(both))]
        gv = np.r_[both, both]
        pv = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), Xv, yv,
                               cv=GroupKFold(n_splits=len(both)), groups=gv)
        f0d = [abs(12 * np.log2(f0(Do[p][0]) / ref[p])) for p in ids if np.isfinite(ref[p])]
        r = dict(condition=name, q1_td_vs_sli=q1, q2_equipment_td=q2, q2_chance95=q2_chance,
                 vowel_a_vs_o_acc=float((pv == yv).mean()), f0_median_abs_semitones=float(np.nanmedian(f0d)),
                 **{f"single_{c}": v for c, v in single.items()})
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "controls.csv", index=False)
    with open(OUT / "controls.txt", "w") as f:
        f.write(f"Controls per correction (inputs of Sharma's network, fold 0). Q1: TD vs SLI from recording descriptors (LOO AUC); "
                f"Q2: equipment within TD (LOO AUC, chance95 from 200 permutations); /a/ vs /o/ accuracy; F0 change.\n\n"
                + R.round(3).to_string(index=False) + "\n")
    print(open(OUT / "controls.txt").read())


if __name__ == "__main__":
    main()
