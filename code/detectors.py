"""Part III of the paper: does the processed audio still reveal the recording channel? (no network is trained)

Two detectors, both logistic regressions (standardized inputs, C = 0.3) evaluated leaving one child out, applied to
the vowel exactly as the network receives it (fold 0, vowel /o/):
  channel detector  : 6 channel measures averaged over 5 windows of 112 ms (DC, energy > 4 kHz, energy 7-8 kHz,
                      energy < 80 Hz, inter-harmonic noise floor, spectral flatness) + skewness of the vowel
  spectrum detector : vowel energy in 24 log-spaced bands from 100 Hz to 7 kHz (sees voice and channel)
Two questions:
  Q1  TD or SLI?                                   (43 TD, 54 SLI)
  Q2  within TD children: 16-kHz dictaphone or 44.1-kHz MiniDisc?   (15 / 28)  <- pure equipment
For Q2 the chance level comes from 500 permutations of the equipment label.

Outputs: results/detectors/detectors.json and figures/<lang>/f10_detectors.png (<lang> = en, es; FIG_LANG=all draws both).
Usage: python detectors.py            (about 15 minutes, CPU)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor
from plotting_lanna2 import FORM_NAME, SLI_C, TD_C, T, each_lang, save, style

PROCESSOR = AudioDataProcessor()
recordings, build = PROCESSOR.recordings, PROCESSOR.build_inputs
SR, RESULTS_DIR = CFG.SAMPLE_RATE, CFG.RESULTS_DIR

FORMS = ["original", "dcpol", "noise_for_all", "donor_v3", "adaptive_v4"]
WIN, K = 1792, 5                                       # 5 windows of 112 ms
_f, _hann = np.fft.rfftfreq(WIN, 1 / SR), np.hanning(WIN)
BANDS = np.geomspace(100, 7000, 25)


def channel_measures(v):
    """The 6 channel measures of a vowel, averaged over 5 evenly spaced 112-ms windows."""
    x = v / (np.abs(v).max() + 1e-12)
    if len(x) < WIN:
        x = np.resize(np.concatenate([x, x[::-1]]), WIN)
    M = []
    for a in np.linspace(0, len(x) - WIN, K).round().astype(int):
        u = x[a:a + WIN]; dc = u.mean()
        P = np.abs(np.fft.rfft((u - dc) * _hann)) ** 2 + 1e-12
        tot = P[_f > 20].sum()
        frac = lambda lo, hi: np.log10(max(P[(_f >= lo) & (_f < hi)].sum() / tot, 1e-9))
        pdb = 10 * np.log10(P); q = P[_f > 80]
        M.append([dc * 100, frac(4000, 8001), frac(7000, 8001), frac(0, 80),
                  (np.quantile(pdb[_f > 80], 0.10) - pdb.max()) / 10, 10 * np.log10(np.exp(np.log(q).mean()) / q.mean()) / 10])
    return np.mean(M, 0)


def features(x):
    n = int(np.nonzero(x)[0].max()) + 1 if np.any(x) else len(x)
    v = x[:n].astype(np.float64); c = v - v.mean()
    skew = np.mean(c ** 3) / (np.mean(c ** 2) ** 1.5 + 1e-12)
    P = np.abs(np.fft.rfft(c * np.hanning(n))) ** 2; f = np.fft.rfftfreq(n, 1 / SR)
    lt = np.array([np.log10(P[(f >= a) & (f < b)].sum() + 1e-12) for a, b in zip(BANDS[:-1], BANDS[1:])])
    return np.r_[channel_measures(v), skew], lt - lt.mean()


def auc_loo(X, y):
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=CFG.DETECTOR_C)), X, y,
                          cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


fs = {pid: sf.info(f).samplerate for pid, f in recordings("o").items()}
res, rng = {}, np.random.default_rng(0)                 # one generator for all conditions, in this order
for form in FORMS:
    data = build(form, 0, "o")
    pids = [p for p in sorted(data) if p < 1000] + [p for p in sorted(data) if p >= 1000]        # TD then SLI
    F = [features(data[p][0]) for p in pids]
    C, L = np.array([f[0] for f in F]), np.array([f[1] for f in F])
    y = np.array([data[p][1] for p in pids]); td = y == 0
    eq = np.array([fs[p] == 16000 for p in pids])[td].astype(int)
    r = {}
    for det, X in (("channel", C), ("spectrum", L)):
        r[f"q1_{det}"] = auc_loo(X, y)
        obs = auc_loo(X[td], eq)
        null = np.array([auc_loo(X[td], rng.permutation(eq)) for _ in range(CFG.DETECTOR_PERMUTATIONS)])
        r[f"q2_{det}"], r[f"q2_{det}_p"], r[f"q2_{det}_chance95"] = obs, float((1 + np.sum(null >= obs)) / (len(null) + 1)), float(np.quantile(null, .95))
    res[form] = r
    print(form, {k: round(v, 3) for k, v in r.items()}, flush=True)
out = RESULTS_DIR / "detectors"; out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "detectors.json", "w"), indent=1)

style()
for _ in each_lang():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, (q, title) in zip(ax, [("q1", "Q1: TD or SLI?  (43 TD, 54 SLI)"), ("q2", "Q2: within TD, 16 kHz or 44.1 kHz?  (15 / 28)")]):
        y = np.arange(len(FORMS))[::-1]
        if q == "q2":
            az = np.mean([res[f]["q2_channel_chance95"] for f in FORMS]); a.axvspan(1 - az, az, color="#d9d9d9", alpha=.6, lw=0, label=T("chance (95%)"))
        for det, mk, cl in (("channel", "o", SLI_C), ("spectrum", "s", TD_C)):
            a.scatter([res[f][f"{q}_{det}"] for f in FORMS], y, marker=mk, s=70, color=cl, zorder=3,
                      label=T("channel detector (6 measures + skewness)") if det == "channel" else T("spectrum detector (24 vowel bands)"))
        a.axvline(.5, c="k", lw=.8, ls=":"); a.set_xlim(.25, 1); a.set_yticks(y); a.set_yticklabels([T(FORM_NAME[f]) for f in FORMS])
        a.set_xlabel(T("detector AUC (0.5 = cannot tell; 1 = perfect)")); a.set_title(T(title), fontsize=10)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, -.08), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(); save(fig, "f10_detectors.png"); plt.close(fig)
