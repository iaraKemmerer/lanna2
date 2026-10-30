"""Part I of the paper: the audit of LANNA (no network is trained).

For every child and vowel (/o/ and /a/) it measures, on the 16-kHz recording:
  sampling rate of the original file; duration of the file, of the vowel (.lbl) and of the silence;
  background noise = RMS level (dBFS) of the silence (outside the vowel, 10-ms margin); vowel level (dBFS);
  SNR = vowel level - noise level; DC offset = mean of the whole file; peak (dBFS) of the whole file;
  skewness of the vowel (its sign is the "polarity"); fraction of vowel energy above 4 kHz;
  fraction of silence energy above 4 kHz and in 45-55 Hz; noise relative to the peak = noise - peak.
Then the AUC of each measure alone, and classifiers that never hear the voice (logistic regression,
leave-one-child-out).

Outputs: results/audit/measures_<vowel>.csv, results/audit/audit.json and the figures
figures/<lang>/f1_sampling_rate.png ... f6_equipment_td.png and figures/<lang>/e6_vowels.png
(<lang> = en, es; FIG_LANG=all draws both from one computation).
Usage: python audit.py          (about 2 minutes, CPU)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor
from plotting_lanna2 import DICT, GRIS, SLI_C, TD_C, N, T, box_points, each_lang, save, style

PROCESSOR = AudioDataProcessor()
recordings, load, group_of = PROCESSOR.recordings, PROCESSOR.load_recording, PROCESSOR.group_of
SR, SILENCE_MARGIN, RESULTS_DIR = CFG.SAMPLE_RATE, CFG.SILENCE_MARGIN, CFG.RESULTS_DIR

OUT = RESULTS_DIR / "audit"; OUT.mkdir(parents=True, exist_ok=True)


def dbfs(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def band_fraction(x, lo, hi=None):
    """Fraction of the energy (Hann-windowed spectrum, above 20 Hz) in [lo, hi)."""
    if len(x) < 256:
        return float("nan")
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(X[(f >= lo) & (f < (hi if hi else SR / 2))].sum() / (X[f > 20].sum() + 1e-20))


def measures(vowel):
    rows = []
    for pid, f in sorted(recordings(vowel).items()):
        w, s, e = load(f)
        voice = w[s:e]
        sil = np.concatenate([w[:max(s - SILENCE_MARGIN, 0)], w[min(e + SILENCE_MARGIN, len(w)):]])
        v = voice - voice.mean()
        r = dict(patient_id=pid, group=group_of(f), y=int(group_of(f) == "SLI"), sampling_rate=sf.info(f).samplerate,
                 file_s=len(w) / SR, vowel_s=len(voice) / SR, silence_s=len(sil) / SR,
                 voice_dbfs=dbfs(voice), noise_dbfs=dbfs(sil) if len(sil) > 160 else np.nan,
                 peak_dbfs=float(20 * np.log10(np.abs(w).max() + 1e-12)), dc=float(w.mean()),
                 skewness=float(np.mean(v ** 3) / (np.mean(v ** 2) ** 1.5 + 1e-12)),
                 voice_above_4k=band_fraction(voice, 4000), silence_above_4k=band_fraction(sil, 4000),
                 silence_50hz=band_fraction(sil, 45, 55))
        r["snr_db"] = r["voice_dbfs"] - r["noise_dbfs"]
        r["noise_rel_peak"] = r["noise_dbfs"] - r["peak_dbfs"]
        r["zeros_1s"] = 1 - min(r["file_s"], 1.0)                 # zero padding to 1 s (Sharma and Singh, 2022)
        rows.append(r)
    return pd.DataFrame(rows)


def auc(d, col):
    ok = d[col].notna()
    return float(roc_auc_score(d.y[ok], d[col][ok]))


def auc_loo(d, cols):
    X = d[cols].values; ok = np.all(np.isfinite(X), 1)
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X[ok], d.y.values[ok],
                          cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return float(roc_auc_score(d.y.values[ok], p)), int(ok.sum())


SINGLE = ["sampling_rate", "file_s", "vowel_s", "silence_s", "noise_dbfs", "snr_db", "dc", "skewness", "voice_above_4k", "noise_rel_peak", "zeros_1s"]
NO_VOICE = {"vowel duration": ["vowel_s"], "file duration": ["file_s"], "% zeros (padding to 1 s)": ["zeros_1s"],
            "sampling rate only": ["sampling_rate"], "silence only (noise, 4 kHz, 50 Hz, DC)": ["noise_dbfs", "silence_above_4k", "silence_50hz", "dc"],
            "silence at network scale (rel. to peak)": ["noise_rel_peak", "silence_above_4k", "silence_50hz"], "polarity only": ["skewness"]}

res, D = {}, {}
for vowel in ("o", "a"):
    d = measures(vowel); d.to_csv(OUT / f"measures_{vowel}.csv", index=False); D[vowel] = d
    r = {"n": {"TD": int((d.y == 0).sum()), "SLI": int((d.y == 1).sum())},
         "auc": {c: auc(d, c) for c in SINGLE},
         "median": {c: {"TD": float(d[c][d.y == 0].median()), "SLI": float(d[c][d.y == 1].median())} for c in SINGLE},
         "no_voice": {}}
    for name, cols in NO_VOICE.items():                       # one measure: direct AUC; several: logistic LOO
        r["no_voice"][name] = max(auc(d, cols[0]), 1 - auc(d, cols[0])) if len(cols) == 1 else auc_loo(d, cols)[0]
    td = d[d.y == 0]
    r["equipment_within_td"] = {c: float(roc_auc_score((td.sampling_rate == 16000).astype(int)[td[c].notna()], td[c][td[c].notna()]))
                                for c in ("noise_dbfs", "snr_db", "voice_above_4k")}
    res[vowel] = r
rng = np.random.default_rng(0)
d = D["o"]
res["chance95_97_children"] = float(np.quantile([roc_auc_score(rng.permutation(d.y.values), d.vowel_s.values) for _ in range(5000)], .95))
json.dump(res, open(OUT / "audit.json", "w"), indent=1)

# ------------------------------------------------------------------ figures (vowel /o/), once per language (FIG_LANG)
style()
for _ in each_lang():
    fig, a = plt.subplots(figsize=(6.2, 2.8))
    for i, g in enumerate(("TD", "SLI")):
        n16 = int(((d.group == g) & (d.sampling_rate == 16000)).sum()); n44 = int(((d.group == g) & (d.sampling_rate == 44100)).sum())
        a.barh(i, n16, color=DICT, edgecolor="white", label=T("16 kHz (dictaphone)") if i == 0 else None)
        a.barh(i, n44, left=n16, color=TD_C if g == "TD" else SLI_C, edgecolor="white", label=T("44.1 kHz") if i == 0 else None)
        a.text(n16 + n44 + 1, i, T("{n16} at 16 kHz · {n44} at 44.1 kHz").format(n16=n16, n44=n44), va="center", fontsize=9)
    a.set_yticks([0, 1]); a.set_yticklabels(["TD", "SLI"]); a.set_xlabel(T("children (vowel /o/)")); a.set_xlim(0, 85)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.3), ncol=2, fontsize=8.5); a.set_title(T("Original sampling rate of each recording"), fontsize=10)
    save(fig, "f1_sampling_rate.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (c, t) in zip(ax, [("file_s", "whole file"), ("vowel_s", "vowel (.lbl segment)"), ("silence_s", "silence (file − vowel)")]):
        box_points(a, [d[c][d.y == 0].dropna().values, d[c][d.y == 1].dropna().values], ["TD", "SLI"], [TD_C, SLI_C])
        a.set_ylabel(T("seconds")); a.set_title(T("{t}\nAUC alone = {auc}").format(t=T(t), auc=N(auc(d, c))), fontsize=10)
    fig.suptitle(T("Duration per group (one point per child): SLI vowels last almost twice as long"), fontsize=10.5)
    fig.tight_layout(); save(fig, "f2_durations.png"); plt.close(fig)

    fig = plt.figure(figsize=(12, 4.2)); gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.6, 1])
    rec = recordings("o")
    for r_, g in enumerate(("TD", "SLI")):
        dd = d[d.group == g]; pid = int(dd.iloc[(dd.file_s - dd.file_s.median()).abs().argsort().iloc[0]].patient_id)
        w, _, _ = load(rec[pid]); w = w / (np.abs(w).max() + 1e-12); x = np.zeros(16000); x[:min(len(w), 16000)] = w[:16000]
        a = fig.add_subplot(gs[r_, :2]); a.plot(np.arange(16000) / 16000, x, lw=.5, color=TD_C if g == "TD" else SLI_C)
        a.axvspan(len(w) / 16000, 1, color="#d9d9d9", alpha=.6, lw=0)
        a.text(min(len(w), 16000) / 16000 + .01, .7, T("{p}% zeros").format(p=N(100 * (1 - min(len(w), 16000) / 16000), ".0f")), fontsize=9)
        a.set_xlim(0, 1); a.set_ylim(-1.05, 1.05); a.set_ylabel(g); a.set_yticks([])
        if r_ == 1: a.set_xlabel(T("time (s) · input of 16,000 samples, as in Sharma and Singh (2022)"))
    a = fig.add_subplot(gs[:, 2])
    for g, cl in (("TD", TD_C), ("SLI", SLI_C)):
        a.hist(100 * d.zeros_1s[d.group == g], bins=np.linspace(0, 100, 21), alpha=.6, color=cl, label=g, edgecolor="white")
    a.set_xlabel(T("% of the input that is zeros")); a.set_ylabel(T("children")); a.legend(frameon=False, loc="upper left", fontsize=8.5)
    a.set_title(T("Counting zeros separates: AUC = {auc}").format(auc=N(max(auc(d, 'zeros_1s'), 1 - auc(d, 'zeros_1s')))), fontsize=10)
    fig.suptitle(T("Whole file zero-padded to 1 s (gray = padding): the padding length reveals the class"), fontsize=10.5)
    fig.tight_layout(); save(fig, "f3_zero_padding.png"); plt.close(fig)

    MEAS = [("noise_dbfs", "background noise (dBFS)", 1), ("snr_db", "SNR (dB)", 1), ("dc", "DC offset (×1000)", 1000),
            ("skewness", "vowel skewness (polarity)", 1), ("voice_above_4k", "vowel energy > 4 kHz (%)", 100), ("noise_rel_peak", "noise relative to peak (dB)", 1)]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
    for a, (c, t, k) in zip(ax.flat, MEAS):
        box_points(a, [(d[c][d.y == 0].dropna() * k).values, (d[c][d.y == 1].dropna() * k).values], ["TD", "SLI"], [TD_C, SLI_C])
        A = auc(d, c); a.set_title(T("{t}\nAUC = {A} (separates {S})").format(t=T(t), A=N(A), S=N(max(A, 1 - A))), fontsize=9.5)
        if c == "skewness": a.axhline(0, c="k", lw=.7, ls=":")
    fig.suptitle(T("Recording measures per group in the original audio (one point per child; vowel /o/)"), fontsize=10.5)
    fig.tight_layout(); save(fig, "f4_channel_measures.png"); plt.close(fig)

    az = res["chance95_97_children"]; names = list(NO_VOICE); vals = [res["o"]["no_voice"][n] for n in names]
    fig, a = plt.subplots(figsize=(8.5, 3.8)); y = np.arange(len(names))[::-1]
    a.axvspan(1 - az, az, color="#d9d9d9", alpha=.7, lw=0, label=T("chance with 97 children (95%, up to {az})").format(az=N(az)))
    for yy, v in zip(y, vals):
        a.barh(yy, v - .5, left=.5, color=SLI_C if v > az else GRIS, height=.55); a.text(v + .01, yy, N(v), va="center", fontsize=9)
    a.set_yticks(y); a.set_yticklabels([T(n) for n in names]); a.set_xlim(.3, 1.05); a.axvline(.5, c="k", lw=.8, ls=":")
    a.set_xlabel(T("AUC (single measure: direct · several: logistic regression, leave-one-child-out)"))
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=8.5); a.set_title(T("Classifiers that never hear the voice"), fontsize=10)
    save(fig, "f5_no_voice_classifiers.png"); plt.close(fig)

    td = d[d.y == 0]; e16 = (td.sampling_rate == 16000)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for a, (c, t, k) in zip(ax, [("noise_dbfs", "background noise (dBFS)", 1), ("snr_db", "SNR (dB)", 1), ("voice_above_4k", "vowel energy > 4 kHz (%)", 100)]):
        box_points(a, [(td[c][e16].dropna() * k).values, (td[c][~e16].dropna() * k).values], [T("TD\n16 kHz"), T("TD\n44.1 kHz")], [DICT, TD_C])
        a.set_title(T("{t}\nequipment AUC = {A}").format(t=T(t), A=N(res['o']['equipment_within_td'][c])), fontsize=9.5)
    fig.suptitle(T("TD children only: the device leaves its own footprint (no diagnosis involved)"), fontsize=10.5)
    fig.tight_layout(); save(fig, "f6_equipment_td.png"); plt.close(fig)

    E6 = [("sampling_rate", "sampling rate"), ("file_s", "file duration"), ("vowel_s", "vowel duration"), ("silence_s", "silence duration"),
          ("noise_dbfs", "background noise (dBFS)"), ("snr_db", "SNR"), ("dc", "DC offset"), ("skewness", "skewness (polarity)"),
          ("voice_above_4k", "energy > 4 kHz"), ("noise_rel_peak", "noise relative to peak"), ("silence only (noise, 4 kHz, 50 Hz, DC)", "silence only (logistic LOO)")]
    fig, a = plt.subplots(figsize=(8.5, 5)); y = np.arange(len(E6))[::-1]
    for dy, v, cl in ((.17, "o", "#1f7a6d"), (-.17, "a", "#b5542d")):
        vals = [res[v]["no_voice"][k] if k in res[v]["no_voice"] else max(res[v]["auc"][k], 1 - res[v]["auc"][k]) for k, _ in E6]
        a.barh(y + dy, np.array(vals) - .5, left=.5, height=.32, color=cl, label=f"/{v}/ ({res[v]['n']['TD']} TD, {res[v]['n']['SLI']} SLI)")
        for yy, val in zip(y + dy, vals): a.text(val + .01, yy, N(val), va="center", fontsize=8)
    a.set_yticks(y); a.set_yticklabels([T(n) for _, n in E6]); a.set_xlim(.45, 1.06); a.axvline(.5, c="k", lw=.8, ls=":")
    a.set_xlabel(T("TD/SLI separation (AUC, folded to ≥ 0.5)").replace("≥", "$\\geq$")); a.set_title(T("The same shortcuts in both vowels"), fontsize=10)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    save(fig, "e6_vowels.png"); plt.close(fig)

for v in ("o", "a"):
    print(f"/{v}/", res[v]["n"], {k: round(max(x, 1 - x), 2) for k, x in res[v]["auc"].items()})
    print("   no voice:", {k: round(x, 3) for k, x in res[v]["no_voice"].items()})
print("   equipment within TD (/o/):", {k: round(x, 2) for k, x in res["o"]["equipment_within_td"].items()})
