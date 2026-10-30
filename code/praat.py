"""Clinical voice measures with Praat (through parselmouth): what each correction costs, and whether the HNR gap that
remains after "noise for all" is voice or channel (no network is trained; vowel /o/, noise banks of fold 0).

  jitter (local), shimmer (local)  from "To PointProcess (periodic, cc)" with pitch range 150-600 Hz
  HNR                              mean of "To Harmonicity (cc)" (time step 0.01 s, 150 Hz, 0.1, 1 period)

Part 1 (Fig. 9): jitter, shimmer and HNR per group for original, DC + polarity, noise for all and donor (v3).
Part 2 (E8):     HNR with noise added 10, 5, 0 and -5 dB below the vowel, on the whole vowel and on its first 150 ms,
                 next to the SNR gap between groups. SNR_total = -10 log10(10^(-SNR_own/10) + 10^(-S_a/10)).

Outputs: results/praat/praat.json, figures/<lang>/f9_praat.png and figures/<lang>/e8_hnr.png
(<lang> = en, es; FIG_LANG=all draws both from one computation).
Usage: python praat.py                (about 10 minutes, CPU)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call
from sklearn.metrics import roc_auc_score

from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor
from channel_conditions_lanna2 import ChannelConditions
from plotting_lanna2 import SLI_C, TD_C, N, T, each_lang, save, style

PROCESSOR = AudioDataProcessor()
recordings, load, group_of = PROCESSOR.recordings, PROCESSOR.load_recording, PROCESSOR.group_of
SR, RESULTS_DIR = CFG.SAMPLE_RATE, CFG.RESULTS_DIR


def praat(w, s, e):
    snd = parselmouth.Sound(np.asarray(w[s:e], dtype=np.float64), SR)
    pp = call(snd, "To PointProcess (periodic, cc)", 150, 600)
    out = dict(jitter=call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
               shimmer=call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
               hnr=call(call(snd, "To Harmonicity (cc)", 0.01, 150, 0.1, 1.0), "Get mean", 0, 0))
    return {k: (float(v) if v is not None and np.isfinite(v) else np.nan) for k, v in out.items()}


conditions = ChannelConditions(PROCESSOR, set(PROCESSOR.partition(0, "o")["train"]))     # noise banks of fold 0
FORMS = ["original", "dcpol", "noise_for_all", "donor_v3"]
LEVELS = [None, 10, 5, 0, -5]
rows = []
for pid, f in sorted(recordings("o").items()):
    w, s, e = load(f); g = group_of(f); y = int(g == "SLI")
    r = {"patient_id": pid, "y": y}
    for form in FORMS:
        x, _ = conditions.apply(form, w, s, e, pid, "o", g)
        r[form] = praat(x, s, e)
    s0 = conditions.own_snr(w - w.mean(), s, e)[1]
    for lev in LEVELS:
        x, _ = conditions.apply("dcpol" if lev is None else "noise_for_all", w, s, e, pid, "o", g, snr_db=lev)
        snr_t = s0 if lev is None else -10 * np.log10((10 ** (-s0 / 10) if np.isfinite(s0) else 0) + 10 ** (-lev / 10))
        r[f"level_{lev}"] = {"hnr": praat(x, s, e)["hnr"], "hnr_150ms": praat(x, s, min(e, s + int(0.150 * SR)))["hnr"],
                             "snr_total": float(snr_t) if np.isfinite(snr_t) else np.nan}
    rows.append(r)
y = np.array([r["y"] for r in rows])


def summary(vals):
    v = np.array(vals, float); ok = np.isfinite(v)
    td, sli = float(np.median(v[ok & (y == 0)])), float(np.median(v[ok & (y == 1)]))
    return {"median_TD": td, "median_SLI": sli, "difference": sli - td, "auc": float(roc_auc_score(y[ok], v[ok]))}


res = {"forms": {f: {m: summary([r[f][m] for r in rows]) for m in ("jitter", "shimmer", "hnr")} for f in FORMS},
       "levels": {str(l): {m: summary([r[f"level_{l}"][m] for r in rows]) for m in ("hnr", "hnr_150ms", "snr_total")} for l in LEVELS}}
out = RESULTS_DIR / "praat"; out.mkdir(parents=True, exist_ok=True)
json.dump({"summary": res, "per_child": rows}, open(out / "praat.json", "w"), indent=1)
for f in FORMS:
    print(f, {m: (round(v["median_TD"], 4), round(v["median_SLI"], 4), round(max(v["auc"], 1 - v["auc"]), 2)) for m, v in res["forms"][f].items()})
for l in LEVELS:
    print("level", l, {m: (round(v["difference"], 2), round(max(v["auc"], 1 - v["auc"]), 2)) for m, v in res["levels"][str(l)].items()})

style()
NAMES = {"original": "original", "dcpol": "DC +\npolarity", "noise_for_all": "noise\nfor all", "donor_v3": "donor\n(v3)"}
for _ in each_lang():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, (m, title, k) in zip(ax, [("jitter", "local jitter (%)", 100), ("shimmer", "local shimmer (%)", 100), ("hnr", "HNR (dB)", 1)]):
        for j, f in enumerate(FORMS):
            for cl, dx, cls in ((TD_C, -.18, 0), (SLI_C, .18, 1)):
                v = np.array([r[f][m] * k for r in rows if r["y"] == cls and np.isfinite(r[f][m])])
                bx = a.boxplot(v, positions=[j + dx], widths=.3, showfliers=False, patch_artist=True, medianprops=dict(color="k"))
                bx["boxes"][0].set(facecolor=cl, alpha=.55, edgecolor=cl)
        a.set_xticks(range(len(FORMS))); a.set_xticklabels([T(NAMES[f]) for f in FORMS], fontsize=9); a.set_title(T(title), fontsize=10, pad=20)
        top = a.get_ylim()[1]
        for j, f in enumerate(FORMS):
            A = res["forms"][f][m]["auc"]; a.text(j, top, f"AUC {N(max(A, 1 - A))}", ha="center", va="bottom", fontsize=8.5, color="#444")
    ax[0].scatter([], [], color=TD_C, alpha=.55, marker="s", label="TD"); ax[0].scatter([], [], color=SLI_C, alpha=.55, marker="s", label="SLI")
    ax[0].legend(frameon=False, loc="upper left")
    fig.suptitle(T("Clinical voice measures (Praat) per condition · top: separation between TD and SLI"), fontsize=10.5, y=1.04)
    fig.tight_layout(); save(fig, "f9_praat.png"); plt.close(fig)

    ticks = [T("no noise") if l is None else f"{l} dB" for l in LEVELS]; xs = np.arange(len(LEVELS))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key, name, cl, mk in (("snr_total", "total SNR", "#8a949a", "s"), ("hnr", "HNR (whole vowel)", SLI_C, "o"), ("hnr_150ms", "HNR (first 150 ms)", "#1f7a6d", "D")):
        ax[0].plot(xs, [res["levels"][str(l)][key]["difference"] for l in LEVELS], marker=mk, color=cl, label=T(name))
        ax[1].plot(xs, [max(res["levels"][str(l)][key]["auc"], 1 - res["levels"][str(l)][key]["auc"]) for l in LEVELS], marker=mk, color=cl, label=T(name))
    for a in ax:
        a.set_xticks(xs); a.set_xticklabels(ticks); a.set_xlabel(T("added noise (dB below the vowel)")); a.legend(frameon=False, fontsize=8.5)
    ax[0].axhline(0, c="k", lw=.7, ls=":"); ax[0].set_ylabel(T("SLI − TD difference (dB, medians)"))
    ax[1].axhline(.5, c="k", lw=.7, ls=":"); ax[1].set_ylabel(T("AUC")); ax[1].set_title(T("TD/SLI separation (AUC)"), fontsize=10)
    fig.suptitle(T("Is the HNR gap voice or channel? If channel, it vanishes when the SNR gap does"), fontsize=10.5)
    fig.tight_layout(); save(fig, "e8_hnr.png"); plt.close(fig)
