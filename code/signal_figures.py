"""Part II figures of the paper: what "noise for all" does to a recording, and why it removes the channel.

  f7  a typical TD and a typical SLI /o/ (the recordings with the median SNR of their group, with at least
      100 ms before the vowel), original and with noise for all, with the long-term spectrum of the vowel
  f8  SNR compression: adding independent noise S_a dB below the vowel turns an SNR S_0 into
      SNR_total = -10 log10(10^(-S_0/10) + 10^(-S_a/10)); medians per group before and after
Needs results/audit/measures_o.csv (run audit.py first). Noise bank: training TD children of fold 0.
Figures go to figures/<lang>/ (<lang> = en, es; FIG_LANG=all draws both).
Usage: python signal_figures.py          (about 1 minute, CPU)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor
from channel_conditions_lanna2 import ChannelConditions
from plotting_lanna2 import GRIS, SLI_C, TD_C, N, T, each_lang, save, style

PROCESSOR = AudioDataProcessor()
recordings, load, group_of = PROCESSOR.recordings, PROCESSOR.load_recording, PROCESSOR.group_of
SR, RESULTS_DIR = CFG.SAMPLE_RATE, CFG.RESULTS_DIR

style()
conditions = ChannelConditions(PROCESSOR, set(PROCESSOR.partition(0, "o")["train"]))     # noise banks of fold 0
rec = recordings("o")


def typical(group):
    cand = []
    for pid, f in sorted(rec.items()):
        if group_of(f) != group:
            continue
        w, s, e = load(f); snr = conditions.own_snr(w - w.mean(), s, e)[1]
        if np.isfinite(snr) and s > 1600:
            cand.append((snr, pid))
    cand.sort()
    return cand[len(cand) // 2][1]


EXAMPLES = {}                                            # the signals of f7, computed once for every language
for g in ("TD", "SLI"):
    pid = typical(g); f = rec[pid]; w, s, e = load(f)
    o = w / np.abs(w[s:e]).max()
    m, _ = conditions.apply("noise_for_all", w, s, e, pid, "o", g); m = m / np.abs(m[s:e]).max()
    EXAMPLES[g] = (f, w, s, e, o, m)

D = pd.read_csv(RESULTS_DIR / "audit" / "measures_o.csv")
total = lambda s0, sa: -10 * np.log10(10 ** (-s0 / 10) + 10 ** (-sa / 10))

for _ in each_lang():
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.4), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    for r, g in enumerate(("TD", "SLI")):
        f, w, s, e, o, m = EXAMPLES[g]
        t = np.arange(len(w)) / SR
        for c, (x, title, cl) in enumerate([(o, "original (vowel peak = 1)", "#5b6770"), (m, "noise for all (vowel peak = 1)", SLI_C)]):
            a = ax[r, c]; a.plot(t, x, lw=.5, color=cl); a.axvspan(s / SR, e / SR, color="#f2c94c", alpha=.25, lw=0)
            a.set_title(f"{g} ({N(sf.info(f).samplerate / 1000, 'g')} kHz) · {T(title)}", fontsize=10); a.set_xlabel(T("time (s)")); a.set_ylabel(T("amplitude"))
        a = ax[r, 2]
        for x, lab, cl in ((o, "original", "#5b6770"), (m, "noise for all", SLI_C)):
            v = x[s:e] / np.sqrt(np.mean(x[s:e] ** 2)); P = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2
            fr = np.fft.rfftfreq(len(v), 1 / SR); a.plot(fr, 10 * np.log10(np.convolve(P, np.ones(25) / 25, "same") + 1e-9), lw=1, color=cl, label=T(lab))
        a.set_xlim(0, 8000); a.set_xlabel(T("frequency (Hz)")); a.set_ylabel(T("energy (dB)")); a.set_title(T("{g} · vowel spectrum").format(g=g), fontsize=10); a.legend(fontsize=8)
    fig.suptitle(T("Two typical /o/ recordings before and after noise for all (shaded = vowel seen by the network; noise added 5 dB below the vowel)"), fontsize=11)
    fig.tight_layout(); save(fig, "f7_noise_for_all_example.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); s0 = np.linspace(0, 35, 300); a = ax[0]
    for sa, ls in ((0, ":"), (5, "-"), (10, "--")):
        a.plot(s0, total(s0, sa), ls=ls, color="#1f7a6d" if sa == 5 else GRIS, lw=2 if sa == 5 else 1.3, label=T("noise added {sa} dB below the voice").format(sa=sa))
    a.plot(s0, s0, color="k", lw=.7, ls=":", label=T("nothing added (identity)"))
    after = {}
    for g, cl in (("TD", TD_C), ("SLI", SLI_C)):
        m = float(D.snr_db[D.group == g].median()); t = float(total(m, 5)); after[g] = t
        a.scatter([m], [t], s=70, color=cl, zorder=4); a.annotate(f"{g}: {N(m, '.1f')} → {N(t, '.1f')} dB", (m, t), (m + 1.5, t - 3.4) if g == "TD" else (m + 1, t + 2.2), fontsize=9, color=cl)
    a.set_xlabel(T("original SNR (dB)")); a.set_ylabel(T("SNR with noise (dB)")); a.set_xlim(0, 35); a.set_ylim(-2, 20)
    hl = a.get_legend_handles_labels(); a.set_title(T("Every SNR is compressed\ntowards that of the added noise"), fontsize=10)
    a = ax[1]
    for g, cl in (("TD", TD_C), ("SLI", SLI_C)):
        v = D.snr_db[D.group == g].dropna().values
        a.hist(v, bins=np.linspace(-2, 32, 25), alpha=.45, color=cl, label=T("{g} original").format(g=g), edgecolor="white")
        a.hist(total(v, 5), bins=np.linspace(-2, 32, 69), alpha=.85, color=cl, label=T("{g} noise for all").format(g=g), histtype="step", lw=2)
    d0 = float(D.snr_db[D.group == "SLI"].median() - D.snr_db[D.group == "TD"].median())
    a.set_xlabel(T("SNR (dB)")); a.set_ylabel(T("children"))
    hr = a.get_legend_handles_labels()
    # legends in their own strip under the panels, so they never cover the curves or the axis labels
    fig.legend(*hl, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=8, frameon=False)
    fig.legend(*hr, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2, fontsize=8, frameon=False)
    fig._print_rect = (0, 0.14, 1, 1)
    a.set_title(T("Difference of medians:\n{a} dB → {b} dB").format(a=N(d0, ".1f"), b=N(after['SLI'] - after['TD'])), fontsize=10)
    fig.suptitle(T("Noise for all, in numbers: each device's own noise ends up under the added noise"), fontsize=10.5)
    fig.tight_layout(); save(fig, "f8_snr_compression.png"); plt.close(fig)
print(f"SNR medians: TD {D.snr_db[D.group == 'TD'].median():.2f} -> {after['TD']:.2f} dB, SLI {D.snr_db[D.group == 'SLI'].median():.2f} -> {after['SLI']:.2f} dB")
