"""
What each correction does to the VOICE (vowel /o/, fold 0: the noise bank comes from its training children).

The corrections of sharma_lanna2.build_inputs that alter the signal (gain, dc, pol, noise, lp4k) are applied to the whole
16-kHz file exactly as there, and the vowel segment (.lbl) is measured with Praat (parselmouth) before and after:
  F1, F2     median formant frequencies (Burg, 5 formants up to 7000 Hz: children), relative change |after/before - 1|
  F0         median pitch (75-600 Hz), absolute change in semitones
  HNR        harmonics-to-noise ratio (cc), change in dB (after - before)
  jitter, shimmer (local), relative change
seg and dur only change which samples the network receives (the vowel is not altered), so they are not listed.
Output: results/analysis/voice_checks.csv (one row per child and correction) and voice_checks.txt (medians).
Usage: python voice_checks.py      (CPU, a few minutes)
"""
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

from channel_conditions_lanna2 import ChannelConditions
from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor
from sharma_lanna2 import lowpass_4k

SR, FOLD = CFG.SAMPLE_RATE, 0
OUT = CFG.CODE_DIR / "results" / "analysis"
CORRECTIONS = [("gain",), ("dc",), ("pol",), ("noise",), ("lp4k",), ("gain", "dc", "pol", "noise", "lp4k")]


def corrected(flags, w, s, e, cc, rng):
    """The same operations and order as sharma_lanna2.build_inputs (dc, noise, pol, lp4k; gain on what is measured)."""
    if "dc" in flags:
        w = w - w.mean()
    if "noise" in flags:
        a = np.sqrt(cc.power(w[s:e]))
        w = w + a * 10 ** (-CFG.NOISE_SNR_DB / 20) * cc.synth_noise(cc.bank(), len(w), rng)
        if "dc" in flags:
            w = w - w.mean()
    if "pol" in flags:
        w = cc.canonical_polarity(w, s, e)
    if "lp4k" in flags:
        w = lowpass_4k(w)
    v = w[s:e]
    if "gain" in flags:
        v = v / (np.abs(v).max() + 1e-12)
    return v


def praat(v):
    snd = parselmouth.Sound(np.asarray(v, dtype=np.float64), SR)
    fm = call(snd, "To Formant (burg)", 0.0, 5, 7000, 0.025, 50)
    t = np.arange(0.03, snd.duration - 0.03, 0.01)
    f1 = np.nanmedian([call(fm, "Get value at time", 1, x, "Hertz", "Linear") for x in t])
    f2 = np.nanmedian([call(fm, "Get value at time", 2, x, "Hertz", "Linear") for x in t])
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    f0 = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    return dict(F1=f1, F2=f2, F0=f0,
                HNR=call(call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0),
                jitter=call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
                shimmer=call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))


def main():
    proc = AudioDataProcessor()
    cc = ChannelConditions(proc, set(proc.partition(FOLD, "o")["train"]))
    rows = []
    for pid, f in sorted(proc.recordings("o").items()):
        w, s, e = proc.load_recording(f)
        ref = praat(w[s:e])
        for flags in CORRECTIONS:
            rng = np.random.default_rng(1000 * pid + ord("o"))          # same noise as build_inputs
            m = praat(corrected(flags, w, s, e, cc, rng))
            rows.append(dict(patient_id=pid, group=proc.group_of(f), correction="+".join(flags) if len(flags) < 5 else "all",
                             dF1_rel=abs(m["F1"] / ref["F1"] - 1), dF2_rel=abs(m["F2"] / ref["F2"] - 1),
                             dF0_st=abs(12 * np.log2(m["F0"] / ref["F0"])), dHNR_dB=m["HNR"] - ref["HNR"],
                             djitter_rel=m["jitter"] / ref["jitter"] - 1, dshimmer_rel=m["shimmer"] / ref["shimmer"] - 1))
    R = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); R.to_csv(OUT / "voice_checks.csv", index=False)
    T = R.groupby("correction", sort=False)[["dF1_rel", "dF2_rel", "dF0_st", "dHNR_dB", "djitter_rel", "dshimmer_rel"]].median().round(3)
    TG = R.groupby(["correction", "group"], sort=False)["dHNR_dB"].median().unstack().round(2)
    txt = ("Voice measures before vs after each correction (vowel /o/, fold 0, medians over children). dF1/dF2: relative "
           "|change|; dF0: |semitones|; dHNR: dB (after - before); jitter/shimmer: relative change.\n\n"
           + T.to_string() + "\n\nHNR change by group (dB):\n" + TG.to_string() + "\n")
    open(OUT / "voice_checks.txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
