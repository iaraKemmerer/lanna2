"""
Intelligibility controls of each correction (beyond the acoustic measures of voice_checks.py), on the five sustained
vowels /a/ /e/ /i/ /o/ /u/ of every child. The corrections that alter the signal (gain, dc, pol, noise, lp4k, and all
of them) are applied to the whole 16-kHz file exactly as in sharma_lanna2.build_inputs (fold 0: the noise bank comes
from its training children) and the vowel segment (.lbl) is compared with the original one:

(1) Vowel recognition, train on ORIGINAL, test on CORRECTED. 5-way logistic regression on 12 MFCC (c1-c12, mean and
    SD over the vowel; c0 is left out, so the level does not count), 10-fold cross-validation by CHILD (the five vowels
    of a child always on the same side). The classifier never sees corrected audio: a drop says that the correction
    makes the vowel less recognizable from what an original vowel looks like. "matched" trains and tests on corrected
    audio (does the vowel information survive at all?). Chance = 0.20. 95% CI: bootstrap over children (2000).
(2) ESTOI (extended short-time objective intelligibility, Jensen & Taal 2016; pystoi), the original vowel as the
    reference. ESTOI needs 384 ms of speech and most LANNA vowels are shorter, so it is computed per child on the five
    vowels concatenated (the same order, original and corrected). 1 = identical, lower = less intelligible.
(3) SQUIM objective (Kumar et al. 2023; torchaudio): reference-free estimates of STOI, PESQ and SI-SDR of each vowel,
    original and corrected (both peak-normalized). Trained on adult speech: read as a relative change.
seg and dur only change which samples the network receives (the vowel is not altered); gain is a scaling, to which
the three measures are invariant by construction (kept as a check).

Output: results/analysis/intelligibility_checks.csv (one row per child, vowel and correction),
        intelligibility_summary.csv and intelligibility_checks.txt (table of the paper)
Usage: python intelligibility_checks.py      (CPU, ~10 min; downloads the SQUIM weights once, 28 MB)
"""
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from pystoi import stoi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from channel_conditions_lanna2 import ChannelConditions
from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor
from voice_checks import corrected

warnings.filterwarnings("ignore")
SR, FOLD, B = CFG.SAMPLE_RATE, 0, 2000
VOWELS = "aeiou"
OUT = CFG.CODE_DIR / "results" / "analysis"
CORRECTIONS = [(), ("gain",), ("dc",), ("pol",), ("noise",), ("lp4k",), ("gain", "dc", "pol", "noise", "lp4k")]


def name_of(flags):
    return "none" if not flags else ("all" if len(flags) == 5 else "+".join(flags))


def mfcc_stats(v):
    m = librosa.feature.mfcc(y=v.astype(np.float32), sr=SR, n_mfcc=13, n_fft=512, hop_length=160)[1:]
    return np.r_[m.mean(1), m.std(1)]


def vowel_recognition(F, ref):
    """F, ref: DataFrames (patient_id, vowel, features...) of a correction and of the original. Returns the per-row
    predictions of 'train on original' and 'matched'."""
    ids = F.patient_id.values
    cols = [c for c in F.columns if c.startswith("m")]
    pred_orig, pred_match = np.empty(len(F), dtype=object), np.empty(len(F), dtype=object)
    for tr, te in GroupKFold(n_splits=10).split(F, groups=ids):
        train_kids = set(ids[tr])
        R = ref[ref.patient_id.isin(train_kids)]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        pred_orig[te] = clf.fit(R[cols].values, R.vowel.values).predict(F.iloc[te][cols].values)
        pred_match[te] = clf.fit(F.iloc[tr][cols].values, F.iloc[tr].vowel.values).predict(F.iloc[te][cols].values)
    return pred_orig, pred_match


def boot_ci(correct, kids, rng):
    """Accuracy and 95% CI resampling children (all the vowels of a drawn child)."""
    u = np.unique(kids)
    per = {k: correct[kids == k] for k in u}
    acc = [np.concatenate([per[k] for k in rng.choice(u, len(u))]).mean() for _ in range(B)]
    return correct.mean(), np.percentile(acc, 2.5), np.percentile(acc, 97.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AudioDataProcessor()
    cc = ChannelConditions(proc, set(proc.partition(FOLD, "o")["train"]))
    squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
    recs = {v: proc.recordings(v) for v in VOWELS}
    kids = sorted(set.intersection(*[set(r) for r in recs.values()]))      # children with the five vowels
    group = {pid: proc.group_of(recs["o"][pid]) for pid in kids}
    rows, concat = [], {}
    for pid in kids:
        for v in VOWELS:
            w, s, e = proc.load_recording(recs[v][pid])
            for flags in CORRECTIONS:
                rng = np.random.default_rng(1000 * pid + ord(v))           # same noise as build_inputs
                x = corrected(flags, w, s, e, cc, rng) if flags else w[s:e]
                x = x / (np.abs(x).max() + 1e-12)
                concat.setdefault((pid, name_of(flags)), []).append(x)
                with torch.no_grad():
                    q_stoi, q_pesq, q_sisdr = (t.item() for t in squim(torch.tensor(x, dtype=torch.float32)[None]))
                rows.append(dict(patient_id=pid, group=group[pid], vowel=v, correction=name_of(flags),
                                 squim_stoi=q_stoi, squim_pesq=q_pesq, squim_sisdr=q_sisdr,
                                 **{f"m{k}": c for k, c in enumerate(mfcc_stats(x))}))
        print(f"child {pid} ({kids.index(pid) + 1}/{len(kids)})", flush=True)
    R = pd.DataFrame(rows)
    estoi = {(pid, c): stoi(np.concatenate(concat[(pid, "none")]), np.concatenate(x), SR, extended=True)
             for (pid, c), x in concat.items()}
    R["estoi_child"] = [estoi[(p, c)] for p, c in zip(R.patient_id, R.correction)]
    ref = R[R.correction == "none"].reset_index(drop=True)
    summary = []
    for flags in CORRECTIONS:
        c = name_of(flags)
        F = R[R.correction == c].reset_index(drop=True)
        po, pm = vowel_recognition(F, ref)
        R.loc[R.correction == c, "pred_train_original"] = po
        R.loc[R.correction == c, "pred_matched"] = pm
        ok_o, ok_m = (po == F.vowel.values).astype(float), (pm == F.vowel.values).astype(float)
        a, lo, hi = boot_ci(ok_o, F.patient_id.values, np.random.default_rng(0))
        am, lom, him = boot_ci(ok_m, F.patient_id.values, np.random.default_rng(0))
        td, sli = F.group.values == "TD", F.group.values == "SLI"
        d = F.groupby("vowel")[["squim_stoi", "squim_pesq", "squim_sisdr"]].median()
        D = F[["squim_stoi", "squim_pesq", "squim_sisdr"]].values - ref[["squim_stoi", "squim_pesq", "squim_sisdr"]].values
        E = F.drop_duplicates("patient_id")
        summary.append(dict(correction=c, n_children=F.patient_id.nunique(),
                            vowel_acc_train_original=a, ci_low=lo, ci_high=hi,
                            vowel_acc_td=ok_o[td].mean(), vowel_acc_sli=ok_o[sli].mean(),
                            vowel_acc_matched=am, ci_low_matched=lom, ci_high_matched=him,
                            estoi_median=E.estoi_child.median(), estoi_p10=E.estoi_child.quantile(.1),
                            estoi_td=E[E.group == "TD"].estoi_child.median(), estoi_sli=E[E.group == "SLI"].estoi_child.median(),
                            squim_stoi=F.squim_stoi.median(), squim_pesq=F.squim_pesq.median(), squim_sisdr=F.squim_sisdr.median(),
                            d_squim_stoi=np.median(D[:, 0]), d_squim_pesq=np.median(D[:, 1]), d_squim_sisdr=np.median(D[:, 2])))
    S = pd.DataFrame(summary)
    R.drop(columns=[c for c in R.columns if c.startswith("m") and c[1:].isdigit()]).to_csv(OUT / "intelligibility_checks.csv", index=False)
    S.to_csv(OUT / "intelligibility_summary.csv", index=False)
    conf = pd.crosstab(R[R.correction == "all"].vowel, R[R.correction == "all"].pred_train_original)
    txt = (f"Intelligibility per correction ({len(kids)} children with the five vowels; noise bank of fold 0).\n"
           "vowel_acc: 5-way vowel recognition, trained on original vowels, tested on corrected vowels of other children "
           "(10-fold by child; chance 0.20); matched: trained and tested on corrected. ESTOI per child (five vowels "
           "concatenated, original as reference). SQUIM: reference-free estimates (medians over vowels).\n\n"
           + S.round(3).to_string(index=False) + "\n\nConfusions with all corrections (rows: true vowel):\n"
           + conf.to_string() + "\n")
    open(OUT / "intelligibility_checks.txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
