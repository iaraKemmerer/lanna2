"""
Models that never hear the voice: an MLP (and a logistic regression) that receives ONLY confound measures of each
child, no audio. Same subject-disjoint folds as the networks (48 train / 24 validation / 24 test children), with
the .lbl corrections.

Measures per child (vowel /o/), on the 16-kHz recording:
  duration       vowel duration (.lbl segment), s
  level          peak of the whole file, dBFS (what a network without normalization sees, e.g. Sharma and Singh)
and, "as the network sees them" (after peak-normalizing the vowel, as every network of the paper does):
  dc             DC offset of the file divided by the vowel peak
  polarity       skewness of the vowel (its sign is the polarity)
  noise          background-noise level relative to the vowel peak, dB
  snr            vowel level minus noise level, dB
  above_4k       fraction of vowel energy above 4 kHz
Missing values (files without measurable silence) get the training median.

Model: the same MLP in every fold (Linear -> BatchNorm -> Cauchy -> Dropout, hidden layer of 8 units as the MLP of the
fold-2 network, dropout 0.1), trained like the networks: SGD (momentum 0.9, weight decay 0.01 except on the Cauchy
parameters), balanced batches of 8 TD + 8 SLI children, the epoch with the best validation F1, 5 runs (seeds
7000..7004). With one example per child there are only 3 batches per epoch, so the learning rates of the networks
(0.0002-0.006, tuned for hundreds of windows per child) barely train it: learning rate 0.01 and 300 epochs instead. Inputs standardized with the training children of the fold. Reference:
logistic regression on the same inputs (train only, no validation needed).

Outputs (results/): measures_o.csv, results.csv (one row per input set, fold, model and run), table.csv (means),
summary.txt, and the figure figures/<es|en>/mlp_confounds.png.
Usage: python mlp_confusores.py        (CPU, a few minutes)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from config_lanna2 import CFG
from dataset_lanna2 import AudioDataProcessor

HERE = CFG.CODE_DIR
OUT, FIG = HERE / "results", HERE / "figures"
EPOCHS, RUNS, HALF_BATCH = int(__import__("os").environ.get("EPOCHS", 300)), int(__import__("os").environ.get("RUNS", 5)), 8

MLP_CONFIG = {"mlp_hidden": [8], "mlp_dropout": 0.1, "learning_rate": 0.01}
# First configuration that was tried: the final MLP of the network selected for each fold (hidden layers, dropout and
# learning rate), 150 epochs. Kept to document why it was replaced, using VALIDATION results only:
#   MLP_MODE=networks python mlp_confusores.py   ->  results_networks/
NETWORK_MLPS = {0: {"mlp_hidden": [8, 6, 12], "mlp_dropout": 0.0946, "learning_rate": 0.0060},
                1: {"mlp_hidden": [8, 25], "mlp_dropout": 0.1149, "learning_rate": 0.0009},
                2: {"mlp_hidden": [8], "mlp_dropout": 0.0553, "learning_rate": 0.0002}}
MODE = __import__("os").environ.get("MLP_MODE", "common")
if MODE == "networks":
    OUT, EPOCHS = HERE / "results_networks", 150
INPUT_SETS = {
    "duration": ["duration"], "level": ["level"], "dc": ["dc"], "polarity": ["polarity"], "noise": ["noise"],
    "snr": ["snr"], "above_4k": ["above_4k"],
    "channel (noise, snr, dc, polarity)": ["noise", "snr", "dc", "polarity"],
    "channel + above_4k": ["noise", "snr", "dc", "polarity", "above_4k"],
    "all (channel + above_4k + duration)": ["noise", "snr", "dc", "polarity", "above_4k", "duration"],
    "all + level": ["noise", "snr", "dc", "polarity", "above_4k", "duration", "level"],
}
LABELS = {"es": {"duration": "duración", "level": "nivel de grabación", "dc": "componente continua", "polarity": "polaridad",
                 "noise": "ruido de fondo", "snr": "SNR", "above_4k": "energía > 4 kHz",
                 "channel (noise, snr, dc, polarity)": "canal (ruido, SNR, continua, polaridad)",
                 "channel + above_4k": "canal + agudos", "all (channel + above_4k + duration)": "canal + agudos + duración",
                 "all + level": "todo + nivel", "fold": "partición", "xlabel": "AUC en prueba (24 niños por partición)",
                 "title": "Un MLP que recibe solo confusores, sin audio", "chance": "azar (95 %)"},
          "en": {"duration": "duration", "level": "recording level", "dc": "DC offset", "polarity": "polarity",
                 "noise": "background noise", "snr": "SNR", "above_4k": "energy > 4 kHz",
                 "channel (noise, snr, dc, polarity)": "channel (noise, SNR, DC, polarity)",
                 "channel + above_4k": "channel + high frequencies", "all (channel + above_4k + duration)": "channel + high freq. + duration",
                 "all + level": "all + level", "fold": "fold", "xlabel": "test AUC (24 children per fold)",
                 "title": "An MLP that receives only confounds, no audio", "chance": "chance (95%)"}}


# =============================================================
# MEASURES
# =============================================================
def dbfs(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def above_4k(x):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / CFG.SAMPLE_RATE)
    return float(X[f >= 4000].sum() / (X[f > 20].sum() + 1e-20))


def measures() -> pd.DataFrame:
    proc, rows = AudioDataProcessor(), []
    for pid, f in sorted(proc.recordings("o").items()):
        w, s, e = proc.load_recording(f)
        voice = w[s:e]; v = voice - voice.mean(); peak = np.abs(voice).max()
        sil = np.concatenate(proc.silence_parts(w, s, e) or [np.zeros(0)])
        noise = dbfs(sil) if len(sil) > CFG.SILENCE_MARGIN else np.nan
        rows.append(dict(patient_id=pid, y=CFG.LABEL_MAP[proc.group_of(f)], duration=len(voice) / CFG.SAMPLE_RATE,
                         level=float(20 * np.log10(np.abs(w).max() + 1e-12)), dc=float(w.mean() / peak),
                         polarity=float(np.mean(v ** 3) / (np.mean(v ** 2) ** 1.5 + 1e-12)),
                         noise=noise - 20 * np.log10(peak) if np.isfinite(noise) else np.nan,
                         snr=dbfs(voice) - noise if np.isfinite(noise) else np.nan, above_4k=above_4k(voice)))
    return pd.DataFrame(rows).set_index("patient_id")


# =============================================================
# MODEL
# =============================================================
class Cauchy(nn.Module):
    """(lambda1 x + lambda2) / (x^2 + d^2 + eps), three learnable scalars (as in the networks, with eps)."""

    def __init__(self):
        super().__init__()
        self.lambda1, self.lambda2, self.d = (nn.Parameter(torch.tensor(1.0)) for _ in range(3))

    def forward(self, x):
        den = x ** 2 + self.d ** 2 + 1e-6
        return (self.lambda1 * x + self.lambda2) / den


def mlp(n_in, hidden, dropout):
    layers, sizes = [], [n_in] + hidden
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.BatchNorm1d(b), Cauchy(), nn.Dropout(dropout)]
    layers.append(nn.Linear(sizes[-1], 2))
    m = nn.Sequential(*layers)
    for l in m:
        if isinstance(l, nn.Linear):
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
    return m


def train_mlp(arch, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = mlp(Xtr.shape[1], arch["mlp_hidden"], arch["mlp_dropout"])
    cauchy = {id(p) for l in m.modules() if isinstance(l, Cauchy) for p in l.parameters()}
    opt = torch.optim.SGD([{"params": [p for p in m.parameters() if id(p) not in cauchy]},
                           {"params": [p for p in m.parameters() if id(p) in cauchy], "weight_decay": 0.0}],
                          lr=arch["learning_rate"], momentum=0.9, weight_decay=0.01)
    Xtr, Xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    cls = [np.nonzero(ytr == c)[0] for c in (0, 1)]
    best, best_state = -1.0, None
    for _ in range(EPOCHS):
        m.train()
        perm = [rng.permutation(c) for c in cls]
        for b in range(min(len(c) for c in cls) // HALF_BATCH):        # balanced batches, as the networks
            idx = np.concatenate([p[b * HALF_BATCH:(b + 1) * HALF_BATCH] for p in perm])
            opt.zero_grad(); loss = F.cross_entropy(m(Xtr[idx]), ytr_t[idx])
            if torch.isfinite(loss):
                loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pva = F.softmax(m(Xva), 1)[:, 1].numpy()
        f1 = f1_score(yva, (pva > 0.5).astype(int), average="macro", zero_division=0) if np.all(np.isfinite(pva)) else -1
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    m.best_val_f1 = float(best)
    return m


# =============================================================
# MAIN
# =============================================================
def main():
    OUT.mkdir(exist_ok=True)
    M = measures(); M.to_csv(OUT / "measures_o.csv")
    rows, preds = [], []
    for fold in range(CFG.N_FOLDS):
        part = AudioDataProcessor.partition(fold, "o")
        arch = NETWORK_MLPS[fold] if MODE == "networks" else MLP_CONFIG
        tr, va, te = (M.loc[[p for p in part[s] if p in M.index]] for s in ("train", "val", "test"))
        for name, cols in INPUT_SETS.items():
            med, mu, sd = tr[cols].median(), tr[cols].mean(), tr[cols].std() + 1e-12
            prep = lambda d: ((d[cols].fillna(med) - mu) / sd).values
            Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
            ytr, yva, yte = tr.y.values, va.y.values, te.y.values
            p = LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            rows.append(dict(inputs=name, fold=fold, model="logistic", run=1, auc=roc_auc_score(yte, p),
                             f1=f1_score(yte, (p > .5).astype(int), average="macro")))
            for k in range(RUNS):
                m = train_mlp(arch, Xtr, ytr, Xva, yva, CFG.RUN_SEED_BASE + k)
                with torch.no_grad():
                    p = F.softmax(m(torch.tensor(Xte, dtype=torch.float32)), 1)[:, 1].numpy()
                rows.append(dict(inputs=name, fold=fold, model="mlp", run=k + 1, auc=roc_auc_score(yte, p),
                                 f1=f1_score(yte, (p > .5).astype(int), average="macro"), val_f1=m.best_val_f1))
                preds += [dict(inputs=name, fold=fold, run=k + 1, patient_id=i, label=int(l), p_sli=float(q))
                          for i, l, q in zip(te.index, yte, p)]
            r = [x for x in rows if x["inputs"] == name and x["fold"] == fold]
            print(f"fold {fold} | {name:38s} | MLP AUC {np.mean([x['auc'] for x in r if x['model'] == 'mlp']):.3f} "
                  f"F1 {np.mean([x['f1'] for x in r if x['model'] == 'mlp']):.3f} | logistic AUC {r[0]['auc']:.3f}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False)
    T = R.groupby(["inputs", "model", "fold"], sort=False)[["auc", "f1"]].mean().unstack("fold").round(3)
    T.to_csv(OUT / "table.csv")
    # chance: 95th percentile of the AUC of a random score on 24 children (12 + 12)
    rng = np.random.default_rng(0); y24 = np.r_[np.zeros(12), np.ones(12)]
    chance = float(np.quantile([roc_auc_score(y24, rng.random(24)) for _ in range(10000)], .95))
    with open(OUT / "summary.txt", "w") as f:
        f.write(f"Models that receive only confound measures (no audio), vowel /o/, test AUC and F1 per fold "
                f"(MLP: mean of {RUNS} runs). Chance (95%, 24 children): AUC {chance:.2f}\n\n{T.to_string()}\n")
    if MODE != "networks":
        plot(R, chance)
    print(T.to_string())


# Printed size of each figure in the paper (inches; text width of the paper = 6.5 in). Figures are drawn at this size,
# so the text inside them is printed at its real point size (about 9-10 pt) instead of being shrunk by LaTeX.
PRINT_SIZE = {
    "f1_sampling_rate.png": (4.2, 2.4), "f2_durations.png": (6.5, 2.7), "f3_zero_padding.png": (6.5, 2.9),
    "f4_channel_measures.png": (6.5, 5.4), "f5_no_voice_classifiers.png": (5.4, 2.7), "f6_equipment_td.png": (6.0, 2.7),
    "e6_vowels.png": (5.2, 3.5), "f8_snr_compression.png": (6.2, 2.8), "mlp_confounds.png": (5.2, 3.6),
    "sharma_factorial.png": (6.5, 5.7), "sharma_effects.png": (6.5, 3.3),
}


def fit_to_print(fig, name, rect=None):
    """Resize the figure to its printed size (PRINT_SIZE; other figures: 6.5 in wide, same aspect) and lay it out again."""
    W, H = fig.get_size_inches()
    w, h = PRINT_SIZE.get(name, (6.5, H * 6.5 / W))
    fig.set_size_inches(w, h)
    try:
        fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    except Exception:
        pass


def paper_font():
    """
    Figures in the typeface of the paper: Latin Modern Roman (the OpenType version of LaTeX's Computer Modern), with
    Computer Modern for math. It is searched in the TeX installation; without it, a generic serif is used.
    """
    import glob
    from matplotlib import font_manager
    found = []
    for d in ("/usr/share/texmf/fonts/opentype/public/lm", "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm",
              "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm", "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm"):
        for f in glob.glob(f"{d}/lmroman10-*.otf"):
            font_manager.fontManager.addfont(f); found.append(f)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.unicode_minus": True,
                         "font.serif": (["Latin Modern Roman"] if found else []) + ["CMU Serif", "DejaVu Serif"]})


def plot(R, chance):
    paper_font()
    colors = {0: "#1f7a6d", 1: "#e07b39", 2: "#7a4f9a"}
    mean = R[R.model == "mlp"].groupby(["inputs", "fold"], sort=False).auc.mean()
    names = list(INPUT_SETS)[::-1]
    for lang, L in LABELS.items():
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.axvspan(0.5, chance, color="#d9dcd6", label=L["chance"])
        for k in range(CFG.N_FOLDS):
            ax.scatter([mean[(n, k)] for n in names], np.arange(len(names)) + (k - 1) * 0.2, color=colors[k], s=26,
                       label=f"{L['fold']} {k}", zorder=3)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([L[n] for n in names])
        ax.set_xlim(0.45, 1.02); ax.set_xlabel(L["xlabel"]); ax.set_title(L["title"], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.3)
        if lang == "es":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ",")))
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=4, frameon=False)
        (FIG / lang).mkdir(parents=True, exist_ok=True)
        fit_to_print(fig, "mlp_confounds.png"); fig.savefig(FIG / lang / "mlp_confounds.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / lang / "mlp_confounds.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}); plt.close(fig)


if __name__ == "__main__":
    main()
