"""
Analysis of the factorial design (sharma_factorial.py): what each confound correction does to the network of
Sharma and Singh (2022), alone and in combination. Child-level test AUC, mean of 5 runs, per fold (folds are reported
separately, never averaged into one number for the paper text).

  results/analysis/conditions.csv   one row per network x fold x combination: flags (0/1), AUC mean and SD
  results/analysis/effects.csv      per network x fold x correction:
                                      alone       = AUC(only this correction) - AUC(none)
                                      leave_out   = AUC(all corrections) - AUC(all but this one)
                                      main        = mean AUC with the correction - mean AUC without it, over every
                                                    combination of the others (factorial main effect)
  results/analysis/permutations.csv full permutation test for 'none' and 'all': true AUC, null AUCs, p
  results/analysis/summary.txt
  figures/<es|en>/sharma_factorial.png   replication, each alone, all but one, all; null band of the permutations
  figures/<es|en>/sharma_effects.png     main effect of each correction, per fold and network
"""
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES, OUT, FIG = HERE / "results" / "sharma", HERE / "results" / "analysis", HERE / "figures"
FLAGS = ("seg", "dur", "gain", "dc", "pol", "noise", "lp4k")
NETS = {"sharma": "1s", "sharmaW": "windows"}
FOLD_C = {0: "#1f7a6d", 1: "#e07b39", 2: "#7a4f9a"}
TXT = {
    "es": {"seg": "silencio (solo la vocal)", "dur": "duración", "gain": "nivel (normalizar al pico)", "dc": "componente continua",
           "pol": "polaridad", "noise": "ruido para todos", "lp4k": "agudos (pasa-bajos 4 kHz)",
           "none": "réplica (nada corregido)", "all": "todas las correcciones", "alone": "solo:",
           "leave": "todas menos:", "1s": "red de Sharma y Singh (1 s)", "windows": "variante con ventanas de 128 ms",
           "auc": "AUC en prueba (24 niños)", "fold": "partición", "null": "azar: percentil 95 de 1000 permutaciones",
           "effect": "efecto principal sobre la AUC", "title_eff": "Cuánto baja la AUC cada corrección (promedio sobre las demás combinaciones)"},
    "en": {"seg": "silence (vowel only)", "dur": "duration", "gain": "level (peak normalization)", "dc": "DC offset",
           "pol": "polarity", "noise": "noise for all", "lp4k": "high frequencies (4 kHz low-pass)",
           "none": "replication (nothing corrected)", "all": "all corrections", "alone": "only:",
           "leave": "all but:", "1s": "Sharma and Singh network (1 s)", "windows": "variant on 128-ms windows",
           "auc": "test AUC (24 children)", "fold": "fold", "null": "chance: 95th percentile of 1000 permutations",
           "effect": "main effect on the AUC", "title_eff": "How much each correction lowers the AUC (mean over the other combinations)"},
}


def load():
    rows, perms = [], []
    for d in sorted(RES.iterdir()):
        f = d / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f)); o = s["options"]
        aucs = [r["test"][o["condition"]]["auc"] for r in s["runs"] if not r["test"][o["condition"]].get("diverged")]
        net = d.name.split("_")[0]
        flags = set() if o["condition"] == "none" else set(o["condition"].split("+"))
        if o.get("permute_labels") is not None:
            variant = "pval" if d.name.endswith("_pval") else "last" if d.name.endswith("_last") else "val"
            perms.append(dict(net=net, fold=o["fold"], condition=o["condition"], variant=variant, seed=o["permute_labels"],
                              auc=float(np.mean(aucs)) if aucs else np.nan))
            continue
        rows.append(dict(net=net, fold=o["fold"], condition=o["condition"], **{k: int(k in flags) for k in FLAGS},
                         auc=float(np.mean(aucs)), sd=float(np.std(aucs)), n_runs=len(aucs)))
    return pd.DataFrame(rows), pd.DataFrame(perms)


def effects(C):
    out = []
    for (net, fold), g in C.groupby(["net", "fold"]):
        flags = [f for f in FLAGS if g[f].any()]                      # windows: no 'dur'
        key = lambda fs: g[np.all(g[flags].values == np.array([int(f in fs) for f in flags]), axis=1)].auc
        none, every = key(set()), key(set(flags))
        for f in flags:
            alone = key({f} | ({"seg"} if f == "dur" else set()))
            loo = key(set(flags) - {f} - ({"dur"} if f == "seg" else set()))
            sub = g[g.seg == 1] if f == "dur" else g                 # dur only exists together with seg
            main = sub[sub[f] == 1].auc.mean() - sub[sub[f] == 0].auc.mean()
            out.append(dict(net=net, fold=fold, correction=f, alone=float(alone.iloc[0] - none.iloc[0]),
                            leave_out=float(every.iloc[0] - loo.iloc[0]), main=float(main)))
    return pd.DataFrame(out)


def permutation_table(C, P):
    out = []
    for (net, fold, cond, variant), g in P.groupby(["net", "fold", "condition", "variant"]):
        true = C[(C.net == net) & (C.fold == fold) & (C.condition == cond)].auc
        null = g.auc.dropna().values
        if len(true) and len(null):
            t = float(true.iloc[0])
            out.append(dict(net=net, fold=fold, condition=cond if cond in ("none", "gain") else "all", variant=variant, true_auc=t,
                            null_spread=float(np.mean(np.abs(null - 0.5))),
                            null_max=float(null.max()), null_median=float(np.median(null)), n=len(null),
                            p=float((1 + (null >= t).sum()) / (1 + len(null)))))
    return pd.DataFrame(out)


def plot_ladder(C, PT, lang):
    """Same rows in both panels (the window network has no 'dur': those rows stay empty)."""
    L = TXT[lang]
    rows = [(L["none"], "none", None)] + [(f"{L['alone']} {L[f]}", "alone", f) for f in FLAGS] \
        + [(f"{L['leave']} {L[f]}", "leave", f) for f in FLAGS] + [(L["all"], "all", None)]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.8), sharey=True)
    for ax, (net, tag) in zip(axes, NETS.items()):
        g0 = C[C.net == net]
        flags = [f for f in FLAGS if g0[f].any()]
        def key(kind, f):
            if f is not None and f not in flags:
                return None
            if kind == "none": return set()
            if kind == "all": return set(flags)
            if kind == "alone": return {f} | ({"seg"} if f == "dur" else set())
            return set(flags) - {f} - ({"dur"} if f == "seg" else set())
        for fold in sorted(g0.fold.unique()):
            g = g0[g0.fold == fold]
            for yy, (_, kind, f) in zip(y, rows):
                k = key(kind, f)
                if k is None:
                    continue
                v = g[np.all(g[flags].values == np.array([int(x in k) for x in flags]), axis=1)].auc.iloc[0]
                ax.scatter(v, yy + (fold - 1) * 0.22, s=22, color=FOLD_C[fold], label=f"{L['fold']} {fold}" if kind == "none" else None, zorder=3)
            # chance: 95th percentile of the null of 1000 permutations x 5 runs (perm_null.py / perm_pvalues.py)
            NP = OUT / "perm_pvalues.csv"
            if NP.exists():
                pv = pd.read_csv(NP)
                for _, r in pv[(pv.net == net) & (pv.fold == fold)].iterrows():
                    yy = y[0] if r.condition == "none" else y[-1]
                    ax.plot([0.5, r.null_p95], [yy + (fold - 1) * 0.22] * 2, color=FOLD_C[fold], alpha=.35, lw=4, solid_capstyle="butt")
        for cut in (1 + len(FLAGS), 1 + 2 * len(FLAGS)):
            ax.axhline(y[cut - 1] - 0.5, color="#bbb", lw=.8)
        ax.axvline(0.5, color="#999", lw=.8, ls=":")
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
        ax.set_title(L[tag], fontsize=10); ax.set_xlabel(L["auc"]); ax.set_xlim(0.2, 1.02)
        ax.grid(axis="x", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
        if lang == "es":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ",")))
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#888", alpha=.5, lw=4)); l.append(L["null"])
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fit_to_print(fig, "sharma_factorial.png", rect=(0, 0.06, 1, 1))
    (FIG / lang).mkdir(parents=True, exist_ok=True); fig.savefig(FIG / lang / "sharma_factorial.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_effects(E, lang):
    """Same rows in both panels (the window network has no 'dur': that row stays empty)."""
    L = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(FLAGS))[::-1]
    for ax, (net, tag) in zip(axes, NETS.items()):
        e = E[E.net == net]
        for fold in sorted(e.fold.unique()):
            for yy, f in zip(y, FLAGS):
                v = e[(e.fold == fold) & (e.correction == f)].main
                if len(v):
                    ax.scatter(v.iloc[0], yy + (fold - 1) * 0.2, s=24, color=FOLD_C[fold],
                               label=f"{L['fold']} {fold}" if f == "gain" else None, zorder=3)
        ax.axvline(0, color="#999", lw=.8)
        ax.set_yticks(y); ax.set_yticklabels([L[f] for f in FLAGS], fontsize=9)
        ax.set_title(L[tag], fontsize=10); ax.set_xlabel(L["effect"]); ax.grid(axis="x", alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
        if lang == "es":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.2f}".replace(".", ",")))
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(L["title_eff"], fontsize=10.5); fit_to_print(fig, "sharma_effects.png")
    fig.savefig(FIG / lang / "sharma_effects.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# Printed size of each figure in the paper (inches; text width of the paper = 6.5 in). Figures are drawn at this size,
# so the text inside them is printed at its real point size (about 9-10 pt) instead of being shrunk by LaTeX.
PRINT_SIZE = {
    "f1_sampling_rate.png": (4.2, 2.4), "f2_durations.png": (6.5, 2.7), "f3_zero_padding.png": (6.5, 2.9),
    "f4_channel_measures.png": (6.5, 5.4), "f5_no_voice_classifiers.png": (5.5, 3.1), "f6_equipment_td.png": (6.0, 2.7),
    "e6_vowels.png": (5.5, 4.2), "f8_snr_compression.png": (6.2, 2.8), "mlp_confounds.png": (5.4, 4.2),
    "sharma_factorial.png": (6.5, 6.4), "sharma_effects.png": (6.5, 3.3),
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


def main():
    paper_font()
    OUT.mkdir(parents=True, exist_ok=True)
    C, P = load()
    C.to_csv(OUT / "conditions.csv", index=False)
    E = effects(C); E.to_csv(OUT / "effects.csv", index=False)
    PT = permutation_table(C, P) if len(P) else pd.DataFrame()
    PT.to_csv(OUT / "permutations.csv", index=False)
    with open(OUT / "summary.txt", "w") as f:
        f.write(f"Factorial design, CNN of Sharma and Singh (2022). Child-level test AUC, mean of 5 runs.\n"
                f"{len(C)} conditions ({C.groupby('net').size().to_dict()}), {len(P)} permutation runs.\n\n")
        for net in NETS:
            g = C[C.net == net]
            f.write(f"== {net}\nreplication (none): " + "  ".join(f"f{k} {g[(g.fold == k) & (g.condition == 'none')].auc.iloc[0]:.3f}" for k in sorted(g.fold.unique())) + "\n")
            full = g[np.all(g[[x for x in FLAGS if g[x].any()]].values == 1, axis=1)]
            f.write("all corrections:    " + "  ".join(f"f{r.fold} {r.auc:.3f}" for r in full.itertuples()) + "\n")
            f.write(E[E.net == net].drop(columns="net").pivot(index="correction", columns="fold").round(3).to_string() + "\n\n")
        if len(PT):
            f.write("Full permutation test (20 label shuffles). variant: val = epoch chosen with the real validation labels "
                    "(same pipeline as the true network; p is computed against this null); pval = validation labels shuffled too; "
                    "last = last epoch. null_spread = mean |AUC - 0.5| of the shuffled networks (pure chance with 24 children: 0.10)\n"
                    + PT.round(3).sort_values(["net", "condition", "variant", "fold"]).to_string(index=False) + "\n")
    for lang in TXT:
        plot_ladder(C, PT, lang); plot_effects(E, lang)
    print(open(OUT / "summary.txt").read())


if __name__ == "__main__":
    main()
