"""Part III statistics of the paper, computed from the predictions written by train.py (no training here).

For each fold (network A_k evaluated only on the test children of fold k) and each audio condition:
  * child-level AUC of every run and their mean (Table "Child-level AUC")
  * 95% CI of the mean AUC by bootstrap over children (2000 class-stratified resamples)
  * paired DeLong test against the original audio (score = p(SLI) averaged over the 5 runs), Holm-corrected per fold
  * cheap permutation test: test labels shuffled among children 10,000 times (same shuffle for all runs)
  * E3a: Spearman correlation between p(SLI) and vowel duration within each class (needs results/audit from audit.py)
and, if available:
  * the full permutation test (runs trained with --permute-labels 1..20): p = (1 + #(permuted AUC >= true AUC)) / (N + 1)
  * the controls E2 (silence only), E3b (vowels cut to 150 ms) and E6 (vowel /a/)
Every random generator is seeded from (fold, condition), so the numbers do not depend on which other results exist.

Outputs: results/stats/stats.json, results/stats/table_auc.csv and the figures f11_auc_conditions.png,
f12_roc_noise_for_all.png, f13_full_permutation.png, e9_confidence_intervals.png, e3a_duration.png, in figures/<lang>/
(<lang> = en, es; FIG_LANG=all draws both from one computation).
Usage: python stats.py                (a few minutes, CPU)
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score, roc_curve

from config_lanna2 import CFG
from plotting_lanna2 import FOLD_C, FORM_NAME, GRIS, SLI_C, TD_C, N, T, each_lang, save, style

RESULTS_DIR = CFG.RESULTS_DIR

TRAIN = RESULTS_DIR / "train"
FOLDS = [("A0", 0), ("A1", 1), ("A2", 2)]
FORMS = CFG.CONDITIONS
CONTROLS = {"E2 silence only": "_silence", "E3b vowel cut to 150 ms": "_cut150", "E6 vowel /a/": "_a"}


def load(name, form):
    f = TRAIN / name / f"predictions_{form}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    P = d.pivot(index="patient_id", columns="run", values="p_sli")
    return P, d.groupby("patient_id").label.first().loc[P.index].values


def mean_auc(P, y):
    return float(np.mean([roc_auc_score(y, P[c].values) for c in P.columns]))


def delong(y, s1, s2):
    """Paired DeLong test (fast version of Sun and Xu, 2014): returns the two-sided p-value."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    def comps(s):
        d = s[pos][:, None] - s[neg][None, :]
        psi = (d > 0) + 0.5 * (d == 0)
        return psi.mean(), psi.mean(1), psi.mean(0)
    a1, x1, y1 = comps(s1); a2, x2, y2 = comps(s2)
    S = np.cov(np.vstack([x1, x2])) / len(pos) + np.cov(np.vstack([y1, y2])) / len(neg)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    return float(2 * st.norm.sf(abs((a1 - a2) / np.sqrt(var)))) if var > 0 else 1.0


def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); out[i] = run
    return out


durations = None
if (RESULTS_DIR / "audit" / "measures_o.csv").exists():
    durations = pd.read_csv(RESULTS_DIR / "audit" / "measures_o.csv").set_index("patient_id").vowel_s

res, rows = {}, []
for arch, k in FOLDS:
    base = load(f"{arch}_f{k}_original", "original")
    if base is None:
        continue
    P0, y = base; r_fold, compared = {}, []
    for j, form in enumerate(FORMS):
        got = load(f"{arch}_f{k}_{form}", form)
        if got is None:
            continue
        P, yy = got; P = P.loc[P0.index]
        A = mean_auc(P, y)
        rng = np.random.default_rng([k, j, 1]); i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; boot = []
        for _ in range(CFG.BOOTSTRAP_SAMPLES):
            idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
            boot.append(np.mean([roc_auc_score(y[idx], P[c].values[idx]) for c in P.columns]))
        rng = np.random.default_rng([k, j, 2])
        null = np.array([np.mean([roc_auc_score(yp, P[c].values) for c in P.columns]) for yp in (rng.permutation(y) for _ in range(CFG.TEST_LABEL_PERMUTATIONS))])
        r = {"auc": A, "auc_runs": [roc_auc_score(y, P[c].values) for c in P.columns], "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
             "cheap_perm_p": float((1 + np.sum(null >= A)) / (CFG.TEST_LABEL_PERMUTATIONS + 1)), "chance95": float(np.quantile(null, .95))}
        if form != "original":
            r["delong_p"] = delong(y, P0.mean(axis=1).values, P.mean(axis=1).values); compared.append(form)
        if durations is not None:
            d = durations.loc[P.index].values; pm = P.mean(axis=1).values
            r["duration_within_class"] = {g: dict(zip(("rho", "p"), map(float, st.spearmanr(pm[y == c], d[y == c])))) for c, g in ((0, "TD"), (1, "SLI"))}
            r["duration_alone_test_auc"] = float(max(roc_auc_score(y, d), 1 - roc_auc_score(y, d)))
        r_fold[form] = r
    for form, p in zip(compared, holm([r_fold[f]["delong_p"] for f in compared])):
        r_fold[form]["delong_p_holm"] = float(p)
    perms = sorted(glob.glob(str(TRAIN / f"{arch}_f{k}_noise_for_all_perm*" / "predictions_noise_for_all.csv")))
    if perms and "noise_for_all" in r_fold:
        pa = []
        for f in perms:
            d = pd.read_csv(f); pa.append(float(roc_auc_score(d.label, d.p_sli)))
        obs = r_fold["noise_for_all"]["auc"]
        r_fold["full_permutation"] = {"true_auc": obs, "permuted_auc": pa, "n": len(pa), "p": float((1 + np.sum(np.array(pa) >= obs)) / (len(pa) + 1))}
    for cname, suf in CONTROLS.items():
        for form in ("original", "noise_for_all"):
            got = load(f"{arch}_f{k}_{form}{suf}", form)
            if got is not None:
                P, yy = got; rng = np.random.default_rng([k, 99, len(suf), FORMS.index(form)]); A = mean_auc(P, yy)
                null = np.array([np.mean([roc_auc_score(yp, P[c].values) for c in P.columns]) for yp in (rng.permutation(yy) for _ in range(CFG.TEST_LABEL_PERMUTATIONS))])
                r_fold.setdefault("controls", {}).setdefault(cname, {})[form] = {"auc": A, "cheap_perm_p": float((1 + np.sum(null >= A)) / (CFG.TEST_LABEL_PERMUTATIONS + 1))}
    res[f"fold{k}"] = r_fold
    for form, r in r_fold.items():
        if form in FORMS:
            rows.append({"fold": k, "network": arch, "condition": form, "auc": round(r["auc"], 3), "ci_low": round(r["ci95"][0], 2), "ci_high": round(r["ci95"][1], 2),
                         "cheap_perm_p": round(r["cheap_perm_p"], 4), "delong_p_holm": round(r.get("delong_p_holm", np.nan), 3)})
out = RESULTS_DIR / "stats"; out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "stats.json", "w"), indent=1)
table = pd.DataFrame(rows); table.to_csv(out / "table_auc.csv", index=False)
print(table.to_string(index=False))
for f, r in res.items():
    if "full_permutation" in r: print(f, "full permutation:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["full_permutation"].items() if k != "permuted_auc"})
    if "controls" in r: print(f, "controls:", json.dumps(r["controls"]))

# ------------------------------------------------------------------ figures
style()
folds = [(a, k) for a, k in FOLDS if f"fold{k}" in res]
F = [f for f in FORMS if f != "adaptive_v4" and all(f in res[f"fold{k}"] for _, k in folds)]
pf = [(a, k) for a, k in folds if "full_permutation" in res[f"fold{k}"]]
F9 = [f for f in FORMS if all(f in res[f"fold{k}"] for _, k in folds)]
for _ in each_lang():
    fig, a = plt.subplots(figsize=(9, 4.2)); yb = np.arange(len(F))[::-1]
    az = np.mean([res[f"fold{k}"][f]["chance95"] for _, k in folds for f in F])
    a.axvspan(1 - az, az, color="#d9d9d9", alpha=.7, lw=0, label=T("chance (95%, up to ≈{az})").format(az=N(az)))
    for j, (arch, k) in enumerate(folds):
        off = (j - (len(folds) - 1) / 2) * .22
        for i, f in enumerate(F):
            r = res[f"fold{k}"][f]
            a.scatter(r["auc"], yb[i] + off, s=80, color=FOLD_C[k] if r["cheap_perm_p"] < .05 else "white", edgecolor=FOLD_C[k], lw=2, zorder=3, label=T("fold {k}").format(k=k) if i == 0 else None)
            a.text(r["auc"] + .012, yb[i] + off, N(r['auc']), va="center", fontsize=8.5, color="#333")
    a.axvline(.5, c="k", lw=.8, ls=":"); a.set_yticks(yb); a.set_yticklabels([T(FORM_NAME[f]) for f in F]); a.set_xlim(.3, 1.02)
    a.set_xlabel(T("child-level AUC on unseen test children (0.5 = chance · 1 = perfect)"))
    a.legend(loc="lower center", bbox_to_anchor=(.5, -.36), ncol=4, frameon=False, fontsize=9)
    save(fig, "f11_auc_conditions.png"); plt.close(fig)

    fig, ax = plt.subplots(1, len(folds), figsize=(4.3 * len(folds), 4.3), squeeze=False)
    for a, (arch, k) in zip(ax[0], folds):
        a.plot([0, 1], [0, 1], ls=":", c="k", lw=.8)
        for form, lab, cl, fill in (("original", "original", GRIS, False), ("noise_for_all", "noise for all", "#1f7a6d", True)):
            P, yy = load(f"{arch}_f{k}_{form}", form)
            aucs = {c: roc_auc_score(yy, P[c].values) for c in P.columns}; m = float(np.mean(list(aucs.values())))
            c = min(aucs, key=lambda q: abs(aucs[q] - m)); fp, tp, _ = roc_curve(yy, P[c].values)   # the run closest to the mean
            a.plot(fp, tp, color=cl, lw=2.2 if fill else 1.6, ls="-" if fill else "--", label=f"{T(lab)}: {N(m)}")
            if fill: a.fill_between(fp, tp, color=cl, alpha=.15)
        a.set_title(T("Fold {k}").format(k=k), fontsize=10); a.set_xlabel(T("false positive rate (TD called SLI)")); a.set_aspect("equal"); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        a.legend(loc="lower right", fontsize=8.5, frameon=False, title=T("AUC (mean of 5 runs)"), title_fontsize=8)
    ax[0, 0].set_ylabel(T("true positive rate (SLI detected)"))
    fig.suptitle(T("Noise for all (color) vs original audio (gray, dashed)"), fontsize=10.5)
    fig.tight_layout(); save(fig, "f12_roc_noise_for_all.png"); plt.close(fig)

    if pf:
        fig, ax = plt.subplots(1, len(pf), figsize=(5 * len(pf), 3.6), sharey=True, squeeze=False)
        for a, (arch, k) in zip(ax[0], pf):
            r = res[f"fold{k}"]["full_permutation"]
            a.hist(r["permuted_auc"], bins=np.linspace(.2, 1, 17), color="#bdbdbd", edgecolor="white")
            a.axvline(r["true_auc"], color=FOLD_C[k], lw=3)
            a.set_title(T("fold {k}: true AUC {A} · p = {p}").format(k=k, A=N(r['true_auc']), p=N(r['p'], '.3f')), fontsize=10)
            a.set_xlabel(T("test AUC")); a.set_xlim(.2, 1)
        ax[0, 0].set_ylabel(T("retrainings with\nshuffled labels"))
        fig.suptitle(T("Full permutation test (noise for all)\ngray bars = retrained with training labels shuffled among children · line = true labels"), fontsize=10)
        fig.tight_layout(); save(fig, "f13_full_permutation.png"); plt.close(fig)

    fig, a = plt.subplots(figsize=(9, 4.6)); yb = np.arange(len(F9))[::-1]
    for j, (arch, k) in enumerate(folds):
        off = (j - (len(folds) - 1) / 2) * .25
        for i, f in enumerate(F9):
            r = res[f"fold{k}"][f]
            a.plot(r["ci95"], [yb[i] + off] * 2, color=FOLD_C[k], lw=2); a.scatter(r["auc"], yb[i] + off, color=FOLD_C[k], s=40, zorder=3, label=T("fold {k}").format(k=k) if i == 0 else None)
            if r.get("delong_p_holm", 1) < .05: a.text(r["ci95"][1] + .01, yb[i] + off, "*", va="center", fontsize=12, color=FOLD_C[k])
    a.axvline(.5, c="k", lw=.8, ls=":"); a.set_yticks(yb); a.set_yticklabels([T(FORM_NAME[f]) for f in F9]); a.set_xlim(.3, 1.05)
    a.set_xlabel(T("AUC (mean of 5 runs) with 95% bootstrap CI over children"))
    a.set_title(T("* = differs from original (DeLong, Holm-corrected p < 0.05)"), fontsize=9.5); a.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "e9_confidence_intervals.png"); plt.close(fig)

    if durations is not None:
        fig, ax = plt.subplots(len(folds), 2, figsize=(10, 3.5 * len(folds)), sharex=True, sharey=True, squeeze=False)
        for r_, (arch, k) in enumerate(folds):
            for c_, form in enumerate(("original", "noise_for_all")):
                P, yy = load(f"{arch}_f{k}_{form}", form); d = durations.loc[P.index].values; a = ax[r_, c_]
                for cls, cl, g in ((0, TD_C, "TD"), (1, SLI_C, "SLI")):
                    m = yy == cls; rr = res[f"fold{k}"][form]["duration_within_class"][g]
                    a.scatter(d[m], P.mean(axis=1).values[m], color=cl, s=30, label=f"{g}: ρ = {N(rr['rho'])} (p = {N(rr['p'])})")
                a.axhline(.5, c="k", lw=.6, ls=":"); a.legend(frameon=False, fontsize=8.5, loc="upper left")
                a.set_title(T("fold {k} · {form}").format(k=k, form=T(FORM_NAME[form])), fontsize=10)
                if r_ == len(folds) - 1: a.set_xlabel(T("vowel duration (s)"))
                if c_ == 0: a.set_ylabel(T("p(SLI) of the child (mean of 5 runs)"))
        fig.suptitle(T("Does the network still use duration? Correlation within each class (≈ 0 if not)"), fontsize=10.5)
        fig.tight_layout(); save(fig, "e3a_duration.png"); plt.close(fig)
