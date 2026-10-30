"""Shared figure style and figure language (English / Spanish).

Language: environment variable FIG_LANG = en (default) | es | all.
  * every visible text goes through T("english text"), which returns the Spanish version from ES when the language is es;
  * numbers written as text go through N(x, spec), which uses a decimal comma in Spanish;
  * save(fig, name) writes CFG.FIGURES_DIR / <language> / name (and, in Spanish, puts decimal commas on numeric axes).
With FIG_LANG=all the scripts compute everything once and draw the figures in both languages (for _ in each_lang(): ...).
Only texts and output paths depend on the language; no computation does.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from config_lanna2 import CFG

TD_C, SLI_C, GRIS, DICT = "#2f6690", "#b5542d", "#8a949a", "#e0b04a"
FOLD_C = {0: "#1f7a6d", 1: "#e07b39", 2: "#7a4f9a"}
FORM_NAME = {"original": "original", "dc": "DC only", "pol": "polarity only", "dcpol": "DC + polarity",
             "noise_for_all": "noise for all", "donor_v3": "donor (v3)", "adaptive_v4": "adaptive (v4)"}

LANGS = ("en", "es")
_requested = os.environ.get("FIG_LANG", "en").strip().lower()
if _requested not in LANGS + ("all",):
    sys.exit(f"FIG_LANG must be en, es or all (got {_requested!r})")
FIG_LANG = "en" if _requested == "all" else _requested
_missing = set()

# Spanish texts. TD and SLI stay as they are in both languages.
ES = {
    # condition names (FORM_NAME)
    "original": "original", "DC only": "solo continua", "polarity only": "solo polaridad", "DC + polarity": "continua + polaridad",
    "noise for all": "ruido para todos", "donor (v3)": "donante (v3)", "adaptive (v4)": "adaptativa (v4)",
    # audit.py: f1
    "16 kHz (dictaphone)": "16 kHz (dictáfono)", "44.1 kHz": "44,1 kHz",
    "{n16} at 16 kHz · {n44} at 44.1 kHz": "{n16} a 16 kHz · {n44} a 44,1 kHz",
    "children (vowel /o/)": "niños (vocal /o/)", "Original sampling rate of each recording": "Frecuencia de muestreo original de cada grabación",
    # f2
    "whole file": "archivo completo", "vowel (.lbl segment)": "vocal (segmento .lbl)", "silence (file − vowel)": "silencio (archivo − vocal)",
    "seconds": "segundos", "{t}\nAUC alone = {auc}": "{t}\nAUC sola = {auc}",
    "Duration per group (one point per child): SLI vowels last almost twice as long":
        "Duración por grupo (un punto por niño): las vocales SLI duran casi el doble",
    # f3
    "{p}% zeros": "{p} % de ceros", "time (s) · input of 16,000 samples, as in Sharma and Singh (2022)":
        "tiempo (s) · entrada de 16.000 muestras, como en Sharma y Singh (2022)",
    "% of the input that is zeros": "% de la entrada que son ceros", "children": "niños",
    "Counting zeros separates: AUC = {auc}": "Contar ceros separa: AUC = {auc}",
    "Whole file zero-padded to 1 s (gray = padding): the padding length reveals the class":
        "Archivo completo rellenado con ceros hasta 1 s (gris = relleno): la longitud del relleno revela la clase",
    # f4, f6, e6
    "background noise (dBFS)": "ruido de fondo (dBFS)", "SNR (dB)": "SNR (dB)", "DC offset (×1000)": "componente continua (×1000)",
    "vowel skewness (polarity)": "asimetría de la vocal (polaridad)", "vowel energy > 4 kHz (%)": "energía de la vocal > 4 kHz (%)",
    "noise relative to peak (dB)": "ruido relativo al pico (dB)", "{t}\nAUC = {A} (separates {S})": "{t}\nAUC = {A} (separa {S})",
    "Recording measures per group in the original audio (one point per child; vowel /o/)":
        "Medidas de la grabación por grupo en el audio original (un punto por niño; vocal /o/)",
    "Every SNR is compressed\ntowards that of the added noise": "Toda SNR se comprime\nhacia la del ruido agregado",
    "SNR with noise (dB)": "SNR con ruido (dB)",
    "Difference of medians:\n{a} dB → {b} dB": "Diferencia de medianas:\n{a} dB → {b} dB",
    "TD 16 kHz": "TD 16 kHz", "TD 44.1 kHz": "TD 44,1 kHz", "TD\n16 kHz": "TD\n16 kHz", "TD\n44.1 kHz": "TD\n44,1 kHz", "{t}\nequipment AUC = {A}": "{t}\nAUC del equipo = {A}",
    "TD children only: the device leaves its own footprint (no diagnosis involved)":
        "Solo niños TD: el equipo deja su propia huella (el diagnóstico no interviene)",
    "sampling rate": "frecuencia de muestreo", "file duration": "duración del archivo", "vowel duration": "duración de la vocal",
    "silence duration": "duración del silencio", "SNR": "SNR", "DC offset": "componente continua",
    "skewness (polarity)": "asimetría (polaridad)", "energy > 4 kHz": "energía > 4 kHz", "noise relative to peak": "ruido relativo al pico",
    "silence only (logistic LOO)": "solo silencio (logística LOO)",
    "TD/SLI separation (AUC, folded to ≥ 0.5)": "separación TD/SLI (AUC, plegada a $\\geq$ 0,5)",
    "The same shortcuts in both vowels": "Los mismos atajos en ambas vocales",
    # f5
    "% zeros (padding to 1 s)": "% de ceros (relleno a 1 s)", "sampling rate only": "solo frecuencia de muestreo",
    "silence only (noise, 4 kHz, 50 Hz, DC)": "solo silencio (ruido, 4 kHz, 50 Hz, continua)",
    "silence at network scale (rel. to peak)": "silencio a escala de la red (rel. al pico)",
    "chance with 97 children (95%, up to {az})": "azar con 97 niños (95 %, hasta {az})",
    "AUC (single measure: direct · several: logistic regression, leave-one-child-out)":
        "AUC (una medida: directa · varias: regresión logística, dejando un niño afuera)",
    "Classifiers that never hear the voice": "Clasificadores que nunca escuchan la voz",
    # signal_figures.py: f7, f8
    "original (vowel peak = 1)": "original (pico de la vocal = 1)", "noise for all (vowel peak = 1)": "ruido para todos (pico de la vocal = 1)",
    "time (s)": "tiempo (s)", "amplitude": "amplitud", "frequency (Hz)": "frecuencia (Hz)", "energy (dB)": "energía (dB)",
    "{g} · vowel spectrum": "{g} · espectro de la vocal",
    "Two typical /o/ recordings before and after noise for all (shaded = vowel seen by the network; noise added 5 dB below the vowel)":
        "Dos grabaciones típicas de /o/ antes y después de ruido para todos (sombreado = vocal que ve la red; ruido agregado 5 dB por debajo de la vocal)",
    "noise added {sa} dB below the voice": "ruido {sa} dB bajo la voz", "nothing added (identity)": "sin agregar nada (identidad)",
    "original SNR (dB)": "SNR original (dB)", "SNR after adding noise (dB)": "SNR después de agregar ruido (dB)",
    "Every SNR is compressed towards that of the added noise": "Toda SNR se comprime hacia la del ruido agregado",
    "{g} original": "{g} original", "{g} noise for all": "{g} ruido para todos",
    "Difference of medians: {a} dB → {b} dB": "Diferencia de medianas: {a} dB → {b} dB",
    "Noise for all, in numbers: each device's own noise ends up under the added noise":
        "Ruido para todos, en números: el ruido propio de cada equipo queda debajo del ruido agregado",
    # praat.py: f9, e8
    "DC +\npolarity": "continua +\npolaridad", "noise\nfor all": "ruido\npara todos", "donor\n(v3)": "donante\n(v3)",
    "local jitter (%)": "jitter local (%)", "local shimmer (%)": "shimmer local (%)", "HNR (dB)": "HNR (dB)",
    "Clinical voice measures (Praat) per condition · top: separation between TD and SLI":
        "Medidas clínicas de la voz (Praat) por condición · arriba: separación entre TD y SLI",
    "no noise": "sin ruido", "total SNR": "SNR total", "HNR (whole vowel)": "HNR (vocal completa)", "HNR (first 150 ms)": "HNR (primeros 150 ms)",
    "added noise (dB below the vowel)": "ruido agregado (dB por debajo de la vocal)",
    "SLI − TD difference (dB, medians)": "diferencia SLI − TD (dB, medianas)", "AUC": "AUC",
    "TD/SLI separation (AUC)": "separación TD/SLI (AUC)",
    "Is the HNR gap voice or channel? If channel, it vanishes when the SNR gap does":
        "¿La brecha de HNR es voz o canal? Si es canal, desaparece cuando desaparece la brecha de SNR",
    # detectors.py: f10
    "Q1: TD or SLI?  (43 TD, 54 SLI)": "P1: ¿TD o SLI?  (43 TD, 54 SLI)",
    "Q2: within TD, 16 kHz or 44.1 kHz?  (15 / 28)": "P2: dentro de TD, ¿16 kHz o 44,1 kHz?  (15 / 28)",
    "chance (95%)": "azar (95 %)", "channel detector (6 measures + skewness)": "detector de canal (6 medidas + asimetría)",
    "spectrum detector (24 vowel bands)": "detector de espectro (24 bandas de la vocal)",
    "detector AUC (0.5 = cannot tell; 1 = perfect)": "AUC del detector (0,5 = no distingue; 1 = perfecto)",
    # stats.py: f11, f12, f13, e9, e3a
    "chance (95%, up to ≈{az})": "azar (95 %, hasta ≈{az})", "fold {k}": "partición {k}", "Fold {k}": "Partición {k}",
    "child-level AUC on unseen test children (0.5 = chance · 1 = perfect)":
        "AUC por niño en niños de prueba no vistos (0,5 = azar · 1 = perfecto)",
    "false positive rate (TD called SLI)": "tasa de falsos positivos (TD clasificado como SLI)",
    "AUC (mean of 5 runs)": "AUC (media de 5 corridas)", "true positive rate (SLI detected)": "tasa de verdaderos positivos (SLI detectado)",
    "Noise for all (color) vs original audio (gray, dashed)": "Ruido para todos (color) vs. audio original (gris, discontinua)",
    "fold {k}: true AUC {A} · p = {p}": "partición {k}: AUC real {A} · p = {p}", "test AUC": "AUC de prueba",
    "retrainings with\nshuffled labels": "reentrenamientos con\netiquetas permutadas",
    "Full permutation test (noise for all)\ngray bars = retrained with training labels shuffled among children · line = true labels":
        "Test de permutación completo (ruido para todos)\nbarras grises = reentrenada con las etiquetas de entrenamiento permutadas entre niños · línea = etiquetas reales",
    "AUC (mean of 5 runs) with 95% bootstrap CI over children": "AUC (media de 5 corridas) con IC 95 % por bootstrap sobre niños",
    "* = differs from original (DeLong, Holm-corrected p < 0.05)": "* = difiere del original (DeLong, p corregida por Holm < 0,05)",
    "fold {k} · {form}": "partición {k} · {form}", "vowel duration (s)": "duración de la vocal (s)",
    "p(SLI) of the child (mean of 5 runs)": "p(SLI) del niño (media de 5 corridas)",
    "Does the network still use duration? Correlation within each class (≈ 0 if not)":
        "¿La red sigue usando la duración? Correlación dentro de cada clase (≈ 0 si no)",
}


def set_lang(lang):
    global FIG_LANG
    FIG_LANG = lang


def each_lang():
    """Languages to draw: FIG_LANG=all gives both (the figure code runs once per language, on the same results)."""
    for lang in (LANGS if _requested == "all" else (FIG_LANG,)):
        set_lang(lang)
        yield lang


def T(text):
    """Visible text in the current figure language (the English text is the key)."""
    if FIG_LANG == "en":
        return text
    if text not in ES:
        if text not in _missing:
            _missing.add(text); print(f"[plotting_lanna2] no Spanish text for: {text!r}", file=sys.stderr)
        return text
    return ES[text]


def N(x, spec=".2f"):
    """A number written as text: decimal comma (and dot for thousands) in Spanish."""
    s = format(x, spec)
    return s.translate(str.maketrans(".,", ",.")) if FIG_LANG == "es" else s


class _CommaFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


# Printed size of each figure in the paper (inches; text width of the paper = 6.5 in). Figures are drawn at this size,
# so the text inside them is printed at its real point size (about 9-10 pt) instead of being shrunk by LaTeX.
PRINT_SIZE = {
    "f1_sampling_rate.png": (4.2, 2.4), "f2_durations.png": (6.5, 2.7), "f3_zero_padding.png": (6.5, 2.9),
    "f4_channel_measures.png": (6.5, 5.4), "f5_no_voice_classifiers.png": (5.5, 3.1), "f6_equipment_td.png": (6.0, 2.7),
    "e6_vowels.png": (5.5, 4.2), "f8_snr_compression.png": (6.5, 3.5), "mlp_confounds.png": (5.4, 4.2),
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


def save(fig, name):
    """Save into CFG.FIGURES_DIR / <language> / name; in Spanish, numeric axes use a decimal comma."""
    if FIG_LANG == "es":
        for a in fig.axes:
            for axis in (a.xaxis, a.yaxis):
                if type(axis.get_major_formatter()) is ScalarFormatter:
                    axis.set_major_formatter(_CommaFormatter())
    fit_to_print(fig, name, getattr(fig, "_print_rect", None))
    out = CFG.FIGURES_DIR / FIG_LANG; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=300)


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


def style():
    paper_font()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})


def box_points(ax, groups, labels, colors, seed=0):
    """Box plot plus one jittered point per child."""
    rng = np.random.default_rng(seed)
    for i, (v, c) in enumerate(zip(groups, colors)):
        ax.boxplot(v, positions=[i], widths=.5, showfliers=False, medianprops=dict(color="k"),
                   boxprops=dict(color=GRIS), whiskerprops=dict(color=GRIS), capprops=dict(color=GRIS))
        ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=11, color=c, alpha=.7, zorder=3)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
