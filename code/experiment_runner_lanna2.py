import json
import time
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from config_lanna2 import CFG
from dataset_lanna2 import AudioAugmenter, AudioDataProcessor, BalancedBatchSampler, WindowDataset
from model_lanna2 import CNN, load_architecture
from trainer_lanna2 import Trainer


class ExperimentRunner:
    """
    One experiment of the paper: a fixed network (A0, A1 or A2), one fold, one audio condition, CFG.RUNS runs.

    The network of fold k is trained on the training children of fold k (augmented), the best epoch is chosen with
    its validation children, and it is tested ONLY on the test children of fold k. The trained network can also be
    tested on other audio conditions (also_test; by default the original audio).

    Output folder results/train/<name>/:
      predictions_<condition>.csv   one row per run and test child: run, patient_id, label, p_sli, pred
      summary.json / summary.txt    AUC, F1, sensitivity and specificity per run and their mean
      historial_epocas_I0.json      train/validation F1 and loss per epoch, per run
      robustez_final_I0.png         those curves, one column per run, with the chosen epoch
    """

    def __init__(self, arch: str, fold: int, condition: str, vowel: str = "o", region: str = "vowel",
                 truncate_ms: Optional[float] = None, snr_db: Optional[float] = None, also_test: List[str] = ("original",),
                 augmentation: int = CFG.AUGMENTATION_FACTOR, runs: Optional[int] = None,
                 permute_labels: Optional[int] = None, device: str = CFG.DEVICE, name: Optional[str] = None):
        self.arch_name, self.fold, self.condition, self.vowel = arch, fold, condition, vowel
        self.region, self.truncate_ms = region, truncate_ms
        self.snr_db = CFG.NOISE_SNR_DB if snr_db is None else snr_db
        self.also_test, self.augmentation, self.permute_labels, self.device = list(also_test), augmentation, permute_labels, device
        self.runs = runs or (1 if permute_labels is not None else CFG.RUNS)
        self.arch = load_architecture(arch)
        self.processor = AudioDataProcessor()
        self._setup_paths(name)
        self._prepare_data()

    def _setup_paths(self, name: Optional[str]):
        parts = [self.arch_name, f"f{self.fold}", self.condition, self.vowel if self.vowel != "o" else "",
                 "silence" if self.region == "silence" else "", f"cut{self.truncate_ms:g}" if self.truncate_ms else "",
                 f"snr{self.snr_db:g}" if self.snr_db != CFG.NOISE_SNR_DB else "",
                 f"perm{self.permute_labels}" if self.permute_labels is not None else ""]
        self.name = name or "_".join(p for p in parts if p)
        self.out_dir = CFG.RESULTS_DIR / "train" / self.name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================
    # DATA
    # =============================================================
    def _prepare_data(self):
        """Inputs, partition, augmentation of the training set, optional label permutation, windows and loaders."""
        part = self.processor.partition(self.fold, self.vowel)
        self.test_ids = part["test"]
        conditions = list(dict.fromkeys([self.condition] + self.also_test))
        data = {c: self.processor.build_inputs(c, self.fold, self.vowel, self.region, self.truncate_ms, self.snr_db) for c in conditions}
        D = data[self.condition]
        tr_w, tr_y, tr_id = [D[p][0] for p in part["train"]], [D[p][1] for p in part["train"]], list(part["train"])
        if self.augmentation:
            tr_w, tr_y, tr_id = AudioAugmenter().apply_pitch_augmentation(tr_w, tr_y, tr_id, self.augmentation)
        if self.permute_labels is not None:
            # full permutation test: the real labels of the training children are shuffled among them (all copies and
            # windows of a child share the new label); validation and test keep their real labels
            kids = sorted(set(tr_id)); true = dict(zip(tr_id, tr_y))
            new = dict(zip(kids, np.random.default_rng(self.permute_labels).permutation([true[p] for p in kids])))
            tr_y = [int(new[p]) for p in tr_id]
        W = self.arch["window_samples"]
        self.ds_train = WindowDataset(tr_w, tr_y, W)
        ds_val = WindowDataset([D[p][0] for p in part["val"]], [D[p][1] for p in part["val"]], W)
        self.dl_train = DataLoader(self.ds_train, batch_sampler=BalancedBatchSampler(self.ds_train.y.numpy()), num_workers=CFG.NUM_WORKERS)
        self.dl_val = DataLoader(ds_val, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)
        self.dl_tests = {c: DataLoader(WindowDataset([data[c][p][0] for p in part["test"]], [data[c][p][1] for p in part["test"]], W),
                                       batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS) for c in conditions}
        print(f"{self.name}: {len(tr_w)} training recordings -> {self.ds_train.counts()} windows, {len(self.dl_train)} balanced "
              f"batches per epoch | {len(part['val'])} validation and {len(part['test'])} test children | {self.runs} run(s) on {self.device}", flush=True)

    # =============================================================
    # MAIN FUNCTION
    # =============================================================
    def run_experiment(self):
        trainer = Trainer(self.device, self.dl_train, self.dl_val)
        self.rows = {c: [] for c in self.dl_tests}; self.results, self.histories = [], []
        t0 = time.time()
        for k in range(self.runs):
            seed = CFG.RUN_SEED_BASE + k
            torch.manual_seed(seed); np.random.seed(seed)
            trainer.set_learning_rate(self.arch["learning_rate"])
            hist = trainer.evaluate_model(CNN(self.arch))
            self.histories.append(hist)
            r = {"run": k + 1, "seed": seed, "best_epoch": hist["best_epoch"], "val_f1": hist["best_val_f1"], "test": {}}
            for c, dl in self.dl_tests.items():
                recs, p, y = trainer.predict_children(dl)
                if not np.all(np.isfinite(p)):                 # the run diverged (NaN loss): no usable prediction
                    r["test"][c] = {"diverged": True}; continue
                ps, pred = p[:, 1], (p[:, 1] > 0.5).astype(int)
                for i, ps_i, pr_i, y_i in zip(recs, ps, pred, y):
                    self.rows[c].append({"run": k + 1, "patient_id": self.test_ids[i], "label": int(y_i), "p_sli": round(float(ps_i), 4), "pred": int(pr_i)})
                r["test"][c] = {"auc": float(roc_auc_score(y, ps)), "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                                "sensitivity": float((pred[y == 1] == 1).mean()), "specificity": float((pred[y == 0] == 0).mean())}
            self.results.append(r)
            print(f"  run {k + 1}: best epoch {hist['best_epoch']}, val F1 {hist['best_val_f1']:.3f} | " +
                  " | ".join(f"test {c}: " + ("diverged" if v.get("diverged") else f"AUC {v['auc']:.3f} F1 {v['f1']:.3f}") for c, v in r["test"].items()) +
                  f" | {(time.time() - t0) / 60:.1f} min", flush=True)
        self._save_final_results()

    # =============================================================
    # RESULTS
    # =============================================================
    def _save_final_results(self):
        ok = lambda c: [r["test"][c] for r in self.results if not r["test"][c].get("diverged")]
        mean = {c: {m: float(np.mean([t[m] for t in ok(c)])) for m in ("auc", "f1", "sensitivity", "specificity")} for c in self.dl_tests if ok(c)}
        for c, rows in self.rows.items():
            if rows:                                          # no file if every run diverged (excluded from the statistics)
                pd.DataFrame(rows).to_csv(self.out_dir / f"predictions_{c}.csv", index=False)
        options = {"arch": self.arch_name, "fold": self.fold, "condition": self.condition, "vowel": self.vowel, "region": self.region,
                   "truncate_ms": self.truncate_ms, "snr_db": self.snr_db, "augmentation": self.augmentation,
                   "permute_labels": self.permute_labels, "runs": self.runs, "epochs": CFG.MAX_EPOCHS}
        json.dump({"options": options, "runs": self.results, "mean": mean}, open(self.out_dir / "summary.json", "w"), indent=1)
        json.dump(self.histories, open(self.out_dir / "historial_epocas_I0.json", "w"))
        with open(self.out_dir / "summary.txt", "w") as f:
            f.write(f"{self.name}: network {self.arch_name}, fold {self.fold}, condition {self.condition}, vowel /{self.vowel}/, "
                    f"{self.runs} run(s), {CFG.MAX_EPOCHS} epochs\n\n{'test condition':16s} {'AUC':>6s} {'F1':>6s} {'sens':>6s} {'spec':>6s}   AUC per run\n")
            for c, m in mean.items():
                f.write(f"{c:16s} {m['auc']:6.3f} {m['f1']:6.3f} {m['sensitivity']:6.3f} {m['specificity']:6.3f}   "
                        + " ".join(f"{t['auc']:.3f}" for t in ok(c)) + "\n")
        self._plot_training_curves()
        print("mean: " + " | ".join(f"{c}: AUC {m['auc']:.3f}" for c, m in mean.items()), flush=True)

    def _plot_training_curves(self):
        """robustez_final_I0.png: F1 and loss per epoch (train per window, validation per child), one column per run."""
        n = len(self.histories)
        fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 6.4), sharex=True, sharey="row", squeeze=False)
        for k, (h, r) in enumerate(zip(self.histories, self.results)):
            ep = np.arange(1, len(h["train_f1"]) + 1)
            for row, m in enumerate(("f1", "loss")):
                ax = axs[row, k]
                ax.plot(ep, h[f"train_{m}"], "--", color="#2a78d6", lw=1.2, label="train (per window)")
                ax.plot(ep, h[f"val_{m}"], "-", color="#eb6834", lw=1.5, label="validation (per child)")
                ax.axvline(r["best_epoch"], color="#55544e", lw=0.9, ls=":", label="chosen epoch")
                ax.grid(True, alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
                if k == 0: ax.set_ylabel("F1" if m == "f1" else "Loss")
                if row == 1: ax.set_xlabel("Epoch")
            t = " · ".join(f"{c} {v['auc']:.3f}" for c, v in r["test"].items() if not v.get("diverged"))
            axs[0, k].set_title(f"Run {k + 1} · epoch {r['best_epoch']}\nTEST AUC {t}\nval F1 {r['val_f1']:.3f}", fontsize=9)
        axs[0, 0].legend(fontsize=8, loc="lower right")
        fig.suptitle(self.name, fontsize=11); fig.tight_layout()
        fig.savefig(self.out_dir / "robustez_final_I0.png", dpi=120); plt.close(fig)
