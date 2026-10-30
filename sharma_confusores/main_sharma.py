"""
The CNN of Sharma and Singh (2022) on one fold, with one set of confound corrections (sharma_lanna2.py).

Training follows Sharma where it can: SGD (momentum 0.9, learning rate 0.01, L2 1e-4), 30 epochs, Glorot
initialization, circular-shift augmentation (10 copies per child). What changes, so that the result is comparable
with the rest of the paper: our subject-disjoint folds (48 train / 24 validation / 24 test children), the epoch
chosen with the validation children (Sharma has no validation set), balanced batches of 5 TD + 5 SLI, 5 runs and the
child-level AUC. Output in results/sharma/<name>/ with the same files as main_lanna2.py.

Examples
--------
    python main_sharma.py -f 0 -c none                  # replication: Sharma's input as it is
    python main_sharma.py -f 0 -c seg+dur               # the vowel only, repeated to 1 s (no duration)
    python main_sharma.py -f 0 -c seg+dur+gain+dc+pol+noise+lp4k   # every confound removed
    python main_sharma.py -f 0 -c none --permute-labels 3          # one network of the full permutation test
    python main_sharma.py -f 0 -c seg+gain --window 2048           # MODIFIED network on 128-ms windows

With --window W the modified network (SharmaWindowCNN) receives windows of W samples with 50% overlap, only up to
the last nonzero sample of the input (the zero padding is never seen), in balanced batches of 16 TD + 16 SLI windows,
and the child-level output is the mean over its windows. Windows hide duration, so "dur" is not used; no
circular-shift copies are made (the windows already give many training examples).
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config_lanna2 import CFG
from dataset_lanna2 import BalancedBatchSampler, WindowDataset
from experiment_runner_lanna2 import ExperimentRunner
from trainer_lanna2 import Trainer
from sharma_lanna2 import SharmaCNN, SharmaWindowCNN, build_inputs, circular_shift_copies, parse_flags

LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 30
COPIES = 10
BATCH_SIZE = 10


class LastEpochTrainer(Trainer):
    """Trainer that keeps the weights of the LAST epoch (validation is still computed, only for the curves)."""

    def evaluate_model(self, model):
        self.set_model(model)
        hist = {"train_f1": [], "train_loss": [], "val_f1": [], "val_loss": []}
        for _ in range(CFG.MAX_EPOCHS):
            tl, tf = self._train_epoch(); vl, vf = self._eval_epoch(self.dataloader_val)
            hist["train_f1"].append(tf); hist["train_loss"].append(tl); hist["val_f1"].append(vf); hist["val_loss"].append(vl)
        hist.update(best_epoch=CFG.MAX_EPOCHS, best_val_f1=float(hist["val_f1"][-1]))
        return hist


class SharmaRunner(ExperimentRunner):
    """ExperimentRunner with Sharma's network and inputs: one 1-s input per child instead of windows."""

    def __init__(self, fold, spec, vowel="o", runs=None, permute_labels=None, snr_db=None, copies=COPIES,
                 window=None, device=CFG.DEVICE, name=None, permute_val=False, epoch_choice="val"):
        self.flags, self.window = parse_flags(spec), window
        self.permute_val, self.epoch_choice = permute_val, epoch_choice
        if window and "dur" in self.flags:
            raise ValueError("with --window the duration is already hidden: do not use 'dur'")
        if window:
            copies = 0
        self.arch_name, self.fold, self.vowel, self.region, self.truncate_ms = "Sharma", fold, vowel, "vowel", None
        self.condition = "+".join(self.flags) or "none"
        self.snr_db = CFG.NOISE_SNR_DB if snr_db is None else snr_db
        self.also_test, self.augmentation, self.permute_labels, self.device = [], copies, permute_labels, device
        self.runs = runs or (1 if permute_labels is not None else CFG.RUNS)
        self.arch = {"learning_rate": LEARNING_RATE}
        from dataset_lanna2 import AudioDataProcessor
        self.processor = AudioDataProcessor()
        name = name or "_".join(p for p in ["sharmaW" if window else "sharma", f"f{fold}", self.condition, vowel if vowel != "o" else "",
                                            f"perm{permute_labels}" if permute_labels is not None else "",
                                            "pval" if permute_val else "", "last" if epoch_choice == "last" else ""] if p)
        self.name, self.out_dir = name, CFG.RESULTS_DIR / "sharma" / name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_data()

    def build_model(self):
        return SharmaWindowCNN() if self.window else SharmaCNN()

    def run_experiment(self):
        if self.epoch_choice == "last":                       # control: no epoch selection, the last epoch is kept
            import experiment_runner_lanna2 as er
            er.Trainer = LastEpochTrainer
        super().run_experiment()

    def _prepare_data(self):
        part = self.processor.partition(self.fold, self.vowel)
        self.test_ids = part["test"]
        D = build_inputs(self.flags, self.fold, self.vowel, self.snr_db)
        y_true = {p: D[p][1] for p in part["train"]}
        if self.permute_labels is not None:                  # shuffled among the training children, copies included
            kids = list(part["train"])
            y_true = dict(zip(kids, np.random.default_rng(self.permute_labels).permutation([y_true[p] for p in kids])))
        rng = np.random.default_rng(CFG.AUGMENTATION_SEED)
        X, Y = [], []
        for p in part["train"]:
            for x in [D[p][0]] + circular_shift_copies(D[p][0], self.augmentation, rng):
                X.append(x); Y.append(int(y_true[p]))
        if self.window:
            ds = lambda xs, ys: WindowDataset(xs, ys, self.window)
            self.ds_train = ds(X, Y)
            self.dl_train = DataLoader(self.ds_train, batch_sampler=BalancedBatchSampler(self.ds_train.y.numpy()))
        else:
            ds = lambda xs, ys: TensorDataset(torch.tensor(np.stack(xs)), torch.tensor(ys, dtype=torch.int64), torch.arange(len(ys)))
            self.ds_train = ds(X, Y)
            self.dl_train = DataLoader(self.ds_train, batch_sampler=BalancedBatchSampler(np.array(Y), BATCH_SIZE))
        y_val = [D[p][1] for p in part["val"]]
        if self.permute_val:                                 # control: the validation labels are shuffled too
            y_val = list(np.random.default_rng(10_000 + self.permute_labels).permutation(y_val))
        self.dl_val = DataLoader(ds([D[p][0] for p in part["val"]], y_val), batch_size=32)
        self.dl_tests = {self.condition: DataLoader(ds([D[p][0] for p in part["test"]], [D[p][1] for p in part["test"]]), batch_size=32)}
        print(f"{self.name}: {len(Y)} training inputs ({sum(Y)} SLI), {len(self.dl_train)} balanced batches per epoch | "
              f"{len(part['val'])} validation and {len(part['test'])} test children | {self.runs} run(s) on {self.device}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--fold", type=int, required=True, choices=range(CFG.N_FOLDS))
    ap.add_argument("-c", "--corrections", default="none", help="'none' or flags joined by '+': seg dur gain dc pol noise lp4k")
    ap.add_argument("-v", "--vowel", default="o", choices=list(CFG.VOWELS))
    ap.add_argument("-r", "--runs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--copies", type=int, default=COPIES, help="circular-shift copies per training child")
    ap.add_argument("--permute-labels", type=int, default=None)
    ap.add_argument("--window", type=int, default=None, help="window length in samples: modified network on windows")
    ap.add_argument("--permute-val", action="store_true", help="with --permute-labels: shuffle the validation labels too")
    ap.add_argument("--epoch-choice", default="val", choices=["val", "last"], help="best validation epoch, or the last one")
    ap.add_argument("--device", default=CFG.DEVICE)
    args = ap.parse_args()
    CFG.MAX_EPOCHS, CFG.WEIGHT_DECAY = args.epochs, WEIGHT_DECAY
    SharmaRunner(args.fold, args.corrections, args.vowel, args.runs, args.permute_labels, copies=args.copies,
                 window=args.window, device=args.device, permute_val=args.permute_val,
                 epoch_choice=args.epoch_choice).run_experiment()


if __name__ == "__main__":
    main()
