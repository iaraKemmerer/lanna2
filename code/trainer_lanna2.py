from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config_lanna2 import CFG


class Trainer:
    """
    Trains one network on windows and evaluates it per child.

    Training: SGD (momentum 0.9, weight decay 0.01, learning rate of the architecture), cross-entropy on windows,
    CFG.MAX_EPOCHS epochs, no early stopping and no learning-rate schedule. After every epoch the child-level F1
    on the validation children is computed; the weights of the FIRST epoch with the highest validation F1
    (strict >) are kept and restored before testing.

    Child-level output: the softmax probabilities averaged over all windows of the child; SLI if p(SLI) > 0.5.
    """

    def __init__(self, device: str, dataloader_train: DataLoader, dataloader_val: DataLoader):
        self.device = device
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.current_lr = None
        self.loss_fn = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # MODEL AND OPTIMIZER
    # ----------------------------------------------------------------
    def set_learning_rate(self, lr: float):
        """The learning rate of the architecture, set before set_model() so the optimizer is created with it."""
        self.current_lr = lr

    def set_model(self, model: nn.Module):
        """Sets the model and creates a new SGD optimizer with self.current_lr."""
        self.model = model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.current_lr, momentum=CFG.MOMENTUM,
                                   weight_decay=CFG.WEIGHT_DECAY)

    # ----------------------------------------------------------------
    # TRAINING AND EVALUATION LOOPS
    # ----------------------------------------------------------------
    def _train_epoch(self) -> Tuple[float, float]:
        """One pass over the balanced training batches. Returns (mean loss, window-level F1 macro)."""
        self.model.train()
        total_loss, preds, labels = 0.0, [], []
        for x, y, _ in self.dataloader_train:
            x, y = x.to(self.device).unsqueeze(1), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            if torch.isnan(loss) or torch.isinf(loss):       # skipped, but counted in the mean below
                continue
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item(); preds.append(outputs.argmax(1)); labels.append(y)
        f1 = f1_score(torch.cat(labels).cpu(), torch.cat(preds).cpu(), average="macro", zero_division=0) if preds else 0.0
        return total_loss / len(self.dataloader_train), float(f1)

    def predict_children(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(recording indices, mean softmax probabilities (n, 2), labels): one row per recording (= child)."""
        self.model.eval()
        P, Y, I = [], [], []
        with torch.no_grad():
            for x, y, i in dataloader:
                P.append(F.softmax(self.model(x.to(self.device).unsqueeze(1)), dim=1).cpu()); Y.append(y); I.append(i)
        P, Y, I = torch.cat(P), torch.cat(Y), torch.cat(I)
        recs = torch.unique(I)
        return (recs.numpy(), torch.stack([P[I == r].mean(0) for r in recs]).numpy(),
                torch.stack([Y[I == r][0] for r in recs]).numpy())

    def _eval_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Child-level (loss, F1 macro): loss = cross-entropy of the averaged probability."""
        _, p, y = self.predict_children(dataloader)
        loss = float(F.nll_loss(torch.log(torch.tensor(p).clamp_min(1e-7)), torch.tensor(y)))
        return loss, float(f1_score(y, p.argmax(1), average="macro", zero_division=0))

    # ----------------------------------------------------------------
    # MAIN FUNCTION
    # ----------------------------------------------------------------
    def evaluate_model(self, model: nn.Module) -> Dict[str, Any]:
        """Trains `model` for CFG.MAX_EPOCHS epochs, restores the best validation epoch and returns the curves."""
        self.set_model(model)
        hist = {"train_f1": [], "train_loss": [], "val_f1": [], "val_loss": []}
        best_f1, best_state, best_epoch = -float("inf"), None, 0
        for epoch in range(CFG.MAX_EPOCHS):
            train_loss, train_f1 = self._train_epoch()
            val_loss, val_f1 = self._eval_epoch(self.dataloader_val)
            hist["train_f1"].append(train_f1); hist["train_loss"].append(train_loss)
            hist["val_f1"].append(val_f1); hist["val_loss"].append(val_loss)
            if val_f1 > best_f1:                              # strict: the first best epoch wins
                best_f1, best_epoch = val_f1, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:                            # None only if every epoch diverged
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        hist.update(best_epoch=best_epoch, best_val_f1=float(best_f1))
        return hist
