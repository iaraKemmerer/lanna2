import json

import torch
import torch.nn as nn

from config_lanna2 import CFG


class CauchyActivation(nn.Module):
    """f(x) = (lambda1 * x + lambda2) / (x^2 + d^2), with three learnable scalars initialised to 1."""

    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x2_d2 = x ** 2 + self.d ** 2
        return self.lambda1 * x / x2_d2 + self.lambda2 / x2_d2


class MLP(nn.Module):
    """Linear -> BatchNorm1d -> Cauchy -> Dropout for each hidden layer, then Linear to the classes (logits)."""

    def __init__(self, layers, dropout_rate):
        super().__init__()
        seq = []
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seq += [nn.BatchNorm1d(layers[i + 1]), CauchyActivation(), nn.Dropout(p=dropout_rate)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    """
    The 1-D CNN of the paper, built from a fixed architecture file (architectures/<name>.json).

        window (B, 1, W)
        -> per conv layer: Conv1d(10 filters, kernel k, stride s, padding p on each side) -> BatchNorm1d
                           -> Cauchy (all but the last conv layer) -> Dropout1d
        -> column pooling: at each time step, [min, mean, max] over the 10 channels: (B, 10, L) -> (B, 3, L)
        -> flatten -> MLP -> 2 logits (TD, SLI)

    The construction order (conv layers, a dummy forward pass in training mode to size the MLP, MLP, Xavier
    initialisation) is kept on purpose: it fixes how the random generator is consumed, so that with the same seed
    the initial weights are identical to the ones of the experiments of the paper.
    """

    def __init__(self, arch: dict):
        super().__init__()
        W = arch["window_samples"]
        layers, in_channels = [], 1
        for i, layer in enumerate(arch["conv_layers"]):
            layers.append(nn.Conv1d(in_channels, CFG.FILTERS, layer["kernel"], layer["stride"], layer["padding"]))
            layers.append(nn.BatchNorm1d(CFG.FILTERS))
            if i < len(arch["conv_layers"]) - 1:           # the last conv layer goes straight to the pooling
                layers.append(CauchyActivation())
            layers.append(nn.Dropout1d(p=arch["conv_dropout"]))
            in_channels = CFG.FILTERS
        self.cnn_block = nn.Sequential(*layers)

        # MLP input size from a dummy forward pass (the module is still in training mode, as in the paper)
        with torch.no_grad():
            mlp_input_size = self._mlp_input(torch.zeros(1, 1, W)).size(1)
        self.mlp_block = MLP([mlp_input_size] + arch["mlp_hidden"] + [CFG.NUM_CLASSES], arch["mlp_dropout"])
        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform weights and zero biases for every Conv1d and Linear layer."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def column_pooling(x):
        """(B, C, L) -> (B, 3, L): minimum, mean and maximum over the channels, time step by time step."""
        return torch.stack((x.amin(dim=1), x.mean(dim=1), x.amax(dim=1)), dim=1)

    def _mlp_input(self, x):
        z = self.column_pooling(self.cnn_block(x))
        return z.view(z.size(0), -1)

    def forward(self, x):
        return self.mlp_block(self._mlp_input(x))


def load_architecture(name: str) -> dict:
    return json.load(open(CFG.ARCHITECTURES_DIR / f"{name}.json"))
