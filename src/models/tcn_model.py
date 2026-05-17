import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.pad:
            out = out[:, :, : -self.pad]
        return out


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)


class TCNClassifier(nn.Module):
    def __init__(self, n_features, num_channels=None, kernel_size=3, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(num_channels):
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        out = self.tcn(x.permute(0, 2, 1))
        out = out[:, :, -1]
        return self.classifier(out).squeeze(-1)
