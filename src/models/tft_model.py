import torch
import torch.nn as nn


class TFTClassifier(nn.Module):
    """Simplified Temporal Fusion Transformer for binary classification."""

    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        gated = last * self.gate(last)
        return self.classifier(gated).squeeze(-1)
