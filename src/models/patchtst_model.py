import torch
import torch.nn as nn


class PatchTSTClassifier(nn.Module):
    def __init__(
        self, n_features, seq_len=30, patch_len=8, stride=4,
        d_model=128, n_heads=8, n_layers=3, dropout=0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)

        self.patch_embed = nn.Linear(n_features * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        patches = []
        for i in range(self.n_patches):
            s = i * self.stride
            patches.append(x[:, s: s + self.patch_len, :].reshape(x.size(0), -1))
        patches = torch.stack(patches, dim=1)  # (B, n_patches, feat*patch_len)

        emb = self.dropout(self.patch_embed(patches) + self.pos_embed)
        out = self.transformer(emb)
        out = self.norm(out.mean(dim=1))
        return self.classifier(out).squeeze(-1)
