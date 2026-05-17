import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class DLModelWrapper:
    def __init__(self, model, seq_len=30, lr=0.001, batch_size=256, epochs=50, patience=10):
        self.model = model
        self.seq_len = seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.feature_cols = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_sequences(self, dfs, feature_cols):
        X_list, y_list = [], []
        for df in dfs:
            arr = df[feature_cols].fillna(0).values.astype(np.float32)
            tgt = df["target"].values.astype(np.float32)
            n = len(arr)
            if n < self.seq_len:
                continue
            for i in range(n - self.seq_len + 1):
                X_list.append(arr[i: i + self.seq_len])
                y_list.append(tgt[i + self.seq_len - 1])
        if not X_list:
            n_feat = len(feature_cols)
            return (
                np.empty((0, self.seq_len, n_feat), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def fit(self, train_dfs, val_dfs, feature_cols):
        self.feature_cols = feature_cols
        self.model = self.model.to(self.device)

        X_train, y_train = self._make_sequences(train_dfs, feature_cols)
        X_val, y_val = self._make_sequences(val_dfs, feature_cols)

        if len(X_train) == 0:
            logger.warning("No training sequences available, skipping fit.")
            return

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        has_val = len(X_val) > 0
        if has_val:
            val_X_t = torch.from_numpy(X_val).to(self.device)
            val_y_t = torch.from_numpy(y_val).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(val_X_t), val_y_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"  Early stopping at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                avg_train = train_loss / len(train_loader)
                if has_val:
                    logger.info(f"  Epoch {epoch + 1}/{self.epochs}  train={avg_train:.4f}  val={val_loss:.4f}")
                else:
                    logger.info(f"  Epoch {epoch + 1}/{self.epochs}  train={avg_train:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba_df(self, df, feature_cols):
        """Returns array of probabilities, length = len(df) - seq_len + 1."""
        if len(df) < self.seq_len:
            return np.array([], dtype=np.float32)

        arr = df[feature_cols].fillna(0).values.astype(np.float32)
        sequences = np.array(
            [arr[i: i + self.seq_len] for i in range(len(arr) - self.seq_len + 1)],
            dtype=np.float32,
        )
        X_t = torch.from_numpy(sequences).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).cpu().numpy()
        return probs

    def predict_latest(self, df, feature_cols):
        """Returns the single probability for the most recent window."""
        if len(df) < self.seq_len:
            return float("nan")

        arr = df[feature_cols].fillna(0).values[-self.seq_len:].astype(np.float32)
        X_t = torch.from_numpy(arr[np.newaxis]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(self.model(X_t)).item()
        return prob

    def save(self, path):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "feature_cols": self.feature_cols, "seq_len": self.seq_len},
            Path(path),
        )

    @classmethod
    def load(cls, path, model_obj, seq_len):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device, weights_only=False)
        model_obj.load_state_dict(state["model_state_dict"])
        model_obj = model_obj.to(device)
        wrapper = cls(model=model_obj, seq_len=seq_len)
        wrapper.device = device
        wrapper.feature_cols = state.get("feature_cols")
        return wrapper
