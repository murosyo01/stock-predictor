"""
深層学習モデル (LSTM / TCN / PatchTST / TFT) の学習スクリプト。
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

# src/ をパスに追加して models パッケージをインポートできるようにする
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from models.dl_wrapper import DLModelWrapper
from models.lstm_model import LSTMClassifier
from models.patchtst_model import PatchTSTClassifier
from models.tcn_model import TCNClassifier
from models.tft_model import TFTClassifier

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

EXCLUDE_COLS = {"ticker", "target", "future_return", "open", "high", "low", "close", "volume"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_and_split(cfg: dict) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """各銘柄ファイルを読み込み、時系列順を保証したうえで train/val 分割して返す。"""
    features_dir = Path(cfg["data"]["features_dir"])
    test_ratio = cfg["model"]["test_ratio"]

    train_dfs, val_dfs = [], []
    for ticker in cfg["tickers"]:
        path = features_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(f"{path} が見つかりません。スキップします。")
            continue
        df = pd.read_parquet(path)
        # DatetimeIndex があれば昇順ソートで時系列順を保証
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        df = df.reset_index(drop=True)
        split = int(len(df) * (1 - test_ratio))
        train_dfs.append(df.iloc[:split].copy())
        val_dfs.append(df.iloc[split:].copy())

    return train_dfs, val_dfs


def get_feature_cols(train_dfs: list[pd.DataFrame]) -> list[str]:
    """全銘柄の train データから共通の特徴量列を取得する。"""
    if not train_dfs:
        raise ValueError("train_dfs が空です。特徴量ファイルを先に生成してください。")
    col_sets = [set(df.columns) - EXCLUDE_COLS for df in train_dfs]
    common: set[str] = col_sets[0]
    for s in col_sets[1:]:
        common &= s
    if not common:
        raise ValueError("全銘柄に共通する特徴量列が存在しません。")
    return sorted(common)


def evaluate(wrapper: DLModelWrapper, val_dfs: list[pd.DataFrame], feature_cols: list[str], seq_len: int) -> None:
    """val データで accuracy と roc_auc を計算してログ出力する。"""
    all_probs, all_labels = [], []
    for df in val_dfs:
        probs = wrapper.predict_proba_df(df, feature_cols)
        if len(probs) == 0:
            continue
        labels = df["target"].values[seq_len - 1:]
        all_probs.append(probs)
        all_labels.append(labels)

    if not all_probs:
        logger.warning("評価できるデータがありません。")
        return

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    logger.info(f"  val accuracy={acc:.4f}  roc_auc={auc:.4f}")


def build_model(name: str, n_features: int, cfg_dl: dict):
    """モデル名と設定から PyTorch モデルを生成する。"""
    if name == "lstm":
        c = cfg_dl["lstm"]
        return LSTMClassifier(
            n_features=n_features,
            hidden_size=c["hidden_size"],
            num_layers=c["num_layers"],
            dropout=c["dropout"],
        ), c["lr"]

    elif name == "tcn":
        c = cfg_dl["tcn"]
        return TCNClassifier(
            n_features=n_features,
            num_channels=c["num_channels"],
            kernel_size=c["kernel_size"],
            dropout=c["dropout"],
        ), c["lr"]

    elif name == "patchtst":
        c = cfg_dl["patchtst"]
        return PatchTSTClassifier(
            n_features=n_features,
            seq_len=cfg_dl["seq_len"],
            patch_len=c["patch_len"],
            stride=c["stride"],
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            n_layers=c["n_layers"],
            dropout=c["dropout"],
        ), c["lr"]

    elif name == "tft":
        c = cfg_dl["tft"]
        return TFTClassifier(
            n_features=n_features,
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            n_layers=c["n_layers"],
            dropout=c["dropout"],
        ), c["lr"]

    else:
        raise ValueError(f"未知のモデル名: {name}")


def main() -> None:
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = load_config()
    cfg_dl = cfg["dl_model"]
    model_dir = Path(cfg["model"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("データを読み込んでいます...")
    train_dfs, val_dfs = load_and_split(cfg)
    if not train_dfs:
        logger.error("有効な銘柄データがありません。終了します。")
        return

    feature_cols = get_feature_cols(train_dfs)
    n_features = len(feature_cols)
    seq_len = cfg_dl["seq_len"]
    logger.info(f"特徴量数: {n_features}  seq_len: {seq_len}")

    model_names = ["lstm", "tcn", "patchtst", "tft"]

    for name in model_names:
        logger.info(f"===== {name.upper()} 学習開始 =====")

        model, lr = build_model(name, n_features, cfg_dl)

        wrapper = DLModelWrapper(
            model=model,
            seq_len=seq_len,
            lr=lr,
            batch_size=cfg_dl["batch_size"],
            epochs=cfg_dl["epochs"],
            patience=cfg_dl["patience"],
        )

        wrapper.fit(train_dfs, val_dfs, feature_cols)

        logger.info(f"  評価中...")
        evaluate(wrapper, val_dfs, feature_cols, seq_len)

        save_path = model_dir / f"{name}_model.pt"
        wrapper.save(save_path)
        logger.info(f"  保存: {save_path}")

    logger.info("全モデルの学習が完了しました。")


if __name__ == "__main__":
    main()
