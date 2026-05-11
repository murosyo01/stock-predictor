"""
特徴量生成スクリプト
テクニカル指標と教師ラベルを生成して Parquet に保存する
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def generate_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fc = cfg["features"]
    close = df["close"]

    for w in fc["ma_windows"]:
        df[f"ma_{w}"] = close.rolling(w).mean()
        df[f"ma_{w}_ratio"] = close / df[f"ma_{w}"]

    windows = fc["ma_windows"]
    if 5 in windows and 25 in windows:
        df["ma_5_25_cross"] = (df["ma_5"] > df["ma_25"]).astype(int)
    if 25 in windows and 60 in windows:
        df["ma_25_60_cross"] = (df["ma_25"] > df["ma_60"]).astype(int)

    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)

    vol_w = fc["volatility_window"]
    df[f"volatility_{vol_w}"] = df["return_1d"].rolling(vol_w).std()

    rsi_w = fc["rsi_window"]
    df[f"rsi_{rsi_w}"] = compute_rsi(close, rsi_w)

    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        close, fc["macd_fast"], fc["macd_slow"], fc["macd_signal"]
    )
    df["macd_norm"] = df["macd"] / close.replace(0, np.nan)

    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)

    df["hl_range"] = (df["high"] - df["low"]) / close.replace(0, np.nan)

    horizon = fc["target_horizon"]
    future_return = close.shift(-horizon) / close - 1
    df["future_return"] = future_return
    df["target"] = np.where(
        future_return.notna(),
        (future_return > 0).astype(float),
        np.nan,
    )

    return df.dropna(subset=["future_return", "target"])


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    feat_dir = Path(cfg["data"]["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 市場インデックスを読み込んでクロスアセット特徴量として使う
    idx_series: dict[str, pd.Series] = {}
    for idx_ticker in cfg.get("market_indices", []):
        safe = idx_ticker.replace("^", "_").replace("=", "_")
        path = raw_dir / f"{safe}.parquet"
        if not path.exists():
            continue
        idx_df = pd.read_parquet(path).sort_index()
        col = f"idx_{safe}_return"
        idx_series[col] = idx_df["close"].pct_change().rename(col)

    for ticker in cfg["tickers"]:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(f"Raw data not found: {path}")
            continue

        df = pd.read_parquet(path).sort_index()

        for col, series in idx_series.items():
            df[col] = series.reindex(df.index)

        df = generate_features(df, cfg)
        label_cols = {"target", "future_return"}
        feat_cols = [c for c in df.columns if c not in label_cols]
        df = df.dropna(subset=feat_cols)

        out_path = feat_dir / f"{ticker}.parquet"
        df.to_parquet(out_path, index=True)
        logger.info(f"Features saved: {ticker} ({len(df)} rows, {len(df.columns)} cols)")

    logger.info("Feature generation complete.")


if __name__ == "__main__":
    main()
