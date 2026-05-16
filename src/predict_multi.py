"""
マルチモデル予測スクリプト
LightGBM / LSTM / TCN / PatchTST / TFT の5モデルでアンサンブル予測を行い CSV に保存する。
"""
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

CONFIG_PATH = ROOT / "config" / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_COLS = {"ticker", "target", "future_return", "open", "high", "low", "close", "volume"}

BACKTEST_PATH = ROOT / "data" / "backtest_by_model.csv"
REQUIRED_BACKTEST_COLS = {"model", "ticker", "accuracy", "precision", "sharpe", "max_drawdown"}


def load_model_weights(model_names: list[str]) -> dict[str, float]:
    """バックテスト結果からモデルの重みを計算して返す。ファイル不在・列不足時は等重み。"""
    equal = {m: 1.0 / len(model_names) for m in model_names}

    if not BACKTEST_PATH.exists():
        logger.warning("Backtest results not found, using equal weights")
        return equal

    try:
        bt = pd.read_csv(BACKTEST_PATH)
    except Exception as e:
        logger.warning(f"Failed to read backtest file: {e}, using equal weights")
        return equal

    if not REQUIRED_BACKTEST_COLS.issubset(bt.columns):
        missing = REQUIRED_BACKTEST_COLS - set(bt.columns)
        logger.warning(f"Backtest file missing columns {missing}, using equal weights")
        return equal

    scores = (
        bt.groupby("model")[["sharpe", "accuracy", "precision", "max_drawdown"]]
        .mean()
    )
    scores["reliability_score"] = (
        scores["sharpe"] * 0.4
        + scores["accuracy"] * 0.3
        + scores["precision"] * 0.2
        - scores["max_drawdown"].abs() * 0.1
    )

    weights: dict[str, float] = {}
    for m in model_names:
        if m in scores.index:
            weights[m] = max(float(scores.loc[m, "reliability_score"]), 0.01)
        else:
            weights[m] = 0.01

    total = sum(weights.values())
    weights = {m: w / total for m, w in weights.items()}
    return weights


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def load_lgbm(model_dir: Path):
    path = model_dir / "lgbm_model.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("LightGBM loaded")
    return bundle["model"], bundle["feature_cols"]


def load_dl_model(name: str, model_dir: Path, model_obj, seq_len: int):
    from models.dl_wrapper import DLModelWrapper

    path = model_dir / f"{name}_model.pt"
    if not path.exists():
        return None
    wrapper = DLModelWrapper.load(path, model_obj, seq_len=seq_len)
    logger.info(f"{name.upper()} loaded")
    return wrapper


def build_dl_models(cfg: dict, n_features: int) -> dict:
    from models.lstm_model import LSTMClassifier
    from models.tcn_model import TCNClassifier
    from models.patchtst_model import PatchTSTClassifier
    from models.tft_model import TFTClassifier

    seq_len = cfg["dl_model"]["seq_len"]
    model_dir = ROOT / cfg["model"]["model_dir"]

    lc = cfg["dl_model"]["lstm"]
    tc = cfg["dl_model"]["tcn"]
    pc = cfg["dl_model"]["patchtst"]
    tfc = cfg["dl_model"]["tft"]

    candidates = {
        "lstm": LSTMClassifier(
            n_features,
            hidden_size=lc["hidden_size"],
            num_layers=lc["num_layers"],
            dropout=lc["dropout"],
        ),
        "tcn": TCNClassifier(
            n_features,
            num_channels=tc["num_channels"],
            kernel_size=tc["kernel_size"],
            dropout=tc["dropout"],
        ),
        "patchtst": PatchTSTClassifier(
            n_features,
            seq_len=seq_len,
            patch_len=pc["patch_len"],
            stride=pc["stride"],
            d_model=pc["d_model"],
            n_heads=pc["n_heads"],
            n_layers=pc["n_layers"],
            dropout=pc["dropout"],
        ),
        "tft": TFTClassifier(
            n_features,
            d_model=tfc["d_model"],
            n_heads=tfc["n_heads"],
            n_layers=tfc["n_layers"],
            dropout=tfc["dropout"],
        ),
    }

    wrappers = {}
    for name, model_obj in candidates.items():
        wrapper = load_dl_model(name, model_dir, model_obj, seq_len)
        if wrapper is not None:
            wrappers[name] = wrapper
    return wrappers


def main() -> None:
    cfg = load_config()
    feat_dir = ROOT / cfg["data"]["features_dir"]
    pred_dir = ROOT / cfg["data"]["predictions_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / cfg["model"]["model_dir"]

    # --- LightGBM ---
    lgbm_model, lgbm_feature_cols = load_lgbm(model_dir)

    # 特徴量列と n_features を最初に見つかった銘柄ファイルから決定する
    sample_df = None
    for t in cfg["tickers"]:
        p = feat_dir / f"{t}.parquet"
        if p.exists():
            sample_df = pd.read_parquet(p)
            break
    if sample_df is None:
        logger.error("特徴量ファイルが1つも見つかりません。features.py を先に実行してください。")
        return
    feature_cols = get_feature_cols(sample_df)
    n_features = len(feature_cols)

    # --- DL モデル ---
    dl_wrappers = build_dl_models(cfg, n_features)

    # モデル名リストを確定し重みを計算する
    model_names = []
    if lgbm_model is not None:
        model_names.append("lgbm")
    model_names.extend(dl_wrappers.keys())

    weights = load_model_weights(model_names)
    logger.info("Using weights: " + " ".join(f"{m}={weights[m]:.4f}" for m in model_names))

    results = []
    for ticker in cfg["tickers"]:
        path = feat_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(f"{ticker}: feature file not found, skipped")
            continue

        df = pd.read_parquet(path).sort_index()
        row: dict = {
            "ticker": ticker,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "close": round(float(df["close"].iloc[-1]), 2),
        }

        probs = []
        prob_weights = []

        if lgbm_model is not None:
            X = df.tail(1)[lgbm_feature_cols].fillna(0)
            prob = float(lgbm_model.predict_proba(X)[0, 1])
            row["lgbm_prob"] = round(prob, 4)
            probs.append(prob)
            prob_weights.append(weights["lgbm"])

        import math
        for name, wrapper in dl_wrappers.items():
            dl_cols = wrapper.feature_cols if wrapper.feature_cols else feature_cols
            prob = wrapper.predict_latest(df, dl_cols)
            if math.isfinite(prob):
                row[f"{name}_prob"] = round(prob, 4)
                probs.append(prob)
                prob_weights.append(weights[name])
            else:
                row[f"{name}_prob"] = None

        if probs:
            w_sum = sum(prob_weights)
            ensemble = sum(p * w for p, w in zip(probs, prob_weights)) / w_sum
        else:
            ensemble = float("nan")
        row["ensemble_prob"] = round(ensemble, 4)
        row["signal"] = "BUY" if ensemble >= 0.5 else "HOLD"

        results.append(row)
        logger.info(
            f"{ticker}: ensemble={ensemble:.4f}  signal={row['signal']}"
        )

    pred_df = pd.DataFrame(results).sort_values("ensemble_prob", ascending=False)

    today = datetime.today().strftime("%Y%m%d")
    out_path = pred_dir / f"predictions_multi_{today}.csv"
    pred_df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}")

    print("\n=== Multi-Model Predictions ===")
    print(pred_df.to_string(index=False))


if __name__ == "__main__":
    main()
