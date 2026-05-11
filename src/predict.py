"""
予測スクリプト
学習済みモデルを使って各銘柄の上昇確率を予測し CSV に保存する
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    feat_dir = Path(cfg["data"]["features_dir"])
    pred_dir = Path(cfg["data"]["predictions_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg["model"]["model_dir"]) / "lgbm_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError("モデルが見つかりません。train.py を先に実行してください。")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    results = []
    for ticker in cfg["tickers"]:
        path = feat_dir / f"{ticker}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path).sort_index()
        latest = df.tail(1)
        X = latest[feature_cols].fillna(0)
        prob = model.predict_proba(X)[0, 1]
        signal = int(prob >= 0.5)

        results.append({
            "ticker": ticker,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "close": round(float(df["close"].iloc[-1]), 2),
            "up_probability": round(prob, 4),
            "signal": "BUY" if signal else "HOLD",
        })
        logger.info(f"{ticker}: prob={prob:.4f}  signal={'BUY' if signal else 'HOLD'}")

    pred_df = pd.DataFrame(results).sort_values("up_probability", ascending=False)
    today = datetime.today().strftime("%Y%m%d")
    out_path = pred_dir / f"predictions_{today}.csv"
    pred_df.to_csv(out_path, index=False)
    logger.info(f"Predictions saved → {out_path}")

    print("\n=== Today's Predictions ===")
    print(pred_df.to_string(index=False))


if __name__ == "__main__":
    main()
