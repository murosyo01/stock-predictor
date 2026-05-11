"""
LightGBM モデル学習スクリプト
全銘柄データを結合して時系列分割で学習し、MLflow に記録する
"""
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

EXCLUDE_COLS = {"ticker", "target", "future_return", "open", "high", "low", "close", "volume"}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def time_split(
    df: pd.DataFrame,
    test_ratio: float,
    purge_gap: int,
    validation_ratio: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_ratio = test_ratio if validation_ratio is None else validation_ratio
    dates = pd.Index(df.index).drop_duplicates().sort_values()
    n_dates = len(dates)
    test_start = int(n_dates * (1 - test_ratio))
    val_start = int(test_start * (1 - validation_ratio))
    train_end = max(0, val_start - purge_gap)
    val_end = max(val_start, test_start - purge_gap)

    if train_end == 0 or val_end <= val_start or test_start >= n_dates:
        raise ValueError(
            "Not enough dates to create train/validation/test splits with "
            f"test_ratio={test_ratio}, validation_ratio={validation_ratio}, "
            f"purge_gap={purge_gap}."
        )

    train_dates = dates[:train_end]
    val_dates = dates[val_start:val_end]
    test_dates = dates[test_start:]
    return (
        df.loc[df.index.isin(train_dates)],
        df.loc[df.index.isin(val_dates)],
        df.loc[df.index.isin(test_dates)],
    )


def main() -> None:
    cfg = load_config()
    feat_dir = Path(cfg["data"]["features_dir"])
    model_dir = Path(cfg["model"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for ticker in cfg["tickers"]:
        path = feat_dir / f"{ticker}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError("特徴量ファイルが見つかりません。features.py を先に実行してください。")

    all_data = pd.concat(dfs).sort_index()
    all_data = all_data.dropna(subset=["future_return"])
    logger.info(f"Total training data: {len(all_data)} rows, tickers: {len(dfs)}")

    feature_cols = get_feature_cols(all_data)
    purge_gap = cfg["features"]["target_horizon"]
    validation_ratio = cfg["model"].get("validation_ratio", cfg["model"]["test_ratio"])
    train_df, val_df, test_df = time_split(
        all_data,
        cfg["model"]["test_ratio"],
        purge_gap=purge_gap,
        validation_ratio=validation_ratio,
    )

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"]

    mc = cfg["model"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        params = {
            "n_estimators": mc["n_estimators"],
            "learning_rate": mc["learning_rate"],
            "num_leaves": mc["num_leaves"],
            "min_child_samples": mc["min_child_samples"],
            "subsample": mc["subsample"],
            "colsample_bytree": mc["colsample_bytree"],
            "random_state": mc["random_state"],
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
        }
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "validation_ratio": validation_ratio,
                "purge_gap": purge_gap,
            }
        )

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "train_size": len(X_train),
            "validation_size": len(X_val),
            "test_size": len(X_test),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        logger.info("=== Evaluation Metrics ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        mlflow.lightgbm.log_model(model, "model")

        model_path = model_dir / "lgbm_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

        fi = pd.Series(
            model.feature_importances_, index=feature_cols, name="importance"
        ).sort_values(ascending=False)
        fi.to_csv(model_dir / "feature_importance.csv")

        logger.info(f"Model saved → {model_path}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
