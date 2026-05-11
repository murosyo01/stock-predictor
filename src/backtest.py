"""
バックテストスクリプト
テスト期間でモデルシグナルに基づく戦略を評価し、指標を CSV に保存する
"""
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

CONFIG_PATH = ROOT / "config" / "config.yaml"
EXCLUDE_COLS = {"ticker", "target", "future_return", "open", "high", "low", "close", "volume"}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(cumulative: pd.Series) -> float:
    rolling_max = cumulative.cummax()
    dd = (cumulative - rolling_max) / rolling_max
    return float(dd.min())


def run_backtest(test_df: pd.DataFrame, commission: float) -> tuple[dict, pd.DataFrame]:
    df = test_df.copy().sort_index()
    df["signal"] = (df["signal_prob"] >= 0.5).astype(int)

    df["strategy_return"] = (
        df["signal"].shift(1) * df["return_1d"]
        - commission * df["signal"].diff().abs().fillna(0)
    )
    df = df.dropna(subset=["strategy_return", "return_1d"])

    strat_cum = (1 + df["strategy_return"]).cumprod()
    bh_cum = (1 + df["return_1d"]).cumprod()

    n_signals = int(df["signal"].sum())
    precision = float(df.loc[df["signal"] == 1, "target"].mean()) if n_signals > 0 else 0.0

    metrics = {
        "accuracy": float((df["signal"] == df["target"]).mean()),
        "precision": precision,
        "sharpe": sharpe_ratio(df["strategy_return"]),
        "max_drawdown": max_drawdown(strat_cum),
        "total_return": float(strat_cum.iloc[-1] - 1),
        "buy_hold_return": float(bh_cum.iloc[-1] - 1),
        "n_trades": n_signals,
    }
    curves = df[["strategy_return", "return_1d"]].copy()
    curves["strat_cum"] = strat_cum
    curves["bh_cum"] = bh_cum
    return metrics, curves


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def get_dl_probs_for_test(
    wrapper, df: pd.DataFrame, feature_cols: list[str], seq_len: int, test_df: pd.DataFrame
) -> pd.Series:
    """DL モデルの全期間確率を計算し、テスト期間にアライン済みの Series を返す。"""
    probs = wrapper.predict_proba_df(df, feature_cols)
    if len(probs) == 0:
        return pd.Series(dtype=float, index=test_df.index)
    # predict_proba_df の出力は df.index[seq_len-1:] に対応する
    prob_series = pd.Series(probs, index=df.index[seq_len - 1:])
    return prob_series.reindex(test_df.index)


def load_dl_wrappers(cfg: dict, model_dir: Path, n_features: int) -> dict:
    """利用可能な DL モデルをロードして {name: wrapper} の辞書で返す。"""
    try:
        from models.dl_wrapper import DLModelWrapper
        from models.lstm_model import LSTMClassifier
        from models.tcn_model import TCNClassifier
        from models.patchtst_model import PatchTSTClassifier
        from models.tft_model import TFTClassifier
    except ImportError:
        logger.warning("DL モデルのインポートに失敗。DL モデルをスキップします。")
        return {}

    seq_len = cfg["dl_model"]["seq_len"]
    lc = cfg["dl_model"]["lstm"]
    tc = cfg["dl_model"]["tcn"]
    pc = cfg["dl_model"]["patchtst"]
    tfc = cfg["dl_model"]["tft"]

    candidates = {
        "lstm": LSTMClassifier(
            n_features, hidden_size=lc["hidden_size"],
            num_layers=lc["num_layers"], dropout=lc["dropout"],
        ),
        "tcn": TCNClassifier(
            n_features, num_channels=tc["num_channels"],
            kernel_size=tc["kernel_size"], dropout=tc["dropout"],
        ),
        "patchtst": PatchTSTClassifier(
            n_features, seq_len=seq_len, patch_len=pc["patch_len"],
            stride=pc["stride"], d_model=pc["d_model"],
            n_heads=pc["n_heads"], n_layers=pc["n_layers"], dropout=pc["dropout"],
        ),
        "tft": TFTClassifier(
            n_features, d_model=tfc["d_model"], n_heads=tfc["n_heads"],
            n_layers=tfc["n_layers"], dropout=tfc["dropout"],
        ),
    }

    wrappers = {}
    for name, model_obj in candidates.items():
        path = model_dir / f"{name}_model.pt"
        if not path.exists():
            continue
        try:
            from models.dl_wrapper import DLModelWrapper
            wrapper = DLModelWrapper.load(path, model_obj, seq_len=seq_len)
            wrappers[name] = wrapper
            logger.info(f"{name.upper()} loaded")
        except Exception as e:
            logger.warning(f"{name} のロードに失敗: {e}")
    return wrappers


def main() -> None:
    cfg = load_config()
    feat_dir = ROOT / cfg["data"]["features_dir"]
    model_dir = ROOT / cfg["model"]["model_dir"]

    if not (model_dir / "lgbm_model.pkl").exists():
        raise FileNotFoundError("モデルが見つかりません。train.py を先に実行してください。")

    with open(model_dir / "lgbm_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    lgbm_model = bundle["model"]
    lgbm_feature_cols = bundle["feature_cols"]

    test_ratio = cfg["model"]["test_ratio"]
    commission = cfg["backtest"]["commission"]
    seq_len = cfg["dl_model"]["seq_len"]

    # DL モデルのロードに必要な n_features を先に取得
    n_features = 0
    for t in cfg["tickers"]:
        p = feat_dir / f"{t}.parquet"
        if p.exists():
            n_features = len(get_feature_cols(pd.read_parquet(p)))
            break

    dl_wrappers = load_dl_wrappers(cfg, model_dir, n_features) if n_features > 0 else {}

    lgbm_results = []    # 既存の backtest_results.csv 用（後方互換）
    by_model_results = []  # 新規の backtest_by_model.csv 用

    for ticker in cfg["tickers"]:
        path = feat_dir / f"{ticker}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path).sort_index()
        n = len(df)
        test_df = df.iloc[int(n * (1 - test_ratio)):].copy()

        # --- LightGBM ---
        X_test = test_df[lgbm_feature_cols].fillna(0)
        test_df["signal_prob"] = lgbm_model.predict_proba(X_test)[:, 1]
        metrics, _ = run_backtest(test_df, commission)
        metrics["ticker"] = ticker
        lgbm_results.append(metrics)
        by_model_results.append({"model": "lgbm", "ticker": ticker,
                                  **{k: v for k, v in metrics.items() if k != "ticker"}})
        logger.info(
            f"lgbm       {ticker}: acc={metrics['accuracy']:.3f}  "
            f"sharpe={metrics['sharpe']:.3f}  ret={metrics['total_return']:.3f}  "
            f"bh={metrics['buy_hold_return']:.3f}"
        )

        # --- DL モデル ---
        feature_cols = get_feature_cols(df)
        for name, wrapper in dl_wrappers.items():
            dl_cols = wrapper.feature_cols if wrapper.feature_cols else feature_cols
            prob_series = get_dl_probs_for_test(wrapper, df, dl_cols, seq_len, test_df)

            dl_test_df = test_df.copy()
            dl_test_df["signal_prob"] = prob_series.values
            dl_test_df = dl_test_df.dropna(subset=["signal_prob"])

            if dl_test_df.empty:
                logger.warning(f"{name} {ticker}: テスト期間の確率が空。スキップ。")
                continue

            dl_metrics, _ = run_backtest(dl_test_df, commission)
            by_model_results.append({"model": name, "ticker": ticker, **dl_metrics})
            logger.info(
                f"{name:<10} {ticker}: acc={dl_metrics['accuracy']:.3f}  "
                f"sharpe={dl_metrics['sharpe']:.3f}  ret={dl_metrics['total_return']:.3f}"
            )

    # 既存フォーマット（lgbm 専用、後方互換）
    out_path = ROOT / "data" / "backtest_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df = pd.DataFrame(lgbm_results)[
        ["ticker", "accuracy", "precision", "sharpe", "max_drawdown",
         "total_return", "buy_hold_return", "n_trades"]
    ]
    results_df.to_csv(out_path, index=False)
    logger.info(f"Backtest results saved → {out_path}")

    # モデル横断バックテスト結果
    if by_model_results:
        by_model_df = pd.DataFrame(by_model_results)[
            ["model", "ticker", "accuracy", "precision", "sharpe",
             "max_drawdown", "total_return", "buy_hold_return", "n_trades"]
        ]
        by_model_path = ROOT / "data" / "backtest_by_model.csv"
        by_model_df.to_csv(by_model_path, index=False)
        logger.info(f"Per-model backtest results saved → {by_model_path}")

        summary = (
            by_model_df
            .groupby("model")[["accuracy", "precision", "sharpe", "max_drawdown", "total_return"]]
            .mean()
            .sort_values("sharpe", ascending=False)
        )
        print("\n=== Model Comparison (averaged across tickers) ===")
        print(summary.to_string(float_format="{:.4f}".format))

    print("\n=== LightGBM Backtest Summary (per ticker) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
