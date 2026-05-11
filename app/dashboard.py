"""
Streamlit ダッシュボード
銘柄選択・価格チャート・予測シグナル・バックテスト・特徴量重要度を表示する
"""
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from models.lstm_model import LSTMClassifier
    from models.tcn_model import TCNClassifier
    from models.patchtst_model import PatchTSTClassifier
    from models.tft_model import TFTClassifier
    from models.dl_wrapper import DLModelWrapper
    _DL_IMPORTS_OK = True
except Exception:
    _DL_IMPORTS_OK = False

from backtest import run_backtest  # noqa: E402

DL_MODEL_NAMES = ["lstm", "tcn", "patchtst", "tft"]
DL_MODEL_DISPLAY = {
    "lgbm": "LightGBM",
    "lstm": "LSTM",
    "tcn": "TCN",
    "patchtst": "PatchTST",
    "tft": "TFT",
}

FEAT_DIR = ROOT / "data" / "features"
PRED_DIR = ROOT / "data" / "predictions"
MODEL_PATH = ROOT / "models" / "lgbm_model.pkl"
FI_PATH = ROOT / "models" / "feature_importance.csv"
BT_RESULTS = ROOT / "data" / "backtest_results.csv"
BT_BY_MODEL = ROOT / "data" / "backtest_by_model.csv"
CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
)

EXCLUDE_COLS = {"ticker", "target", "future_return", "open", "high", "low", "close", "volume"}


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model() -> dict | None:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data(ttl=3600)
def load_ticker_data(ticker: str) -> pd.DataFrame:
    path = FEAT_DIR / f"{ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path).sort_index()
    return pd.DataFrame()


def _build_dl_model_instance(name: str, n_features: int, seq_len: int = 30):
    """DL モデルのインスタンスを生成する。インポート失敗時は None を返す。"""
    if not _DL_IMPORTS_OK:
        return None
    try:
        if name == "lstm":
            return LSTMClassifier(n_features=n_features)
        elif name == "tcn":
            return TCNClassifier(n_features=n_features)
        elif name == "patchtst":
            return PatchTSTClassifier(n_features=n_features, seq_len=seq_len)
        elif name == "tft":
            return TFTClassifier(n_features=n_features)
    except Exception as e:
        logger.error(f"Failed to build {name} model instance: {e}")
        return None
    return None


@st.cache_resource
def load_dl_model(name: str, n_features: int, seq_len: int = 30):
    """DL モデルをファイルからロードして DLModelWrapper を返す。存在しない場合は None。"""
    if not _DL_IMPORTS_OK:
        return None
    path = ROOT / "models" / f"{name}_model.pt"
    if not path.exists():
        return None
    model_instance = _build_dl_model_instance(name, n_features, seq_len)
    if model_instance is None:
        return None
    try:
        wrapper = DLModelWrapper.load(path, model_instance, seq_len=seq_len)
        return wrapper
    except Exception as e:
        logger.error(f"Failed to load {name} model: {e}")
        return None


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """EXCLUDE_COLS を除いた特徴量列名を返す。"""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def get_dl_probs(df: pd.DataFrame, feature_cols: list[str], seq_len: int = 30) -> dict[str, float | None]:
    """利用可能な DL モデルの上昇確率を辞書で返す。ロード失敗 / 推論失敗は None。"""
    n_features = len(feature_cols)
    result = {}
    for name in DL_MODEL_NAMES:
        wrapper = load_dl_model(name, n_features, seq_len)
        if wrapper is None:
            result[name] = None
            continue
        try:
            prob = wrapper.predict_latest(df, feature_cols)
            result[name] = float(prob) if not (prob != prob) else None  # NaN check
        except Exception:
            result[name] = None
    return result


def compute_ensemble_prob(lgbm_prob: float, dl_probs: dict[str, float | None]) -> float:
    """LightGBM + 利用可能な DL モデルのアンサンブル平均確率を返す。"""
    values = [lgbm_prob]
    for v in dl_probs.values():
        if v is not None:
            values.append(v)
    return float(np.mean(values))


def get_available_tickers() -> list[str]:
    return sorted([p.stem for p in FEAT_DIR.glob("*.parquet")])


def plot_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=ticker,
    ))
    colors = {"ma_5": "orange", "ma_25": "royalblue", "ma_60": "crimson"}
    for col, color in colors.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col.upper(),
                line=dict(color=color, width=1), opacity=0.8,
            ))
    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date", yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI(14)", line=dict(color="purple")))
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.update_layout(title="RSI (14)", yaxis=dict(range=[0, 100]), height=300)
    return fig


def plot_macd(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="orange")))
    colors = ["green" if v >= 0 else "red" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogram", marker_color=colors, opacity=0.6))
    fig.update_layout(title="MACD (12, 26, 9)", height=300)
    return fig


def plot_backtest_curves(curves: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curves.index, y=curves["strat_cum"],
        name="Strategy", line=dict(color="green", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=curves.index, y=curves["bh_cum"],
        name="Buy & Hold", line=dict(color="steelblue", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"{ticker} — Strategy vs Buy & Hold (test period)",
        yaxis_title="Cumulative Return", height=400,
    )
    return fig


def compute_predictions_for_all(model_bundle: dict, tickers: list[str]) -> pd.DataFrame:
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    rows = []
    for ticker in tickers:
        df = load_ticker_data(ticker)
        if df.empty:
            continue
        latest = df.tail(1)[feature_cols].fillna(0)
        lgbm_prob = model.predict_proba(latest)[0, 1]

        # DL モデルの確率（利用可能なもののみ）
        all_feature_cols = get_feature_cols(df)
        dl_probs = get_dl_probs(df, all_feature_cols)
        ensemble_prob = compute_ensemble_prob(lgbm_prob, dl_probs)

        rows.append({
            "ticker": ticker,
            "close": round(float(df["close"].iloc[-1]), 2),
            "lgbm_prob": round(lgbm_prob, 4),
            "ensemble_prob": round(ensemble_prob, 4),
            "signal": "BUY 🟢" if ensemble_prob >= 0.5 else "HOLD 🔴",
        })
    return pd.DataFrame(rows).sort_values("ensemble_prob", ascending=False)


@st.cache_data(ttl=3600)
def load_backtest_by_model() -> pd.DataFrame:
    if BT_BY_MODEL.exists():
        try:
            return pd.read_csv(BT_BY_MODEL)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_multi_predictions() -> pd.DataFrame:
    """最新のマルチモデル予測 CSV を読み込む。存在しない場合は空 DataFrame。"""
    files = sorted(PRED_DIR.glob("predictions_multi_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_csv(files[0])
    except Exception:
        return pd.DataFrame()


def main() -> None:
    st.title("📈 Stock Prediction Dashboard")
    st.caption("LightGBM-based 5-day direction prediction for US equities")

    tickers = get_available_tickers()
    model_bundle = load_model()
    cfg = load_config()

    if not tickers:
        st.error("特徴量データが見つかりません。以下を実行してください:\n"
                 "```\npython src/data_ingest.py\npython src/features.py\n```")
        return

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        selected = st.selectbox("Ticker", tickers)
        lookback = st.slider("Chart lookback (days)", 60, 1500, 365)
        st.divider()

        if model_bundle:
            st.success("✅ LightGBM loaded")
        else:
            st.warning("⚠️ No model — run train.py first")

        # DL モデルの利用状況表示
        if _DL_IMPORTS_OK:
            dl_sample_df = load_ticker_data(tickers[0]) if tickers else pd.DataFrame()
            if not dl_sample_df.empty:
                _sample_fcols = get_feature_cols(dl_sample_df)
                _n_feat = len(_sample_fcols)
                available_dl = [
                    n for n in DL_MODEL_NAMES
                    if (ROOT / "models" / f"{n}_model.pt").exists()
                ]
                if available_dl:
                    st.success(f"✅ DL models: {', '.join(available_dl).upper()}")
                else:
                    st.info("ℹ️ DL models not trained yet")

        if model_bundle:
            st.subheader("📊 All Tickers Ranking (Ensemble)")
            pred_df = compute_predictions_for_all(model_bundle, tickers)
            show_cols = ["ticker", "close", "ensemble_prob", "signal"]
            fmt = {"ensemble_prob": "{:.1%}", "close": "${:.2f}"}
            st.dataframe(
                pred_df[show_cols].style.format(fmt),
                hide_index=True,
                use_container_width=True,
            )

    # ── Top KPI cards ─────────────────────────────────────────────────────────
    df = load_ticker_data(selected)
    if df.empty:
        st.error(f"{selected} のデータが見つかりません。")
        return

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    latest_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2] if len(df) > 1 else latest_close
    daily_ret = (latest_close / prev_close - 1) * 100

    col1.metric("Ticker", selected)
    col2.metric("Latest Close", f"${latest_close:.2f}", f"{daily_ret:+.2f}%")
    col3.metric("Date", str(df.index[-1].date()))

    seq_len: int = cfg.get("dl_model", {}).get("seq_len", 30)
    lgbm_prob: float | None = None
    feature_cols: list[str] = []
    model = None

    if model_bundle:
        model = model_bundle["model"]
        feature_cols = model_bundle["feature_cols"]
        latest = df.tail(1)[feature_cols].fillna(0)
        lgbm_prob = model.predict_proba(latest)[0, 1]

        # DL モデルのアンサンブル確率
        all_feature_cols = get_feature_cols(df)
        dl_probs = get_dl_probs(df, all_feature_cols, seq_len=seq_len)
        ensemble_prob = compute_ensemble_prob(lgbm_prob, dl_probs)

        col4.metric("LightGBM Prob", f"{lgbm_prob:.1%}")
        col5.metric("Ensemble Prob", f"{ensemble_prob:.1%}")
        col6.metric("Signal", "BUY 🟢" if ensemble_prob >= 0.5 else "HOLD 🔴")
    else:
        col4.metric("LightGBM Prob", "—")
        col5.metric("Ensemble Prob", "—")
        col6.metric("Signal", "—")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    df_plot = df.tail(lookback)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📉 Price Chart", "📐 Technical Indicators", "🧪 Backtest", "🔍 Feature Importance", "🤖 Model Comparison"]
    )

    with tab1:
        st.plotly_chart(plot_price_chart(df_plot, selected), use_container_width=True)

        with st.expander("Volume"):
            fig_vol = go.Figure(go.Bar(x=df_plot.index, y=df_plot["volume"], name="Volume", opacity=0.7))
            if "volume_ma20" in df_plot.columns:
                fig_vol.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot["volume_ma20"],
                    name="Volume MA20", line=dict(color="orange"),
                ))
            fig_vol.update_layout(height=250)
            st.plotly_chart(fig_vol, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if "rsi_14" in df_plot.columns:
                st.plotly_chart(plot_rsi(df_plot), use_container_width=True)
        with c2:
            if "macd" in df_plot.columns:
                st.plotly_chart(plot_macd(df_plot), use_container_width=True)

        if "volatility_20" in df_plot.columns:
            fig_vola = go.Figure(go.Scatter(
                x=df_plot.index, y=df_plot["volatility_20"] * np.sqrt(252),
                name="Annualized Volatility", fill="tozeroy", line=dict(color="teal"),
            ))
            fig_vola.update_layout(title="20-day Rolling Volatility (annualized)", height=280)
            st.plotly_chart(fig_vola, use_container_width=True)

    with tab3:
        if not model_bundle:
            st.info("モデルを学習してから再度確認してください。")
        else:
            test_ratio = cfg["model"]["test_ratio"]
            commission = cfg["backtest"]["commission"]
            n = len(df)
            test_df = df.iloc[int(n * (1 - test_ratio)):].copy()
            X_test = test_df[feature_cols].fillna(0)
            test_df["signal_prob"] = model.predict_proba(X_test)[:, 1]
            metrics, curves = run_backtest(test_df, commission)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            m2.metric("Precision", f"{metrics['precision']:.1%}")
            m3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            m4.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
            m5.metric("Total Return", f"{metrics['total_return']:.1%}",
                      f"vs B&H {metrics['buy_hold_return']:.1%}")

            st.plotly_chart(plot_backtest_curves(curves, selected), use_container_width=True)

            if BT_RESULTS.exists():
                st.subheader("All Tickers Summary")
                bt_df = pd.read_csv(BT_RESULTS)
                st.dataframe(
                    bt_df.style.format({
                        "accuracy": "{:.1%}", "precision": "{:.1%}",
                        "sharpe": "{:.2f}", "max_drawdown": "{:.1%}",
                        "total_return": "{:.1%}", "buy_hold_return": "{:.1%}",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

            bt_by_model = load_backtest_by_model()
            if not bt_by_model.empty:
                st.divider()
                st.subheader("🏆 Model Comparison (averaged across tickers)")

                summary = (
                    bt_by_model
                    .groupby("model")[["accuracy", "precision", "sharpe", "max_drawdown", "total_return"]]
                    .mean()
                    .reset_index()
                    .sort_values("sharpe", ascending=False)
                )
                summary["model"] = summary["model"].map(
                    lambda x: DL_MODEL_DISPLAY.get(x, x.upper())
                )
                st.dataframe(
                    summary.style.format({
                        "accuracy": "{:.1%}", "precision": "{:.1%}",
                        "sharpe": "{:.2f}", "max_drawdown": "{:.1%}",
                        "total_return": "{:.1%}",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

                metric_options = {
                    "Sharpe Ratio": "sharpe",
                    "Total Return": "total_return",
                    "Accuracy": "accuracy",
                    "Precision": "precision",
                    "Max Drawdown": "max_drawdown",
                }
                selected_metric = st.selectbox(
                    "比較する指標", list(metric_options.keys()), key="bt_metric"
                )
                col_key = metric_options[selected_metric]

                pivot = bt_by_model.pivot_table(
                    index="ticker", columns="model", values=col_key
                ).reset_index()
                pivot.columns = [
                    "ticker" if c == "ticker" else DL_MODEL_DISPLAY.get(c, c.upper())
                    for c in pivot.columns
                ]
                fig_bt_cmp = go.Figure()
                for model_col in [c for c in pivot.columns if c != "ticker"]:
                    fig_bt_cmp.add_trace(go.Bar(
                        name=model_col,
                        x=pivot["ticker"],
                        y=pivot[model_col],
                    ))
                fmt = ".1%" if col_key != "sharpe" else ".2f"
                fig_bt_cmp.update_layout(
                    barmode="group",
                    title=f"{selected_metric} — per ticker",
                    yaxis_tickformat=fmt,
                    height=400,
                )
                st.plotly_chart(fig_bt_cmp, use_container_width=True)
            else:
                st.info(
                    "モデル別バックテスト結果がありません。\n"
                    "`python src/backtest.py` を実行すると `data/backtest_by_model.csv` が生成されます。"
                )

    with tab4:
        if not model_bundle:
            st.info("モデルを学習してから再度確認してください。")
        elif FI_PATH.exists():
            fi = pd.read_csv(FI_PATH, index_col=0).head(25).reset_index()
            fi.columns = ["feature", "importance"]
            fig_fi = px.bar(
                fi, x="importance", y="feature", orientation="h",
                title="Top 25 Feature Importances",
                color="importance", color_continuous_scale="Blues",
            )
            fig_fi.update_layout(height=600, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("backtest.py を実行して feature_importance.csv を生成してください。")

    with tab5:
        st.subheader(f"🤖 Model Comparison — {selected}")

        if not model_bundle:
            st.info("モデルを学習してから再度確認してください。")
        else:
            # 各モデルの上昇確率を収集
            model_probs: dict[str, float | None] = {"lgbm": lgbm_prob}
            all_feature_cols_t5 = get_feature_cols(df)
            dl_probs_t5 = get_dl_probs(df, all_feature_cols_t5, seq_len=seq_len)
            model_probs.update(dl_probs_t5)

            # バーチャート用データ作成
            bar_data = []
            for name, prob in model_probs.items():
                bar_data.append({
                    "model": DL_MODEL_DISPLAY.get(name, name.upper()),
                    "prob": round(prob, 4) if prob is not None else 0.0,
                    "available": prob is not None,
                })
            bar_df = pd.DataFrame(bar_data)

            colors = [
                "#2196F3" if avail else "#BDBDBD"
                for avail in bar_df["available"]
            ]
            fig_cmp = go.Figure(go.Bar(
                x=bar_df["model"],
                y=bar_df["prob"],
                marker_color=colors,
                text=[f"{p:.1%}" if avail else "N/A" for p, avail in zip(bar_df["prob"], bar_df["available"])],
                textposition="outside",
            ))
            fig_cmp.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Threshold 50%")
            fig_cmp.update_layout(
                title=f"{selected} — Up Probability by Model",
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                xaxis_title="Model",
                yaxis_title="Up Probability",
                height=400,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # 利用不可モデルの注記
            unavailable = [DL_MODEL_DISPLAY.get(n, n.upper()) for n, p in model_probs.items() if p is None]
            if unavailable:
                st.caption(f"グレー表示のモデルは未学習です: {', '.join(unavailable)}")

            # アンサンブル確率サマリ
            avail_count = sum(1 for p in model_probs.values() if p is not None)
            ens_prob_t5 = compute_ensemble_prob(
                lgbm_prob if lgbm_prob is not None else 0.0,
                {k: v for k, v in dl_probs_t5.items()},
            )
            st.info(
                f"アンサンブル平均確率: **{ens_prob_t5:.1%}** "
                f"({avail_count} モデルの平均)"
            )

            st.divider()

            # 全銘柄マルチモデル予測 CSV があれば表示
            multi_pred_df = load_multi_predictions()
            if not multi_pred_df.empty:
                st.subheader("📋 All Tickers — Multi-Model Predictions")
                files_list = sorted(PRED_DIR.glob("predictions_multi_*.csv"), reverse=True)
                if files_list:
                    st.caption(f"Source: {files_list[0].name}")
                st.dataframe(multi_pred_df, hide_index=True, use_container_width=True)
            else:
                st.info(
                    "マルチモデル予測 CSV が見つかりません。\n"
                    "DL モデルを学習後に予測スクリプトを実行すると\n"
                    "`data/predictions/predictions_multi_*.csv` が生成されます。"
                )


if __name__ == "__main__":
    main()
