import time
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
})
try:
    session.get("https://finance.yahoo.com", timeout=10)
except Exception as e:
    print(f"[warmup] {e}")

pred_dir = Path("data/predictions")
eval_dir = Path("data/evaluations")
eval_dir.mkdir(parents=True, exist_ok=True)

cache = {}

def fetch_history(ticker, start, end):
    key = (ticker, start, end)
    if key in cache:
        return cache[key]
    last_err = None
    for attempt in range(4):
        try:
            t = yf.Ticker(ticker, session=session)
            h = t.history(start=start, end=end, auto_adjust=True)
            if h is not None and len(h) > 0:
                cache[key] = h
                return h
        except Exception as e:
            last_err = e
        time.sleep(3 * (attempt + 1))
    if last_err:
        print(f"  {ticker}: {last_err}")
    return None

results = []
for csv_file in sorted(pred_dir.glob("predictions_multi_*.csv"))[-3:]:
    df = pd.read_csv(csv_file)
    pred_date = csv_file.stem.replace("predictions_multi_", "")
    pred_dt = datetime.strptime(pred_date, "%Y%m%d")
    end_dt = pred_dt + timedelta(days=7)
    start_s = pred_dt.strftime("%Y-%m-%d")
    end_s = end_dt.strftime("%Y-%m-%d")
    print(f"\n[{csv_file.name}] rows={len(df)} range={start_s}..{end_s}")
    for _, row in df.iterrows():
        ticker = row["ticker"]
        hist = fetch_history(ticker, start_s, end_s)
        if hist is None or len(hist) < 2:
            n = 0 if hist is None else len(hist)
            print(f"  {ticker}: insufficient data ({n} rows)")
            continue
        close_series = hist["Close"]
        if hasattr(close_series, "columns"):
            close_series = close_series.iloc[:, 0]
        first = float(close_series.iloc[0])
        last = float(close_series.iloc[-1])
        actual_return = (last - first) / first
        actual_signal = 1 if actual_return > 0 else 0
        pred_signal = 1 if row["signal"] == "BUY" else 0
        correct = int(pred_signal == actual_signal)
        results.append({
            "pred_date": pred_date,
            "ticker": ticker,
            "predicted_signal": row["signal"],
            "ensemble_prob": row["ensemble_prob"],
            "actual_return_pct": round(actual_return * 100, 2),
            "actual_signal": "UP" if actual_signal else "DOWN",
            "correct": correct,
        })
        time.sleep(1.5)

eval_df = pd.DataFrame(results)
if not eval_df.empty:
    today = datetime.today().strftime("%Y%m%d")
    out_path = eval_dir / f"evaluation_{today}.csv"
    eval_df.to_csv(out_path, index=False)
    accuracy = eval_df["correct"].mean()
    print(f"\n=== Evaluation Summary ===")
    print(f"総サンプル数: {len(eval_df)}")
    print(f"総合精度: {accuracy:.1%}")
    print("\n--- Per-ticker ---")
    print(eval_df.groupby("ticker")[["correct", "actual_return_pct"]].mean().round(3).to_string())
    print("\n--- Per prediction date ---")
    print(eval_df.groupby("pred_date")[["correct", "actual_return_pct"]].mean().round(3).to_string())
    print(f"\n詳細結果保存先: {out_path}")
else:
    print("\n[WARN] No evaluation results produced.")
