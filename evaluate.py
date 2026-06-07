import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

pred_dir = Path("data/predictions")
eval_dir = Path("data/evaluations")
eval_dir.mkdir(parents=True, exist_ok=True)

results = []
for csv_file in sorted(pred_dir.glob("predictions_multi_*.csv"))[-3:]:
    df = pd.read_csv(csv_file)
    pred_date = csv_file.stem.replace("predictions_multi_", "")
    pred_dt = datetime.strptime(pred_date, "%Y%m%d")
    end_dt = pred_dt + timedelta(days=7)
    print(f"\n--- Evaluating {csv_file.name} ---")
    for _, row in df.iterrows():
        ticker = row["ticker"]
        try:
            hist = yf.download(ticker, start=pred_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            if len(hist) >= 2:
                actual_return = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
                if hasattr(actual_return, 'item'):
                    actual_return = actual_return.item()
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
                    "correct": correct
                })
            else:
                print(f"{ticker}: insufficient data ({len(hist)} rows)")
        except Exception as e:
            print(f"{ticker}: error - {e}")

eval_df = pd.DataFrame(results)
if not eval_df.empty:
    today = datetime.today().strftime("%Y%m%d")
    out_path = eval_dir / f"evaluation_{today}.csv"
    eval_df.to_csv(out_path, index=False)
    accuracy = eval_df["correct"].mean()
    print(f"\n=== Evaluation Summary ===")
    print(f"総サンプル数: {len(eval_df)}")
    print(f"総合精度: {accuracy:.1%}")
    print("\n--- 銘柄別精度・実際リターン ---")
    print(eval_df.groupby("ticker")[["correct", "actual_return_pct"]].mean().round(3).to_string())
    print(f"\n--- 予測日別精度 ---")
    print(eval_df.groupby("pred_date")["correct"].agg(['mean', 'count']).round(3).to_string())
    print(f"\n詳細結果保存先: {out_path}")
else:
    print("評価データなし")
