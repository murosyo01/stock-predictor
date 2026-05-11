"""
株価データ取得スクリプト
yfinance で OHLCV を取得し Parquet + DuckDB に保存する
"""
import logging
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import yaml
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_ticker(ticker: str, start: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, auto_adjust=True)
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df.index = df.index.tz_localize(None)
        df.index.name = "date"
        df["ticker"] = ticker
        keep = [c for c in ["open", "high", "low", "close", "volume", "ticker"] if c in df.columns]
        return df[keep]
    except Exception as e:
        logger.error(f"Failed to download {ticker}: {e}")
        return pd.DataFrame()


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    logger.info(f"Saved {len(df)} rows → {path}")


def build_duckdb(raw_dir: Path) -> None:
    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        return
    db_path = raw_dir / "stock.db"
    con = duckdb.connect(str(db_path))
    con.execute(f"""
        CREATE OR REPLACE TABLE prices AS
        SELECT * FROM read_parquet('{raw_dir}/*.parquet')
    """)
    count = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()
    logger.info(f"DuckDB updated: {count} rows → {db_path}")


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    start = cfg["data"]["start_date"]

    all_tickers = cfg["tickers"] + cfg.get("market_indices", [])

    for ticker in all_tickers:
        logger.info(f"Downloading {ticker} ...")
        df = download_ticker(ticker, start)
        if df.empty:
            continue
        safe = ticker.replace("^", "_").replace("=", "_")
        save_parquet(df, raw_dir / f"{safe}.parquet")

    build_duckdb(raw_dir)
    logger.info("Data ingestion complete.")


if __name__ == "__main__":
    main()
