"""
Data_Extraction_Pipeline.py
----------------------------
Pulls historical 15-minute OHLCV candlestick data from Binance using ccxt
and saves it to a CSV file for downstream feature engineering.

No API keys are required — this script only uses public market data endpoints.

Usage
-----
    python Data_Extraction_Pipeline.py
    # Output: BTC_USDT_15m_data.csv
"""

import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd


def fetch_crypto_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    days_back: int = 60,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance and save it to a CSV file.

    Parameters
    ----------
    symbol : str
        The trading pair to download (e.g. "BTC/USDT").
    timeframe : str
        Candle timeframe recognised by ccxt (e.g. "15m", "1h").
    days_back : int
        How many calendar days of history to pull.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    print(f"Initializing data pull for {symbol} on {timeframe} timeframe...")

    # No API keys needed for public OHLCV endpoints.
    exchange = ccxt.binance({"enableRateLimit": True})

    # Starting timestamp in milliseconds.
    since_ts = int(
        (datetime.now() - timedelta(days=days_back)).timestamp() * 1000
    )
    all_ohlcv: list = []

    # Pagination loop — Binance caps responses at 1 000 candles per request.
    while True:
        try:
            print(
                f"  Fetching from: "
                f"{datetime.fromtimestamp(since_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ts, limit=1000
            )

            if not ohlcv:
                break  # All available history has been retrieved.

            all_ohlcv.extend(ohlcv)

            # Advance the window by 1 ms past the last returned candle.
            since_ts = ohlcv[-1][0] + 1

            # Respect the exchange rate limit.
            time.sleep(exchange.rateLimit / 1000)

        except Exception as exc:
            print(f"An error occurred while fetching data: {exc}")
            break

    if not all_ohlcv:
        raise RuntimeError(
            "No OHLCV data was returned. "
            "Check your internet connection and the symbol/timeframe."
        )

    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    filename = f"{symbol.replace('/', '_')}_{timeframe}_data.csv"
    df.to_csv(filename, index=False)

    print(f"\nSuccess! Downloaded {len(df):,} rows of data.")
    print(f"Data saved to: {filename}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    historical_data = fetch_crypto_data(
        symbol="BTC/USDT", timeframe="15m", days_back=60
    )
    print(historical_data.head())
