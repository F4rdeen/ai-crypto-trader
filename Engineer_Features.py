"""
Engineer_Features.py
--------------------
Reads raw OHLCV data from a CSV file, computes a rich set of technical
indicators using pandas-ta, and saves the enriched dataset to a new CSV.

Indicators added
----------------
Trend      : EMA-9, EMA-21, EMA-50
Momentum   : RSI-14, MACD(12,26,9)
Volatility : Bollinger Bands(20,2), ATR-14
Volume     : OBV

Usage
-----
    python Engineer_Features.py
"""

import pandas as pd
import pandas_ta as ta


def engineer_features(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Load raw OHLCV data, compute technical indicators, drop NaN rows,
    and write the result to *output_csv*.

    Parameters
    ----------
    input_csv : str
        Path to the raw OHLCV CSV produced by Data_Extraction_Pipeline.py.
    output_csv : str
        Destination path for the feature-enriched CSV.

    Returns
    -------
    pd.DataFrame
        The feature-enriched DataFrame (also written to disk).
    """
    print(f"Loading raw data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # pandas-ta works best when the timestamp is a proper datetime column.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Calculating Technical Indicators (Features)...")

    # ------------------------------------------------------------------
    # 1. TREND INDICATORS
    #    Moving averages give the model a sense of the broader price direction.
    # ------------------------------------------------------------------
    df.ta.ema(length=9, append=True)   # Fast trend (EMA_9)
    df.ta.ema(length=21, append=True)  # Medium trend (EMA_21)
    df.ta.ema(length=50, append=True)  # Slow trend (EMA_50)

    # ------------------------------------------------------------------
    # 2. MOMENTUM INDICATORS
    #    Help identify overbought/oversold conditions and trend strength.
    # ------------------------------------------------------------------
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # ------------------------------------------------------------------
    # 3. VOLATILITY INDICATORS
    #    Crucial for dynamic stop-loss sizing and spotting choppy markets.
    # ------------------------------------------------------------------
    df.ta.bbands(length=20, std=2, append=True)  # Bollinger Bands
    df.ta.atr(length=14, append=True)            # Average True Range

    # ------------------------------------------------------------------
    # 4. VOLUME INDICATOR
    #    Validates price moves — rising price on low volume is often a trap.
    # ------------------------------------------------------------------
    df.ta.obv(append=True)  # On-Balance Volume

    # ------------------------------------------------------------------
    # CLEANUP
    #    Long-period indicators (EMA-50, etc.) produce NaN for the first
    #    ~50 rows.  ML models cannot handle NaNs, so we drop those rows.
    # ------------------------------------------------------------------
    initial_rows = len(df)
    df.dropna(inplace=True)
    dropped = initial_rows - len(df)

    print(f"Dropped {dropped} rows containing NaN values.")

    df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Feature-rich data saved to: {output_csv}")
    print(f"Total features available for the ML model: {len(df.columns)}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE = "BTC_USDT_15m_data.csv"
    OUTPUT_FILE = "BTC_USDT_15m_features.csv"

    features_df = engineer_features(INPUT_FILE, OUTPUT_FILE)

    print("\nNew Dataset Columns:")
    print(features_df.columns.tolist())
