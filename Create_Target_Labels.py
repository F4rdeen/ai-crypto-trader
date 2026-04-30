"""
Create_Target_Labels.py
-----------------------
Reads a feature-enriched CSV, generates a binary classification target column,
and saves the final ML-ready dataset.

Target definition
-----------------
For every row, look ahead *lookahead_periods* candles (default: 4 = 1 hour on
15m data).  If the maximum *high* price reached within that window represents a
gain of at least *profit_target_pct* % from the current *close*, label the row
``target = 1`` (BUY signal), otherwise ``target = 0``.

The last *lookahead_periods* rows are dropped because they have no future data
to label against.

Usage
-----
    python Create_Target_Labels.py
"""

import pandas as pd


def create_target_labels(
    input_csv: str,
    output_csv: str,
    lookahead_periods: int = 4,
    profit_target_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Generate binary buy/no-buy labels via a forward-looking profit window.

    Parameters
    ----------
    input_csv : str
        Path to the feature-enriched CSV produced by Engineer_Features.py.
    output_csv : str
        Destination path for the ML-ready CSV.
    lookahead_periods : int
        Number of 15-minute candles to look ahead (4 = 1 hour).
    profit_target_pct : float
        Minimum percentage gain required to label a row as a buy signal.

    Returns
    -------
    pd.DataFrame
        The labelled DataFrame (also written to disk).
    """
    print(f"Loading feature data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(
        f"Generating Target Labels "
        f"(Lookahead: {lookahead_periods} periods, Target: {profit_target_pct}%)..."
    )

    # ------------------------------------------------------------------
    # Step 1 – Capture the highest *high* across the next N candles.
    #          We shift the 'high' column by 1..N periods, then take
    #          the row-wise maximum of those shifted columns.
    # ------------------------------------------------------------------
    future_highs = pd.DataFrame(
        {f"shift_{i}": df["high"].shift(-i) for i in range(1, lookahead_periods + 1)}
    )
    df["future_max_high"] = future_highs.max(axis=1)

    # ------------------------------------------------------------------
    # Step 2 – Calculate the maximum potential percentage gain from the
    #          current close price.
    # ------------------------------------------------------------------
    df["max_potential_gain_pct"] = (
        (df["future_max_high"] - df["close"]) / df["close"]
    ) * 100

    # ------------------------------------------------------------------
    # Step 3 – Apply the binary threshold to create the target column.
    # ------------------------------------------------------------------
    df["target"] = (df["max_potential_gain_pct"] >= profit_target_pct).astype(int)

    # ------------------------------------------------------------------
    # Step 4 – Remove the final N rows (they have NaN future data) and
    #          drop the temporary helper columns so the model cannot
    #          "cheat" by looking at future information.
    # ------------------------------------------------------------------
    df.dropna(subset=["future_max_high"], inplace=True)
    df.drop(columns=["future_max_high", "max_potential_gain_pct"], inplace=True)

    # Save the final, ML-ready dataset.
    df.to_csv(output_csv, index=False)

    # Summary statistics.
    total_samples = len(df)
    buy_signals = int(df["target"].sum())
    buy_pct = round((buy_signals / total_samples) * 100, 2) if total_samples else 0

    print(f"\nTarget Generation Complete!")
    print(f"Total rows          : {total_samples}")
    print(f"Buy signals (1s)    : {buy_signals} ({buy_pct}% of data)")
    print(f"ML-ready dataset    : {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE = "BTC_USDT_15m_features.csv"
    OUTPUT_FILE = "BTC_USDT_15m_ML_Ready.csv"

    final_df = create_target_labels(INPUT_FILE, OUTPUT_FILE)
