import ccxt
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta

def fetch_crypto_data(symbol='BTC/USDT', timeframe='15m', days_back=30):
    print(f"Initializing data pull for {symbol} on {timeframe} timeframe...")
    
    # Initialize the Binance exchange (no API keys needed for public data)
    exchange = ccxt.binance({
        'enableRateLimit': True, # Crucial: prevents getting banned by the exchange
    })

    # Calculate the starting timestamp (in milliseconds)
    since_ts = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    all_ohlcv = []

    # Pagination loop: Binance limits how many candles you can pull at once (usually 1000)
    while True:
        try:
            print(f"Fetching data since: {datetime.fromtimestamp(since_ts/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Fetch the data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
            
            if len(ohlcv) == 0:
                break # We've caught up to the current time
                
            all_ohlcv.extend(ohlcv)
            
            # Update the 'since' timestamp to the last candle's time + 1 millisecond
            since_ts = ohlcv[-1][0] + 1
            
            # Be polite to the API
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Convert the raw array into a clean Pandas DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp to a readable datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to CSV
    filename = f"{symbol.replace('/', '_')}_{timeframe}_data.csv"
    df.to_csv(filename, index=False)
    
    print(f"\nSuccess! Downloaded {len(df)} rows of data.")
    print(f"Data saved to: {filename}")
    
    return df

# Execute the function
if __name__ == "__main__":
    # Let's pull the last 60 days to give our AI plenty of training data
    historical_data = fetch_crypto_data(symbol='BTC/USDT', timeframe='15m', days_back=60)
    print(historical_data.head())
    

def engineer_features(input_csv, output_csv):
    print(f"Loading raw data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # pandas-ta needs the timestamp to be a datetime object (if not already)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Calculating Technical Indicators (Features)...")

    # --- 1. TREND INDICATORS ---
    # Moving Averages help the model understand the broader direction
    df.ta.ema(length=9, append=True)   # Fast trend
    df.ta.ema(length=21, append=True)  # Medium trend
    df.ta.ema(length=50, append=True)  # Slow trend

    # --- 2. MOMENTUM INDICATORS ---
    # Helps the model identify overbought/oversold conditions and trend strength
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # --- 3. VOLATILITY INDICATORS ---
    # Crucial for setting dynamic stop-losses and understanding market chop
    df.ta.bbands(length=20, std=2, append=True) # Bollinger Bands
    df.ta.atr(length=14, append=True)           # Average True Range

    # --- 4. VOLUME INDICATORS ---
    # Validates price movements (price moving up on low volume is often a trap)
    df.ta.obv(append=True) # On-Balance Volume

    # --- CLEANUP ---
    # Indicators like the 50-EMA require 50 previous rows to calculate.
    # This creates 'NaN' (Not a Number) values at the top of our dataset.
    # Machine learning models crash if fed NaNs, so we drop them.
    initial_rows = len(df)
    df.dropna(inplace=True)
    
    print(f"Dropped {initial_rows - len(df)} rows containing NaN values.")
    
    # Save the enriched dataset
    df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Feature-rich data saved to: {output_csv}")
    print(f"Total features available for ML model: {len(df.columns)}")
    
    return df

if __name__ == "__main__":
    # Ensure this matches the filename from the first script
    INPUT_FILE = "BTC_USDT_15m_data.csv"
    OUTPUT_FILE = "BTC_USDT_15m_features.csv"
    
    features_df = engineer_features(INPUT_FILE, OUTPUT_FILE)
    
    # Display the new columns to verify
    print("\nNew Dataset Columns:")
    print(features_df.columns.tolist())
    

def create_target_labels(input_csv, output_csv, lookahead_periods=4, profit_target_pct=0.5):
    print(f"Loading feature data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Generating Target Labels (Lookahead: {lookahead_periods} periods, Target: {profit_target_pct}%)...")

    # 1. Capture the highest price over the next 'n' periods
    # We do this explicitly to make the logic bulletproof and easy to read
    future_highs = pd.DataFrame()
    for i in range(1, lookahead_periods + 1):
        future_highs[f'shift_{i}'] = df['high'].shift(-i)
    
    # Get the maximum high from those future columns
    df['future_max_high'] = future_highs.max(axis=1)

    # 2. Calculate the maximum potential percentage gain from the CURRENT close
    df['max_potential_gain_pct'] = ((df['future_max_high'] - df['close']) / df['close']) * 100

    # 3. Create the Binary Target Column (1 = Hit Profit Target, 0 = Did Not Hit)
    # If the potential gain is greater than or equal to our threshold, it's a Buy signal
    df['target'] = (df['max_potential_gain_pct'] >= profit_target_pct).astype(int)

    # 4. Clean up the dataset
    # We must drop the last few rows because they don't have future data to look at!
    # If we keep them, they will have 'NaN' and crash the ML model.
    df.dropna(subset=['future_max_high'], inplace=True)
    
    # We also drop the temporary calculation columns so the AI doesn't cheat by 
    # looking at 'future_max_high' during training
    df.drop(columns=['future_max_high', 'max_potential_gain_pct'], inplace=True)

    # Save the final, ML-ready dataset
    df.to_csv(output_csv, index=False)
    
    # Print some statistics to see class balance
    total_samples = len(df)
    buy_signals = df['target'].sum()
    print(f"\nTarget Generation Complete!")
    print(f"Total rows: {total_samples}")
    print(f"Total 'Buy' signals (1s): {buy_signals} ({round((buy_signals/total_samples)*100, 2)}% of data)")
    print(f"Final ML-ready dataset saved to: {output_csv}")

    return df

if __name__ == "__main__":
    INPUT_FILE = "BTC_USDT_15m_features.csv"
    OUTPUT_FILE = "BTC_USDT_15m_ML_Ready.csv"
    
    final_df = create_target_labels(INPUT_FILE, OUTPUT_FILE)