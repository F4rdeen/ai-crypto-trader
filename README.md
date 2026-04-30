# AI Crypto Trader

An end-to-end, fully automated machine-learning trading bot that uses **XGBoost** to predict short-term momentum breakouts on 15-minute candle data.

---

## Architecture Overview

```
Data_Extraction_Pipeline.py
        │  (raw OHLCV CSV)
        ▼
Engineer_Features.py
        │  (feature-enriched CSV)
        ▼
Create_Target_Labels.py
        │  (ML-ready CSV with binary target)
        ▼
XGBoost.py
        │  (trained model → btc_xgboost_bot_v1.pkl)
        ▼
Live_Execution_Engine.py   ◄── 24/7 daemon (this is the final piece)
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `Data_Extraction_Pipeline.py` | Pulls historical 15-minute OHLCV data from Binance via `ccxt` and saves to CSV. |
| `Engineer_Features.py` | Adds EMA-9/21/50, RSI-14, MACD, Bollinger Bands, ATR-14, and OBV using `pandas-ta`. |
| `Create_Target_Labels.py` | Labels each row `1` (BUY) if the next 4 candles reach ≥ 0.5 % gain, else `0`. |
| `XGBoost.py` | Trains an `XGBClassifier`, evaluates on precision, saves the model, and runs SHAP feature analysis. |
| `Live_Execution_Engine.py` | **The live 24/7 daemon** — fetches data, runs inference, places orders, manages risk. |

---

## Pipeline Walkthrough

### Step 1 — Data Extraction
```bash
python Data_Extraction_Pipeline.py
# Output: BTC_USDT_15m_data.csv
```

### Step 2 — Feature Engineering
```bash
python Engineer_Features.py
# Input:  BTC_USDT_15m_data.csv
# Output: BTC_USDT_15m_features.csv
```

### Step 3 — Target Labels
```bash
python Create_Target_Labels.py
# Input:  BTC_USDT_15m_features.csv
# Output: BTC_USDT_15m_ML_Ready.csv
```

### Step 4 — Train the Model
```bash
python XGBoost.py
# Input:  BTC_USDT_15m_ML_Ready.csv
# Output: btc_xgboost_bot_v1.pkl  +  btc_xgboost_bot_v1_shap_importance.csv
```

### Step 5 — Run the Live Engine
```bash
# Set environment variables first (see Configuration section below)
python Live_Execution_Engine.py
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `EXCHANGE_ID` | `binance` | Any ccxt-supported exchange ID |
| `EXCHANGE_API_KEY` | *(required)* | Your exchange API key |
| `EXCHANGE_SECRET` | *(required)* | Your exchange API secret |
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `TIMEFRAME` | `15m` | Candle timeframe |
| `TRADE_AMOUNT_USDT` | `50` | Quote-currency trade size |
| `WEBHOOK_URL` | *(optional)* | POST endpoint for JSON trade/error alerts (n8n, Telegram, etc.) |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `MODEL_PATH` | `btc_xgboost_bot_v1.pkl` | Path to the trained model |

---

## Risk Management

The engine automatically applies the following risk rules for every trade:

- **Take-Profit**: Limit sell at entry price × 1.005 (+0.5 %)
- **Stop-Loss**: `max(entry − 1×ATR, entry × 0.995)` — dynamic ATR floor with a hard -0.5 % minimum
- **State Guard**: A second BUY signal is ignored while a position is already open

---

## Feature Engineering Details

| Category | Indicator | Column(s) |
|----------|-----------|-----------|
| Trend | EMA | `EMA_9`, `EMA_21`, `EMA_50` |
| Momentum | RSI | `RSI_14` |
| Momentum | MACD | `MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9` |
| Volatility | Bollinger Bands | `BBL_20_2.0`, `BBM_20_2.0`, `BBU_20_2.0`, `BBB_20_2.0`, `BBP_20_2.0` |
| Volatility | ATR | `ATRr_14` |
| Volume | OBV | `OBV` |

---

## SHAP & RFE Feature Optimisation

**XGBoost.py** now supports two feature-selection strategies:

```python
# SHAP (default ON) — computes mean absolute SHAP value per feature on the test set.
# Features with SHAP < 1% of the top feature are flagged for removal.
train_trading_model(INPUT_FILE, MODEL_FILE, run_shap=True)

# RFECV (off by default — takes several minutes) — cross-validated recursive elimination.
train_trading_model(INPUT_FILE, MODEL_FILE, run_rfe=True)
```

---

## AWS Deployment (Docker)

### Build
```bash
docker build -t ai-crypto-trader .
```

### Run on EC2
```bash
docker run -d --restart unless-stopped \
  -e EXCHANGE_ID=binance \
  -e EXCHANGE_API_KEY=<your_key> \
  -e EXCHANGE_SECRET=<your_secret> \
  -e SYMBOL=BTC/USDT \
  -e TIMEFRAME=15m \
  -e TRADE_AMOUNT_USDT=50 \
  -e WEBHOOK_URL=https://your-webhook-url \
  -v $(pwd)/btc_xgboost_bot_v1.pkl:/app/btc_xgboost_bot_v1.pkl:ro \
  -v $(pwd)/logs:/app/logs \
  --name ai-crypto-trader \
  ai-crypto-trader
```

View live logs:
```bash
docker logs -f ai-crypto-trader
```

---

## Logging & Alerting

- All events are written to **`trading_bot.log`** and stdout simultaneously.
- When `WEBHOOK_URL` is set, the engine POSTs a JSON payload on:
  - Bot start/stop
  - Trade opened / position closed
  - Insufficient funds or order errors
  - Fatal crashes

Example payload (trade opened):
```json
{
  "event": "trade_opened",
  "symbol": "BTC/USDT",
  "entry_price": 67450.12,
  "base_amount": 0.000741,
  "take_profit": 67787.37,
  "stop_loss": 67112.45,
  "model_confidence_pct": 73.4,
  "atr": 337.67,
  "timestamp": "2025-01-15T09:30:02+00:00"
}
```

---

## Disclaimer

This software is for **educational purposes only**. Cryptocurrency trading involves significant financial risk. Never risk more than you can afford to lose. Past model performance does not guarantee future returns.
