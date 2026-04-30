"""
Live_Execution_Engine.py
------------------------
24/7 trading daemon for the AI Crypto Trader.

Architecture
------------
1. Load the trained XGBoost model from disk.
2. Connect to a ccxt-compatible exchange using API keys stored in environment
   variables (no hard-coded secrets).
3. Every time a 15-minute candle closes, fetch the most recent OHLCV data,
   run the exact same feature-engineering logic used during training, and
   pass the final row to the model for inference.
4. If the model predicts a BUY (1):
   - Place a market buy order.
   - Immediately set a take-profit limit order at +0.5 %.
   - Immediately set a stop-loss at max(ATR-based distance, hard 0.5 % floor).
5. While a position is open the bot will not open another one (state guard).
6. All events are logged locally and broadcast to an optional webhook
   (e.g. n8n / Telegram) as a JSON payload.

Environment Variables
---------------------
EXCHANGE_ID        Exchange ID recognised by ccxt (default: "binance")
EXCHANGE_API_KEY   Your exchange API key
EXCHANGE_SECRET    Your exchange API secret
SYMBOL             Trading pair (default: "BTC/USDT")
TIMEFRAME          Candle timeframe (default: "15m")
TRADE_AMOUNT_USDT  Amount in quote currency per trade (default: 50)
WEBHOOK_URL        (Optional) POST target for trade/error alerts
LOG_LEVEL          Python log level (default: "INFO")

Usage
-----
    python Live_Execution_Engine.py

PEP-8 compliant. Python 3.9+.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & environment-driven configuration
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.getenv("MODEL_PATH", "btc_xgboost_bot_v1.pkl")
EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance")
EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_SECRET: str = os.getenv("EXCHANGE_SECRET", "")
SYMBOL: str = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")
TRADE_AMOUNT_USDT: float = float(os.getenv("TRADE_AMOUNT_USDT", "50"))
WEBHOOK_URL: Optional[str] = os.getenv("WEBHOOK_URL")

# Candle duration in seconds (15 minutes = 900 s)
CANDLE_SECONDS: int = 900

# Minimum number of historical candles needed to compute all indicators.
# EMA-50 needs 50 rows; add a safety buffer.
LOOKBACK_CANDLES: int = 200

# Risk-management parameters
TAKE_PROFIT_PCT: float = 0.005   # +0.5 %
STOP_LOSS_PCT: float = 0.005     # -0.5 % (hard floor)

# Retry settings for API calls
MAX_RETRIES: int = 5
RETRY_BASE_DELAY: float = 1.0   # seconds; doubles on each attempt


# ---------------------------------------------------------------------------
# Webhook / alerting
# ---------------------------------------------------------------------------

def send_webhook_alert(payload: dict) -> None:
    """
    POST *payload* as JSON to WEBHOOK_URL.

    Silently skips if WEBHOOK_URL is not configured.  Errors are logged but
    never allowed to crash the trading loop.
    """
    if not WEBHOOK_URL:
        return
    try:
        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        log.debug("Webhook alert delivered (HTTP %s).", response.status_code)
    except Exception as exc:  # noqa: BLE001
        log.warning("Webhook delivery failed: %s", exc)


# ---------------------------------------------------------------------------
# API retry decorator with exponential back-off
# ---------------------------------------------------------------------------

def with_retry(func, *args, **kwargs):
    """
    Call *func*(*args*, **kwargs) up to MAX_RETRIES times, doubling the wait
    between each attempt (exponential back-off with full jitter).

    Raises the last exception if all attempts are exhausted.
    """
    delay = RETRY_BASE_DELAY
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as exc:
            last_exc = exc
            jitter = delay * (0.5 + 0.5 * (attempt / MAX_RETRIES))
            log.warning(
                "API call failed (attempt %d/%d): %s — retrying in %.1f s",
                attempt, MAX_RETRIES, exc, jitter,
            )
            time.sleep(jitter)
            delay *= 2
        except ccxt.AuthenticationError as exc:
            log.critical("Authentication error — check your API keys: %s", exc)
            raise
        except ccxt.InsufficientFunds as exc:
            log.error("Insufficient funds: %s", exc)
            raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """
    Deserialise and return the XGBoost model from *model_path*.

    Raises FileNotFoundError if the .pkl file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path!r}. "
            "Run XGBoost.py first to train and save the model."
        )
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    log.info("Model loaded from %s", model_path)
    return model


# ---------------------------------------------------------------------------
# Exchange initialisation
# ---------------------------------------------------------------------------

def create_exchange() -> ccxt.Exchange:
    """
    Initialise and return an authenticated ccxt exchange instance.

    The exchange class is resolved dynamically from EXCHANGE_ID so you can
    swap to a different broker by changing a single environment variable.
    """
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class(
        {
            "apiKey": EXCHANGE_API_KEY,
            "secret": EXCHANGE_SECRET,
            "enableRateLimit": True,   # ccxt auto-throttles to stay under limits
            "options": {"defaultType": "spot"},
        }
    )
    log.info(
        "Exchange initialised: %s | Symbol: %s | Timeframe: %s",
        EXCHANGE_ID, SYMBOL, TIMEFRAME,
    )
    return exchange


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_recent_ohlcv(exchange: ccxt.Exchange, limit: int = LOOKBACK_CANDLES) -> pd.DataFrame:
    """
    Fetch the most recent *limit* closed OHLCV candles for SYMBOL/TIMEFRAME.

    Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
    The most-recent (potentially still-open) candle is intentionally excluded
    to avoid acting on incomplete price data.
    """
    raw = with_retry(
        exchange.fetch_ohlcv,
        SYMBOL,
        timeframe=TIMEFRAME,
        limit=limit + 1,    # +1 so we can drop the live candle
    )
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Drop the last row — it represents the still-forming candle.
    df = df.iloc[:-1].reset_index(drop=True)

    log.debug("Fetched %d closed candles (latest: %s).", len(df), df["timestamp"].iloc[-1])
    return df


# ---------------------------------------------------------------------------
# Feature engineering (mirrors Engineer_Features.py exactly)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the identical technical-indicator pipeline used during training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with NaN rows removed.
    """
    # --- Trend ---
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)

    # --- Momentum ---
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # --- Volatility ---
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    # --- Volume ---
    df.ta.obv(append=True)

    # Drop rows that contain NaN (leading rows without enough history).
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_latest_feature_row(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Return the last row of *df* formatted as the feature matrix expected by
    the model, plus the latest ATR value for dynamic stop-loss sizing.

    Columns dropped to match training: 'timestamp', 'target' (if present).
    """
    drop_cols = [c for c in ["timestamp", "target"] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    # The ATR column is named "ATRr_14" by pandas-ta.
    atr_col = [c for c in df.columns if c.lower().startswith("atr")][0]
    atr_value: float = float(df[atr_col].iloc[-1])

    # Isolate the very last row as a single-row DataFrame for .predict().
    latest_row = feature_df.tail(1).reset_index(drop=True)
    return latest_row, atr_value


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def place_market_buy(exchange: ccxt.Exchange, amount_usdt: float) -> dict:
    """
    Place a market buy order for *amount_usdt* worth of SYMBOL.

    Returns the raw order dict from the exchange.
    """
    log.info("Placing MARKET BUY | %s | ~$%.2f USDT", SYMBOL, amount_usdt)
    order = with_retry(
        exchange.create_order,
        SYMBOL,
        "market",
        "buy",
        None,               # quantity = None when using quoteOrderQty
        params={"quoteOrderQty": amount_usdt},
    )
    log.info("BUY order filled | ID: %s | Price: %s", order.get("id"), order.get("price"))
    return order


def place_take_profit_order(
    exchange: ccxt.Exchange, amount_base: float, entry_price: float
) -> dict:
    """
    Place a limit sell order at entry_price × (1 + TAKE_PROFIT_PCT).
    """
    tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)
    log.info("Setting TAKE-PROFIT LIMIT SELL | Price: %.2f", tp_price)
    order = with_retry(
        exchange.create_limit_sell_order,
        SYMBOL,
        amount_base,
        tp_price,
    )
    log.info("Take-profit order placed | ID: %s", order.get("id"))
    return order


def place_stop_loss_order(
    exchange: ccxt.Exchange,
    amount_base: float,
    entry_price: float,
    atr_value: float,
    current_close: float,
) -> dict:
    """
    Place a stop-market sell order.

    The stop price is the greater of:
      • entry_price − 1× ATR  (volatility-scaled)
      • entry_price × (1 − STOP_LOSS_PCT)  (hard -0.5 % floor)

    This ensures we always have a minimum protection even in low-volatility
    regimes, while also respecting dynamic market conditions.
    """
    atr_stop = entry_price - atr_value
    hard_stop = entry_price * (1 - STOP_LOSS_PCT)
    stop_price = round(max(atr_stop, hard_stop), 2)

    log.info(
        "Setting STOP-LOSS | Entry: %.2f | ATR stop: %.2f | Hard stop: %.2f → Using: %.2f",
        entry_price, atr_stop, hard_stop, stop_price,
    )

    # ccxt uses 'stopPrice' in the params dict for stop-market orders.
    order = with_retry(
        exchange.create_order,
        SYMBOL,
        "stop_market",
        "sell",
        amount_base,
        params={"stopPrice": stop_price},
    )
    log.info("Stop-loss order placed | ID: %s", order.get("id"))
    return order


def cancel_open_orders(exchange: ccxt.Exchange) -> None:
    """Cancel all open orders for SYMBOL (used when exiting a position)."""
    try:
        open_orders = with_retry(exchange.fetch_open_orders, SYMBOL)
        for order in open_orders:
            with_retry(exchange.cancel_order, order["id"], SYMBOL)
            log.info("Cancelled order %s", order["id"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not cancel open orders: %s", exc)


# ---------------------------------------------------------------------------
# Position state management
# ---------------------------------------------------------------------------

class TradeState:
    """
    Lightweight in-memory state tracker.

    Attributes
    ----------
    in_position : bool
        True while the bot holds an open trade.
    entry_price : float | None
        The fill price of the current BUY order.
    base_amount : float | None
        How many units of the base asset were purchased.
    tp_order_id : str | None
        Exchange order ID of the take-profit leg.
    sl_order_id : str | None
        Exchange order ID of the stop-loss leg.
    """

    def __init__(self) -> None:
        self.in_position: bool = False
        self.entry_price: Optional[float] = None
        self.base_amount: Optional[float] = None
        self.tp_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None

    def open_trade(
        self,
        entry_price: float,
        base_amount: float,
        tp_order_id: str,
        sl_order_id: str,
    ) -> None:
        self.in_position = True
        self.entry_price = entry_price
        self.base_amount = base_amount
        self.tp_order_id = tp_order_id
        self.sl_order_id = sl_order_id
        log.info(
            "Trade OPENED | Entry: %.2f | Amount: %.6f",
            entry_price, base_amount,
        )

    def close_trade(self, reason: str = "unknown") -> None:
        log.info(
            "Trade CLOSED | Reason: %s | Entry was: %.2f",
            reason, self.entry_price or 0,
        )
        self.in_position = False
        self.entry_price = None
        self.base_amount = None
        self.tp_order_id = None
        self.sl_order_id = None


# ---------------------------------------------------------------------------
# Position monitoring (check if TP or SL was already filled)
# ---------------------------------------------------------------------------

def check_position_status(exchange: ccxt.Exchange, state: TradeState) -> None:
    """
    Poll the TP and SL order statuses.  If either has been filled or
    cancelled by the exchange, clean up the other leg and mark the
    position as closed.
    """
    if not state.in_position:
        return

    try:
        for order_id, label in [
            (state.tp_order_id, "take-profit"),
            (state.sl_order_id, "stop-loss"),
        ]:
            if order_id is None:
                continue
            order = with_retry(exchange.fetch_order, order_id, SYMBOL)
            if order["status"] in ("closed", "filled"):
                log.info("%s order %s has been filled.", label.capitalize(), order_id)
                cancel_open_orders(exchange)
                state.close_trade(reason=label)
                send_webhook_alert(
                    {
                        "event": "position_closed",
                        "reason": label,
                        "order_id": order_id,
                        "symbol": SYMBOL,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                return
    except Exception as exc:  # noqa: BLE001
        log.warning("Error while checking position status: %s", exc)


# ---------------------------------------------------------------------------
# Candle-close synchronisation
# ---------------------------------------------------------------------------

def seconds_until_next_candle(candle_seconds: int = CANDLE_SECONDS) -> float:
    """
    Return the number of seconds until the next *candle_seconds*-aligned
    boundary in UTC.  Adds a 2-second buffer so the candle has time to
    be published by the exchange before we query it.
    """
    now_ts = time.time()
    seconds_into_period = now_ts % candle_seconds
    wait = candle_seconds - seconds_into_period + 2.0
    return wait


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------

def run_trading_bot() -> None:
    """
    Entry point for the live trading daemon.

    Runs indefinitely, waking at the close of each 15-minute candle.
    """
    log.info("=" * 60)
    log.info("AI Crypto Trading Bot — STARTING UP")
    log.info("Symbol: %s | Timeframe: %s | Trade size: $%.2f USDT",
             SYMBOL, TIMEFRAME, TRADE_AMOUNT_USDT)
    log.info("=" * 60)

    # 1. Load model
    model = load_model(MODEL_PATH)

    # 2. Initialise exchange connection
    exchange = create_exchange()

    # 3. Warm up: verify connectivity
    try:
        with_retry(exchange.load_markets)
        log.info("Markets loaded. Exchange connectivity confirmed.")
    except Exception as exc:
        log.critical("Cannot connect to exchange: %s", exc)
        send_webhook_alert({"event": "fatal_error", "detail": str(exc)})
        raise

    # 4. Initialise trade state
    state = TradeState()

    # Retrieve the column names the model was trained on so we can ensure
    # the live feature row has exactly the same columns in the same order.
    try:
        trained_feature_names: list[str] = model.get_booster().feature_names
    except AttributeError:
        trained_feature_names = None  # type: ignore[assignment]

    send_webhook_alert(
        {
            "event": "bot_started",
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    # 5. Main loop
    while True:
        # ------------------------------------------------------------------
        # Synchronise to the next candle close
        # ------------------------------------------------------------------
        wait_secs = seconds_until_next_candle()
        next_close = datetime.now(timezone.utc)
        log.info(
            "Sleeping %.1f s until next candle close (~%s UTC) ...",
            wait_secs,
            datetime.fromtimestamp(
                time.time() + wait_secs, tz=timezone.utc
            ).strftime("%H:%M:%S"),
        )
        time.sleep(wait_secs)

        log.info("-" * 50)
        log.info("Candle closed. Running inference cycle ...")

        # ------------------------------------------------------------------
        # Check if an existing position has been closed by TP/SL
        # ------------------------------------------------------------------
        if state.in_position:
            check_position_status(exchange, state)

        # ------------------------------------------------------------------
        # Fetch recent candles & engineer features
        # ------------------------------------------------------------------
        try:
            raw_df = fetch_recent_ohlcv(exchange)
        except Exception as exc:
            log.error("Failed to fetch OHLCV data: %s", exc)
            send_webhook_alert({"event": "data_fetch_error", "detail": str(exc)})
            continue

        try:
            feature_df = engineer_features(raw_df.copy())
            latest_row, atr_value = get_latest_feature_row(feature_df)
        except Exception as exc:
            log.error("Feature engineering failed: %s", exc)
            continue

        # Align columns to the exact order/set used during training.
        if trained_feature_names is not None:
            missing = set(trained_feature_names) - set(latest_row.columns)
            if missing:
                log.error("Feature mismatch — missing columns: %s", missing)
                continue
            latest_row = latest_row[trained_feature_names]

        # ------------------------------------------------------------------
        # Model inference
        # ------------------------------------------------------------------
        try:
            prediction: int = int(model.predict(latest_row)[0])
            probability: float = float(model.predict_proba(latest_row)[0][1])
        except Exception as exc:
            log.error("Model inference failed: %s", exc)
            continue

        current_close: float = float(feature_df["close"].iloc[-1])
        log.info(
            "Prediction: %s | Confidence: %.2f%% | Close: %.2f | ATR: %.4f",
            "BUY" if prediction == 1 else "HOLD",
            probability * 100,
            current_close,
            atr_value,
        )

        # ------------------------------------------------------------------
        # Execution gate
        # ------------------------------------------------------------------
        if prediction != 1:
            log.info("Signal = HOLD. No action taken.")
            continue

        if state.in_position:
            log.info(
                "Signal = BUY but already in a position (entry: %.2f). Skipping.",
                state.entry_price,
            )
            continue

        # ------------------------------------------------------------------
        # Enter trade — buy, then set TP + SL
        # ------------------------------------------------------------------
        try:
            buy_order = place_market_buy(exchange, TRADE_AMOUNT_USDT)

            # Extract fill details from the order response.
            fill_price: float = float(
                buy_order.get("average") or buy_order.get("price") or current_close
            )
            base_amount: float = float(
                buy_order.get("filled") or buy_order.get("amount")
                or (TRADE_AMOUNT_USDT / fill_price)
            )

            # Place protective orders.
            tp_order = place_take_profit_order(exchange, base_amount, fill_price)
            sl_order = place_stop_loss_order(
                exchange, base_amount, fill_price, atr_value, current_close
            )

            state.open_trade(
                entry_price=fill_price,
                base_amount=base_amount,
                tp_order_id=str(tp_order.get("id", "")),
                sl_order_id=str(sl_order.get("id", "")),
            )

            alert_payload = {
                "event": "trade_opened",
                "symbol": SYMBOL,
                "entry_price": fill_price,
                "base_amount": base_amount,
                "take_profit": round(fill_price * (1 + TAKE_PROFIT_PCT), 2),
                "stop_loss": round(
                    max(fill_price - atr_value, fill_price * (1 - STOP_LOSS_PCT)), 2
                ),
                "model_confidence_pct": round(probability * 100, 2),
                "atr": round(atr_value, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            log.info("Trade alert: %s", alert_payload)
            send_webhook_alert(alert_payload)

        except ccxt.InsufficientFunds as exc:
            log.error("Insufficient funds — cannot place order: %s", exc)
            send_webhook_alert({"event": "insufficient_funds", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected error during order placement: %s", exc)
            send_webhook_alert({"event": "order_error", "detail": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_trading_bot()
    except KeyboardInterrupt:
        log.info("Shutdown requested by user (KeyboardInterrupt).")
        send_webhook_alert(
            {
                "event": "bot_stopped",
                "reason": "KeyboardInterrupt",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        log.critical("Fatal unhandled exception: %s", exc, exc_info=True)
        send_webhook_alert({"event": "fatal_crash", "detail": str(exc)})
        raise
