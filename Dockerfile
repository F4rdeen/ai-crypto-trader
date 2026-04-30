# ============================================================
# Dockerfile — AI Crypto Trader Live Execution Engine
#
# Build:
#   docker build -t ai-crypto-trader .
#
# Run (supply your own API keys as env vars — never hard-code them):
#   docker run -d --restart unless-stopped \
#     -e EXCHANGE_ID=binance \
#     -e EXCHANGE_API_KEY=<your_key> \
#     -e EXCHANGE_SECRET=<your_secret> \
#     -e SYMBOL=BTC/USDT \
#     -e TIMEFRAME=15m \
#     -e TRADE_AMOUNT_USDT=50 \
#     -e WEBHOOK_URL=https://your-webhook-url \
#     -v $(pwd)/btc_xgboost_bot_v1.pkl:/app/btc_xgboost_bot_v1.pkl:ro \
#     -v $(pwd)/logs:/app/logs \
#     --name ai-crypto-trader \
#     ai-crypto-trader
# ============================================================

# Use a slim, official Python base image.
FROM python:3.11-slim

# Metadata
LABEL maintainer="ai-crypto-trader" \
      description="XGBoost-powered 15-minute momentum trading bot"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# (important so log lines appear immediately in docker logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user for security.
RUN addgroup --system trader && adduser --system --ingroup trader trader

# Set the working directory.
WORKDIR /app

# Install OS-level dependencies needed by some Python packages (e.g. numpy).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-caching optimisation —
# only re-run pip install when requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application source code.
COPY Live_Execution_Engine.py .
COPY Engineer_Features.py .

# The trained model is mounted at runtime (see docker run -v above).
# A placeholder is created so the image builds cleanly; the real .pkl
# must be provided at runtime or baked in with a second COPY layer.
RUN touch btc_xgboost_bot_v1.pkl

# Create a logs directory that the container user can write to.
RUN mkdir -p /app/logs && chown trader:trader /app/logs

# Drop privileges.
USER trader

# Health-check: verify the Python environment is intact.
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import ccxt, xgboost, pandas_ta; print('OK')" || exit 1

# Default command: run the live execution engine.
CMD ["python", "Live_Execution_Engine.py"]
