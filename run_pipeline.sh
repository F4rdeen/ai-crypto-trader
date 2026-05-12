#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh — AI Crypto Trader Training Pipeline
#
# Runs the four data-preparation and model-training steps in
# sequence, stopping immediately if any step fails.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# After a successful run you will have:
#   btc_xgboost_bot_v1.pkl          ← trained model
#   btc_xgboost_bot_v1_shap_importance.csv
#   BTC_USDT_15m_data.csv
#   BTC_USDT_15m_features.csv
#   BTC_USDT_15m_ML_Ready.csv
# ============================================================

set -euo pipefail

echo "=============================================="
echo "  AI Crypto Trader — Training Pipeline"
echo "=============================================="
echo ""

echo "[1/4] Fetching historical OHLCV data from Binance..."
python Data_Extraction_Pipeline.py
echo ""

echo "[2/4] Engineering technical-indicator features..."
python Engineer_Features.py
echo ""

echo "[3/4] Generating binary target labels..."
python Create_Target_Labels.py
echo ""

echo "[4/4] Training the XGBoost model..."
python XGBoost.py
echo ""

echo "=============================================="
echo "  Pipeline complete!"
echo "  Model saved to: btc_xgboost_bot_v1.pkl"
echo "  Run the live engine with:"
echo "    python Live_Execution_Engine.py"
echo "=============================================="
