import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, classification_report
import pickle

def train_trading_model(input_csv, model_filename):
    print(f"Loading ML-ready data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Separate Features (X) and Target (y)
    # We drop the timestamp and our target column from the features
    X = df.drop(columns=['timestamp', 'target'])
    y = df['target']
    
    print(f"Dataset Shape: {X.shape[0]} rows, {X.shape[1]} features.")

    # 2. Chronological Train/Test Split (80% Train, 20% Test)
    # DO NOT use random train_test_split for time-series!
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training on first {len(X_train)} candles...")
    print(f"Testing (simulating live trading) on next {len(X_test)} candles...")

    # 3. Initialize and Train the XGBoost Classifier
    # scale_pos_weight helps handle the imbalance (since we have way more 0s than 1s)
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=ratio, 
        random_state=42,
        eval_metric='logloss'
    )
    
    print("\nTraining the XGBoost brain. This may take a few seconds...")
    model.fit(X_train, y_train)

    # 4. Evaluate the Model on the Test Set
    predictions = model.predict(X_test)
    
    print("\n=== AI PERFORMANCE REPORT ===")
    # Accuracy is good, but PRECISION is what makes us money. 
    # Precision answers: "When the AI said BUY, how often was it actually right?"
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Trade Precision (Win Rate on Buy Signals): {precision * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions))

    # 5. Save the Model
    # This .pkl file is the actual "brain". We will upload this file to our AWS server.
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
        
    print(f"\nSuccess! Trained AI model saved to: {model_filename}")
    
    # 6. Show Feature Importance (What did the AI learn?)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Most Important Technical Indicators for this Model:")
    print(importance.head(5).to_string(index=False))

if __name__ == "__main__":
    INPUT_FILE = "BTC_USDT_15m_ML_Ready.csv"
    MODEL_FILE = "btc_xgboost_bot_v1.pkl"
    
    train_trading_model(INPUT_FILE, MODEL_FILE)