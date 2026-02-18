
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
from sampo2.config import OUTPUT_DIR

RENKO_FILE = OUTPUT_DIR / "renko_bricks_dataset.csv"
MODEL_FILE = OUTPUT_DIR / "renko_predictor_xgboost.json"

def train_predictor():
    print(f"Loading Renko Dataset from {RENKO_FILE}...")
    df = pd.read_csv(RENKO_FILE)
    
    # Drop Non-Feature Columns
    # We should exclude the target-derived columns from the features to prevent leakage
    # We keep 'Close', 'Renko_Value', 'Flag' as they are known at time T
    # We drop 'Target_State', 'Class', 'Blue', 'Red', 'Yellow'... actually Blue/Red/Yellow AT TIME T are valid state features!
    # The Target is specific to T+1.
    
    target_col = 'Target_State'
    drop_cols = ['Time', 'Class', 'Target_State', 'Target_Next_Blue'] 
    # Exclude self-generated predictions to prevent leakage if re-training
    drop_cols += [c for c in df.columns if 'renko_' in c or 'prob_' in c]
    
    # Also drop columns that might not exist or are raw strings
    df = df.select_dtypes(include=[np.number]) # Helper to only keep numeric
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    
    print(f"Features: {X.columns.tolist()}")
    
    # Time Series Split (No shuffling)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Set: {len(X_train)} samples")
    print(f"Test Set: {len(X_test)} samples")
    
    # Calculate Class Weights to handle imbalance (Pure Blue is rare)
    # XGBoost supports 'scale_pos_weight' for binary, but for multi-class we might need sample_weight
    # For now, let's run vanilla to establish baseline.
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    
    print("Training XGBoost Model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50, early_stopping_rounds=50)
    
    print("Evaluating...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Pure Red", "Pure Blue", "Red/Yellow", "Blue/Yellow"]))
    
    # Save Model
    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

def predict_full_dataset(input_csv, renko_file=RENKO_FILE):
    """
    Appends Renko Model Predictions to the H1 dataset by merging predictions from the Bricks dataset.
    Args:
        input_csv: Path to H1 csv (final_combined_dataset.csv)
        renko_file: Path to Bricks csv (optional override)
    Returns:
        DataFrame (H1) with predictions added.
    """
    print(f"Running Inference & Merge...")
    
    # 1. Prediction on BRICKS (Correct Shape)
    # We must load the dataset used for training/testing
    renko_path = Path(renko_file)
    if not renko_path.exists():
        raise FileNotFoundError(f"Renko Bricks file missing: {renko_path}")
        
    df_bricks = pd.read_csv(renko_path)
    if 'Time' in df_bricks.columns:
        df_bricks['Time'] = pd.to_datetime(df_bricks['Time'])
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    # Feature Select (Same as training)
    target_col = 'Target_State'
    drop_cols = ['Time', 'Class', 'Target_State', 'Target_Next_Blue'] 
    drop_cols += [c for c in df_bricks.columns if 'renko_' in c or 'prob_' in c]
    
    df_features = df_bricks.select_dtypes(include=[np.number])
    X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
    
    print(f"Predicting on Bricks: {X.shape}")
    
    probs = model.predict_proba(X)
    
    # Create Prediction DataFrame
    pred_df = pd.DataFrame(index=df_bricks.index)
    pred_df['Time'] = df_bricks['Time']
    pred_df['renko_prob_red'] = probs[:, 0]
    pred_df['renko_prob_blue'] = probs[:, 1]
    pred_df['renko_prob_ry'] = probs[:, 2]
    pred_df['renko_prob_by'] = probs[:, 3]
    pred_df['renko_pred_class'] = model.predict(X)
    
    # 2. Merge to H1 (input_csv)
    print(f"Merging predictions to {input_csv}...")
    df_h1 = pd.read_csv(input_csv)
    if 'Time' in df_h1.columns:
        df_h1['Time'] = pd.to_datetime(df_h1['Time'])
        
    df_h1 = df_h1.sort_values('Time')
    pred_df = pred_df.sort_values('Time')
    
    # Merge AsOf (Backward) -> For each H1 timestamp, get the latest available Brick Prediction
    df_merged = pd.merge_asof(df_h1, pred_df, on='Time', direction='backward')
    
    # Fill any NaNs (before first brick) with 0 or neutral
    df_merged.fillna(0, inplace=True)
    
    if 'Time' in df_merged.columns:
        df_merged.set_index('Time', inplace=True)
        
    print(f"Merge Complete. Final Shape: {df_merged.shape}")
    return df_merged

if __name__ == "__main__":
    train_predictor()
