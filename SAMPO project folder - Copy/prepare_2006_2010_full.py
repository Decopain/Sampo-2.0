
import os
# Fix Protobuf issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from pathlib import Path
import sys
import os

# Add root to sys.path
sys.path.append(os.getcwd())

from sampo2.config import OUTPUT_DIR
from sampo2.data.aggregator import SampoDataAggregator
from sampo2.data.garch import GarchVolatilityModel
from sampo2.data.renko_extractor import RenkoIndicator, TradeBar

# CONFIG
RAW_FILE = "outputs/phase9_raw/EUR_USD_H1_2006_2010_merged.csv"
FINAL_FILE = "outputs/tuning_dataset_2006_2010.csv"
PNN_MODEL_PATH = "outputs/pnn_model_full.keras" # User mentioned recently created models are here or in sampo2/outputs
RENKO_MODEL_PATH = "outputs/renko_predictor_xgboost.json"

def main():
    print("=== DATA PREPARATION 2006-2010 (Inference Mode) ===")
    
    # 1. LOAD RAW
    print(f"[1/5] Loading Raw Data: {RAW_FILE}")
    df_raw = pd.read_csv(RAW_FILE)
    if 'Time' in df_raw.columns:
        df_raw['Time'] = pd.to_datetime(df_raw['Time'], utc=True)
    if 'volume' not in df_raw.columns and 'Volume' in df_raw.columns:
        df_raw.rename(columns={'Volume': 'volume'}, inplace=True)

    # Clean duplicates
    df_raw = df_raw.sort_values('Time').drop_duplicates('Time').reset_index(drop=True)
        
    # 2. AGGREGATION (H1, D, M15)
    print("[2/5] Aggregating Multi-Timeframe Features...")
    agg = SampoDataAggregator(config="oanda.cfg")
    
    # H1
    df_h1 = agg.run(instrument="EUR_USD", granularity="H1", raw_df=df_raw)
    
    # D (Resample)
    d_raw = df_raw.copy().set_index('Time').resample('1D').agg(
        {'Open':'first','High':'max','Low':'min','Close':'last','volume':'sum'}).dropna().reset_index()
    if 'volume' not in d_raw.columns: d_raw['volume'] = 0
    df_d = agg.run(instrument="EUR_USD", granularity="D", raw_df=d_raw)
    
    # M15 (Resample/Upsample)
    m_raw = df_raw.copy().set_index('Time').resample('15min').ffill().reset_index()
    df_m15 = agg.run(instrument="EUR_USD", granularity="M15", raw_df=m_raw)
    
    # 3. GARCH
    print("[3/5] Computing GARCH Volatility...")
    garch = GarchVolatilityModel()
    if 'returns' not in df_h1.columns: df_h1['returns'] = df_h1['Close'].pct_change().fillna(0)
    # Fit GARCH on this specific dataset to adapt to its regime
    try:
        df_h1['volatility_h1'] = garch.fit_predict(df_h1['returns'])
    except Exception as e:
        print(f"GARCH failed ({e}), using rolling std dev fallback.")
        df_h1['volatility_h1'] = df_h1['returns'].rolling(24).std().fillna(0)

    # 4. PNN FEATURES
    print("[4/5] Extracting PNN Embeddings...")
    if Path(PNN_MODEL_PATH).exists():
        try:
            # Align D/M15 to H1
            df_d = df_d.reindex(df_h1.index, method='ffill')
            df_m15 = df_m15.reindex(df_h1.index, method='ffill')
            
            # Common Numeric Cols
            numeric = df_h1.select_dtypes(include=[np.number]).columns
            common = [c for c in numeric if c in df_d.columns and c in df_m15.columns]
            
            X_h1 = df_h1[common].fillna(0).values.astype('float32')
            X_d = df_d[common].fillna(0).values.astype('float32')
            X_m15 = df_m15[common].fillna(0).values.astype('float32')
            
            # Z-Score Norm
            def z_norm(x): return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
            X_h1, X_d, X_m15 = z_norm(X_h1), z_norm(X_d), z_norm(X_m15)
            
            # Load & Predict
            pnn = tf.keras.models.load_model(PNN_MODEL_PATH)
            # We assume the model has a layer named 'enriched_features' or similar
            # If not, we take the output of the penultimate layer or the model output itself
            try:
                fusion_layer = pnn.get_layer('enriched_features')
                extractor = tf.keras.Model(inputs=pnn.inputs, outputs=fusion_layer.output)
                embeddings = extractor.predict([X_h1, X_d, X_m15], verbose=0)
            except ValueError:
                print("Layer 'enriched_features' not found. Using model output as PNN signal.")
                embeddings = pnn.predict([X_h1, X_d, X_m15], verbose=0)
                
            emb_df = pd.DataFrame(embeddings, columns=[f'pnn_{i}' for i in range(embeddings.shape[1])])
            emb_df.index = df_h1.index
            df_h1 = df_h1.join(emb_df, how='inner')
            print(f"Added {embeddings.shape[1]} PNN features.")
            
        except Exception as e:
             print(f"PNN Extraction Failed: {e}. Filling with 0s.")
             for i in range(100): df_h1[f'pnn_{i}'] = 0.0
    else:
        print("PNN Model not found. Filling with 0s.")
        for i in range(100): df_h1[f'pnn_{i}'] = 0.0

    # 5. RENKO PREDICTION
    print("[5/5] Generating Renko Predictions...")
    renko_model = xgb.XGBClassifier()
    if Path(RENKO_MODEL_PATH).exists():
        renko_model.load_model(RENKO_MODEL_PATH)
        
        # Make Bricks
        renko = RenkoIndicator(name="Renko50", block_points=0.0005, instrument="EURUSD")
        renko_data = []
        for idx, row in df_h1.iterrows():
            bar = TradeBar(row.name, row['Open'], row['High'], row['Low'], row['Close'])
            renko.Update(bar)
            if renko.IsReady:
                state = renko.get_ObjectDictionary()
                d = {'Time': row.name, 'Close': row['Close'], 'Renko_Value': state['Ren_Value'], 
                     'Blue': int(state['Blue']), 'Red': int(state['Red']), 
                     'Yellow': int(state['Yellow']), 'Blue_Yellow': int(state['Blue/Yellow']), 
                     'Red_Yellow': int(state['Red/Yellow']), 'Flag': state['Flag']}
                # Add Context (includes PNN cols now)
                for col in row.index: 
                    if col not in d and col != 'Time': d[col] = row[col]
                renko_data.append(d)
        
        renko_df = pd.DataFrame(renko_data)
        if not renko_df.empty:
            bricks = renko_df[renko_df['Renko_Value'].shift() != renko_df['Renko_Value']].copy()
            
            # Predict
            drop_cols = ['Time', 'Class', 'Target_State', 'Target_Next_Blue'] 
            drop_cols += [c for c in bricks.columns if 'renko_' in c and 'prob' in c]
            
            df_features = bricks.select_dtypes(include=[np.number])
            X_renko = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
            
            # Match Booster Features
            booster = renko_model.get_booster()
            expected = booster.feature_names
            if expected:
                missing = [f for f in expected if f not in X_renko.columns]
                if missing: 
                    for m in missing: X_renko[m] = 0
                X_renko = X_renko[expected]
            
            probs = renko_model.predict_proba(X_renko)
            
            pred_df = pd.DataFrame(index=bricks.index)
            pred_df['Time'] = bricks['Time']
            pred_df['renko_prob_red'] = probs[:, 0]
            pred_df['renko_prob_blue'] = probs[:, 1]
            pred_df['renko_prob_ry'] = probs[:, 2]
            pred_df['renko_prob_by'] = probs[:, 3]
            pred_df['renko_pred_class'] = renko_model.predict(X_renko)
            
            # Merge
            df_h1['Time_Link'] = df_h1.index
            pred_df = pred_df.sort_values('Time')
            df_merged = pd.merge_asof(df_h1, pred_df, left_on='Time_Link', right_on='Time', direction='backward', suffixes=('', '_renko'))
            df_merged.set_index('Time_Link', inplace=True)
            df_merged.fillna(0, inplace=True)
            df_h1 = df_merged

    # SAVE
    print(f"Saving Final Dataset: {FINAL_FILE}")
    df_h1.to_csv(FINAL_FILE)
    print("Done!")

if __name__ == "__main__":
    main()
