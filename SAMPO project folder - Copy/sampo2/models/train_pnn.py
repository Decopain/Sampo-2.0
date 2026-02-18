
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sampo2.data.loader import DataLoader
from sampo2.models.pnn import ParallelNeuralNetwork
from sampo2.config import OUTPUT_DIR, DAILY_DATA_FILE, H1_DATA_FILE, M15_DATA_FILE

def prepare_targets(df, steps_ahead=1):
    """
    Creates target: Next prediction Return. 
    Can be expanded to predict direction, or multiple horizons.
    """
    # Simple target: Next Close / Current Close - 1
    # aligned with time t. target at t is returns at t+1
    target = df['Close'].pct_change().shift(-steps_ahead).fillna(0)
    return target.values.reshape(-1, 1)

def run_training():
    loader = DataLoader() # Helper for loading/validating
    
    print("Loading Multi-Timeframe Data...")
    
    # Load all 3 files
    try:
        loader.filepath = DAILY_DATA_FILE
        df_daily = loader.load_data()
        df_daily = loader.add_returns(df_daily)
        
        loader.filepath = H1_DATA_FILE
        df_h1 = loader.load_data()
        df_h1 = loader.add_returns(df_h1)
        
        loader.filepath = M15_DATA_FILE
        df_m15 = loader.load_data()
        df_m15 = loader.add_returns(df_m15)
        
        print(f"Loaded: Daily({len(df_daily)}), H1({len(df_h1)}), M15({len(df_m15)})")
        
    except FileNotFoundError as e:
        print(f"Error loading source files: {e}")
        return

    # Align Data
    # Base is M15 (Highest Frequency)
    base_index = df_m15.index
    
    # Reindex Daily to M15 (ffill)
    # Using reindex logic similar to notebook: `resample('15min').ffill()`
    # Reindex does essentially the same if we have the target index.
    print("Aligning Daily and H1 to M15 index...")
    
    df_daily_aligned = df_daily.reindex(base_index, method='ffill').fillna(0)
    df_h1_aligned = df_h1.reindex(base_index, method='ffill').fillna(0)
    
    # Feature Engineering (Basic)
    # df = loader.add_returns(df) # features already in CSV
    
    # Full Feature Set from "Official SAMPO Training.ipynb" (56 columns, excluding Close/Time)
    features_cols = [
        'Ren_Value', 'Blue', 'Red', 'dir', 'Yellow', 'Blue/Yellow', 'Red/Yellow', 
        'Frac_Value', 'Flag', 'HI_Value', 'BearWarning', 'BullWarning', 'OBV', 'VWAP', 
        'bullish_engulfing', 'bullish_harami', 'dark_cloud_cover', 'doji', 'doji_star', 
        'dragonfly_doji', 'gravestone_doji', 'hammer', 'hanging_man', 'morning_star', 
        'morning_star_doji', 'piercing_pattern', 'rain_drop', 'rain_drop_doji', 
        'shooting_star', 'star', 'Open', 'High', 'Low', 'volume', 'complete', 'spread', 
        'returns', 'SMA_S', 'SMA_C', 'SMA_L', 'up_sma1', 'up_sma2', 'down_sma1', 
        'down_sma2', 'flat_sma1', 'flat_sma2', 'peak', 'crust', 'atr', 'buy_Frac', 
        'sell_Frac', 'neg_buy_Frac', 'neg_sell_Frac', 'hi_up', 'hi_down', 't_dir'
    ]
    
    def get_norm_matrix(dataframe):
        mat = dataframe[features_cols].values
        # Simple standardization
        return (mat - np.mean(mat, axis=0)) / (np.std(mat, axis=0) + 1e-8)

    daily_matrix = get_norm_matrix(df_daily_aligned)
    h1_matrix = get_norm_matrix(df_h1_aligned)
    m15_matrix = get_norm_matrix(df_m15)
    
    # Inputs for PNN
    X = [daily_matrix, h1_matrix, m15_matrix] 
    
    # Target (based on M15 Close)
    y = prepare_targets(df_m15)
    
    # Shapes
    input_shapes = {
        'daily': (daily_matrix.shape[1],),
        'h1': (h1_matrix.shape[1],),
        'm15': (m15_matrix.shape[1],)
    }
    
    print(f"Building PNN with input dims: D={daily_matrix.shape}, H1={h1_matrix.shape}, M15={m15_matrix.shape}")
    pnn = ParallelNeuralNetwork(input_shapes, output_dim=1)
    
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
    
    # Checkpoint Paths
    checkpoint_dir = OUTPUT_DIR / "checkpoints_pnn"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "pnn_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras"
    best_model_path = OUTPUT_DIR / "pnn_best_model.keras" # Best model based on val_loss

    # Callbacks
    callbacks = [
        # Save checkpoints every epoch
        ModelCheckpoint(filepath=str(checkpoint_path), save_weights_only=False, verbose=1),
        # Save best model separately
        ModelCheckpoint(filepath=str(best_model_path), monitor='val_loss', save_best_only=True, verbose=1),
        # Stop if no improvement for 10 epochs
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        # Tensorboard
        TensorBoard(log_dir=str(OUTPUT_DIR / "tensorboard_pnn"))
    ]
    
    print("Training PNN...")
    # Train with validation split to enable monitoring
    pnn.fit(X, y, epochs=50, batch_size=64, validation_split=0.2, callbacks=callbacks) 
    
    # Save
    model_path = OUTPUT_DIR / "pnn_model.keras"
    pnn.save(model_path)
    
    # Extract Enriched Features
    print("Extracting Enriched Features...")
    extractor = pnn.get_feature_extractor()
    enriched = extractor.predict(X)
    
    # Save Enriched
    enriched_df = pd.DataFrame(enriched, index=base_index, 
                               columns=[f'enriched_{i}' for i in range(enriched.shape[1])])
    output_csv = OUTPUT_DIR / "enriched_features.csv"
    enriched_df.to_csv(output_csv)
    print(f"Enriched features saved to {output_csv}")

if __name__ == "__main__":
    run_training()
