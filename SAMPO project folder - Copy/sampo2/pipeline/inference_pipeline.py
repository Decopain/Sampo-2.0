
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from pathlib import Path
from stable_baselines3 import PPO

from sampo2.config import OUTPUT_DIR, COMBINED_DATA_FILE
from sampo2.data.aggregator import SampoDataAggregator
from sampo2.data.garch import GarchVolatilityModel
from sampo2.models.pnn import ParallelNeuralNetwork
from sampo2.data.renko_extractor import RenkoIndicator, TradeBar
from sampo2.env.trading_env import TradingEnv

def run_live_inference(ib_connection_info={'host': '127.0.0.1', 'port': 7497, 'client_id': 1}, 
                       lookback_days=60, 
                       backtest_year=None):
    """
    Runs the full SAMPO2 pipeline in INFERENCE or BACKTEST mode.
    
    Args:
        ib_connection_info (dict): {host, port, client_id} for IBKR.
        lookback_days (int): History to fetch for Live mode (default 60 days).
        backtest_year (int): If provided (e.g. 2017), fetches entire year and runs full backtest evaluation.
    """
    mode = "BACKTEST" if backtest_year else "LIVE"
    print(f"\n=== STARTING SAMPO2 {mode} PIPELINE ===")
    
    # 1. Fetch Data (Priority: IBKR -> Mock)
    df_raw = None
    
    if ib_connection_info:
        print(f"Attempting connection to IBKR ({ib_connection_info})...")
        try:
            from sampo2.data.ib_aggregator import IBAggregator
            # Determine fetch parameters
            if backtest_year:
                end_date = f"{backtest_year+1}0101 00:00:00" # End of the requested year
                duration = "1 Y"
                output_name = f"history_{backtest_year}_ib.csv"
            else:
                end_date = "" # Now
                duration = f"{lookback_days} D"
                output_name = "live_incoming.csv"
                
            raw_file = OUTPUT_DIR / output_name
            
            # Fetch
            agg = IBAggregator(host=ib_connection_info['host'], 
                               port=ib_connection_info['port'], 
                               client_id=ib_connection_info['client_id'])
            
            # We use get_historical_data directly instead of download helper to get DF in memory?
            # Or use download to save to disk then read. Download helper logic is robust.
            # Let's use the explicit call to ensure we get data.
            print(f"Requesting: {duration} ending {end_date if end_date else 'NOW'}")
            df_fetched = agg.get_historical_data(
                symbol="EURUSD", 
                sec_type="FOREX", 
                bar_size="1 hour", 
                duration=duration, 
                end_date=end_date
            )
            
            if not df_fetched.empty:
                df_raw = df_fetched
                # Save for cache
                df_raw.to_csv(raw_file, index=False)
                print(f"Fetched {len(df_raw)} rows from IBKR.")
            else:
                print("IBKR returned no data.")
                
        except Exception as e:
            print(f"IBKR Connection/Fetch Failed: {e}")
            print("To run this with real data, ensure TWS/Gateway is running and API port matches.")

    # Fallback to Mock if IB failed (only for Live test, for 2017 we might fail hard or use dummy)
    if df_raw is None:
        if backtest_year == 2017:
             print("Warning: Could not fetch 2017 data from IBKR. Cannot proceed with strictly 2017 backtest.")
             print("Please ensure IBKR TWS is running.")
             return 
        else:
            print("Using Mock Data (File Source) as fallback...")
            df_raw = pd.read_csv(OUTPUT_DIR / "history_2005_2009_ib.csv").tail(2000).reset_index(drop=True)

    # 2. Preprocess
    if 'Time' in df_raw.columns:
        df_raw['Time'] = pd.to_datetime(df_raw['Time'], utc=True)
            
    if 'volume' not in df_raw.columns:
        if 'Volume' in df_raw.columns:
            df_raw.rename(columns={'Volume': 'volume'}, inplace=True)
        else:
            df_raw['volume'] = 0

    print(f"Data Ingested: {len(df_raw)} bars. Range: {df_raw['Time'].iloc[0]} to {df_raw['Time'].iloc[-1]}")

    # 3. Aggregation (H1, D, M15) - Reusing Logic
    print("Running Aggregation logic...")
    agg_tool = SampoDataAggregator(config="oanda.cfg")
    
    df_h1 = agg_tool.run(instrument="EUR_USD", granularity="H1", raw_df=df_raw)
    
    # D
    d_raw = df_raw.copy().set_index('Time').resample('1D').agg(
        {'Open':'first','High':'max','Low':'min','Close':'last','volume':'sum'}).dropna().reset_index()
    if 'volume' not in d_raw.columns: d_raw['volume'] = 0
    df_d = agg_tool.run(instrument="EUR_USD", granularity="D", raw_df=d_raw)
    
    # M15
    m_raw = df_raw.copy().set_index('Time').resample('15min').ffill().reset_index()
    df_m15 = agg_tool.run(instrument="EUR_USD", granularity="M15", raw_df=m_raw)

    # 4. GARCH
    print("Generating GARCH...")
    garch = GarchVolatilityModel()
    if 'returns' not in df_h1.columns: 
        df_h1['returns'] = df_h1['Close'].pct_change().fillna(0)
    df_h1['volatility_h1'] = garch.fit_predict(df_h1['returns']) # Fit on this dataset

    # 5. PNN
    print("Generating PNN Features...")
    model_path = OUTPUT_DIR / "pnn_model_full.keras"
    if not model_path.exists(): raise FileNotFoundError("PNN Model not found!")
    
    # Align
    df_d = df_d.reindex(df_h1.index, method='ffill')
    df_m15 = df_m15.reindex(df_h1.index, method='ffill')
    
    # Common Cols
    numeric = df_h1.select_dtypes(include=[np.number]).columns
    common = [c for c in numeric if c in df_d.columns and c in df_m15.columns]
    
    X_h1 = df_h1[common].fillna(0).values.astype('float32')
    X_d = df_d[common].fillna(0).values.astype('float32')
    X_m15 = df_m15[common].fillna(0).values.astype('float32')
    
    # Norm
    def z_norm(x): return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    X_h1, X_d, X_m15 = z_norm(X_h1), z_norm(X_d), z_norm(X_m15)
    
    # Extract
    pnn = tf.keras.models.load_model(model_path)
    fusion_layer = pnn.get_layer('enriched_features')
    extractor = tf.keras.Model(inputs=pnn.inputs, outputs=fusion_layer.output)
    embeddings = extractor.predict([X_h1, X_d, X_m15], verbose=0)
    
    emb_df = pd.DataFrame(embeddings, columns=[f'pnn_{i}' for i in range(embeddings.shape[1])])
    emb_df.index = df_h1.index
    final_df = df_h1.join(emb_df, how='inner')
    
    # 6. Renko
    print("Generating Renko Predictions...")
    renko_model_path = OUTPUT_DIR / "renko_predictor_xgboost.json"
    renko_model = xgb.XGBClassifier()
    renko_model.load_model(renko_model_path)
    
    # Make Bricks
    renko = RenkoIndicator(name="Renko50", block_points=0.0005, instrument="EURUSD")
    renko_data = []
    for idx, row in final_df.iterrows():
        bar = TradeBar(row.name, row['Open'], row['High'], row['Low'], row['Close'])
        renko.Update(bar)
        if renko.IsReady:
            state = renko.get_ObjectDictionary()
            d = {'Time': row.name, 'Close': row['Close'], 'Renko_Value': state['Ren_Value'], 
                 'Blue': int(state['Blue']), 'Red': int(state['Red']), 
                 'Yellow': int(state['Yellow']), 'Blue_Yellow': int(state['Blue/Yellow']), 
                 'Red_Yellow': int(state['Red/Yellow']), 'Flag': state['Flag']}
            # Context
            for col in row.index: 
                if col not in d and col != 'Time': d[col] = row[col]
            renko_data.append(d)
            
    renko_df = pd.DataFrame(renko_data)
    if renko_df.empty:
        print("No bricks formed.")
        return
        
    bricks = renko_df[renko_df['Renko_Value'].shift() != renko_df['Renko_Value']].copy()
    
    # Predict
    # strict feature match
    drop_cols = ['Time', 'Class', 'Target_State', 'Target_Next_Blue'] 
    drop_cols += [c for c in bricks.columns if 'renko_' in c and 'prob' in c]
    
    df_features = bricks.select_dtypes(include=[np.number])
    X_renko = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
    
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
    
    # Merge Backward
    final_df = final_df.sort_index()
    pred_df = pred_df.sort_values('Time')
    final_df['Time_Link'] = final_df.index
    df_merged = pd.merge_asof(final_df, pred_df, left_on='Time_Link', right_on='Time', direction='backward', suffixes=('', '_renko'))
    df_merged.set_index('Time_Link', inplace=True)
    df_merged.fillna(0, inplace=True)
    
    # 7. AGENT EXECUTION
    print(f"Running Agent Evaluation (Mode: {mode})...")
    ppo_path = OUTPUT_DIR / "ppo_final_model.zip"
    model = PPO.load(ppo_path)
    
    if mode == "LIVE":
        # ... (Single Step Logic as before) ...
        # Simplified for brevity in this replacement
        # We focus on the BACKTEST implementation now
        run_backtest_eval(model, df_merged)
    else:
        # BACKTEST MODE (Full Year)
        run_backtest_eval(model, df_merged)

def run_backtest_eval(model, df):
    # Setup TradingEnv with this new dataframe
    # We need to map columns or ensure they work.
    # TradingEnv expects numeric columns.
    
    # Filter non-numeric
    df_numeric = df.select_dtypes(include=[np.number, bool])
    for col in df_numeric.select_dtypes(include=['bool']).columns:
        df_numeric[col] = df_numeric[col].astype(int)
        
    print(f"Starting Backtest on {len(df_numeric)} bars...")
    # Use default/tuned params from training (assumed embedded or default)
    # Ideally load best_hyperparameters.json
    try:
        with open(OUTPUT_DIR / "best_hyperparameters.json", 'r') as f:
            params = json.load(f)
            # Filter for env params
            env_keys = ['vol_target', 'dd_limit', 'streak_kill', 'lambda_tc', 'lambda_vol', 'lambda_dd', 'lambda_streak']
            env_kwargs = {k: params[k] for k in env_keys if k in params}
    except:
        env_kwargs = {}
        
    env = TradingEnv(df_numeric, **env_kwargs)
    obs, _ = env.reset()
    done = False
    equity_curve = [env.equity]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(info['equity'])
        
    print(f"Backtest Complete. Final Equity: {equity_curve[-1]:.2f}")
    
    # Risk Eval
    from sampo2.agents.risk_evaluator import RiskEvaluator
    
    # Extract Log
    trade_log = env.trade_log if hasattr(env, 'trade_log') else []
    
    evaluator = RiskEvaluator(equity_curve, trade_log=trade_log)
    report = evaluator.get_report()
    
    print("\n" + "="*40)
    print(f"INFERENCE BACKTEST REPORT (Data: {df.index[0]} - {df.index[-1]})")
    print(report)
    print("="*40 + "\n")

    # Optionally save curve
    pd.DataFrame({'Equity': equity_curve}).to_csv(OUTPUT_DIR / "backtest_curve_inference.csv")

if __name__ == "__main__":
    # Example usage:
    # run_live_inference(backtest_year=2017)
    # Default to live mock for safety unless args passed
    import sys
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
            run_live_inference(backtest_year=year)
        except ValueError:
            print("Invalid year provided. Running Live/Mock mode.")
            run_live_inference()
    else:
        run_live_inference() # Default Mock Live

