import streamlit as st
import pandas as pd
import json
import time
import datetime
import os
import plotly.express as px
from pathlib import Path

# --- CONFIG ---
LOG_FILE = "tuning_log.json"
STATUS_FILE = "current_trial.json"
CONTROL_FILE = "tuning_control.json"
REFRESH_RATE = 2.0

st.set_page_config(page_title="SAMPO2 Tuning Dashboard", layout="wide", page_icon="🚀")

# --- UTILS ---
def load_data():
    """Load historical trial data."""
    if not Path(LOG_FILE).exists():
        return pd.DataFrame()
    try:
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

def load_status():
    """Load current trial status."""
    if not Path(STATUS_FILE).exists():
        return None
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def set_control(command):
    with open(CONTROL_FILE, 'w') as f:
        json.dump({"command": command}, f)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # Status Indicator
    status_data = load_status()
    is_active = status_data is not None
    
    # Check if stale
    if is_active and 'last_updated' in status_data:
        last_upd = datetime.datetime.fromisoformat(status_data['last_updated'])
        if (datetime.datetime.now() - last_upd).total_seconds() > 30:
            st.error("⚠️ SIGNAL LOST")
            st.caption("Backend stopped updating.")
            is_active = False
        else:
            st.success("🟢 ONLINE")
    else:
        st.warning("⚪ OFFLINE")

    # Buttons
    c1, c2, c3 = st.columns(3)
    if c1.button("▶️ Start"): set_control("run")
    if c2.button("⏸️ Pause"): set_control("pause")
    if c3.button("🛑 Stop"): set_control("stop")
    
    st.divider()
    
    # Timer
    if is_active and 'start_time' in status_data:
        start_ts = status_data['start_time']
        start_dt = datetime.datetime.fromtimestamp(start_ts)
        elapsed = datetime.datetime.now() - start_dt
        st.metric("⏱️ Runtime", str(elapsed).split('.')[0])

# --- MAIN LAYOUT ---
st.title("🚀 SAMPO2 Hyperparameter Tuning")
st.caption("Monitoring Optimization of PPO Agent & Execution Overlays")

# 1. LIVE PROGRESS
if is_active:
    st.subheader("🟢 Live Trial Progress")
    
    # Top Row: Trial Info
    c1, c2, c3 = st.columns([1, 2, 1])
    current_trial = status_data.get('trial_number', 0)
    total_trials = status_data.get('total_trials', 50)
    
    c1.metric("Trial", f"#{current_trial} / {total_trials}")
    
    steps_done = status_data.get('current_step', 0)
    total_steps = status_data.get('total_steps', 200000)
    
    prog_val = min(steps_done / total_steps, 1.0) if total_steps > 0 else 0
    c2.progress(prog_val, text=f"Step Progress: {steps_done} / {total_steps}")
    
    best_sharpe = status_data.get('best_sharpe', 0.0)
    c3.metric("Best Sharpe So Far", f"{best_sharpe:.4f}")
    
    # Second Row: Metrics
    if 'current_metrics' in status_data:
        m = status_data['current_metrics']
        cols = st.columns(6)
        cols[0].metric("Return", f"{m.get('return', 0):.2%}")
        cols[1].metric("Sharpe", f"{m.get('sharpe', 0):.2f}")
        cols[2].metric("MaxDD", f"{m.get('max_dd', 0):.2%}", delta_color="inverse")
        cols[3].metric("Win Rate", f"{m.get('win_rate', 0):.1%}")
        cols[4].metric("Trades", m.get('trades', 0))
        cols[5].metric("Turnover", f"{m.get('turnover', 0):.1f}x")

else:
    st.info("Waiting for Tuning Process to Start...")

st.divider()

# 2. COMPLETED TRIALS
df = load_data()
if not df.empty:
    st.subheader("📊 Completed Trials Analysis")
    
    # Styled Table (Try Gradient, Fallback to Progress Config)
    st.caption("Detailed Performance Log")
    try:
        # User requested exact restoration of gradient
        st.dataframe(
            df.sort_values("Sharpe", ascending=False).style.background_gradient(subset=['Sharpe'], cmap='Greens'),
            use_container_width=True
        )
    except Exception as e:
        # Fallback if matplotlib/pandas incompatibility exists
        st.dataframe(
            df.sort_values("Sharpe", ascending=False),
            use_container_width=True,
            column_config={
                "Sharpe": st.column_config.ProgressColumn("Sharpe", format="%.2f", min_value=0, max_value=3),
                "Return": st.column_config.NumberColumn("Return", format="%.2f%%"),
                "MaxDD": st.column_config.NumberColumn("MaxDD", format="%.2f%%"),
            }
        )

    # Tabs
    tab1, tab2 = st.tabs(["📈 Risk/Reward", "📋 Full Log"])
    with tab1:
        fig = px.scatter(df, x="MaxDD", y="Return", color="Sharpe", size="Trade Count", 
                         title="Pareto Frontier (Return vs Risk)", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.dataframe(df)

# Auto Refresh
time.sleep(REFRESH_RATE)
if hasattr(st, 'rerun'):
    st.rerun()
else:
    st.experimental_rerun()
