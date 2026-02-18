
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from pathlib import Path

# --- CONFIG ---
STATUS_FILE = "current_training_status.json"
HISTORY_FILE = "monitor_history.csv" # Local cache for dashboard history

st.set_page_config(
    page_title="SAMPO2 Training Monitor",
    page_icon="🏋️",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .stProgress > div > div > div > div {
        background-color: #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🏋️ SAMPO2 Production Training Monitor")
st.markdown("Watching **Trial 25 Champion** train for **5 Million Steps**.")

# --- LOAD STATUS ---
def load_status():
    if not os.path.exists(STATUS_FILE):
        return None
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

status = load_status()

if status is None:
    st.warning("⚠️ Waiting for Training Pulse... (Check back in 60 seconds)")
    st.stop()

# --- HISTORY MANAGEMENT ---
# Since agent overwrites status file, we must accumulate history ourselves
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Step', 'Equity', 'Progress', 'Trades'])

# Check for new data point
last_step = st.session_state.history['Step'].iloc[-1] if not st.session_state.history.empty else -1

if status['step'] > last_step:
    new_row = {
        'Time': pd.Timestamp.now(),
        'Step': status['step'],
        'Equity': status['equity'],
        'Progress': status['progress'],
        'Trades': status['trade_count']
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Progress", f"{status['progress']*100:.2f}%", f"{status['step']:,} / {status['total_steps']:,}")
    st.progress(status['progress'])

with col2:
    start_equity = 10000.0 # Assumption
    current_equity = status['equity']
    roi = (current_equity - start_equity) / start_equity
    st.metric("Current Equity", f"${current_equity:,.2f}", f"{roi:+.2%}")

with col3:
    st.metric("Total Trades", f"{status['trade_count']}")

with col4:
    # Estimate ETA
    # Simple calculation based on collected history speed
    if len(st.session_state.history) > 2:
        recent = st.session_state.history.iloc[-5:] # Last 5 points
        avg_steps_per_sec = (recent['Step'].diff().mean()) / (recent['Time'].diff().dt.total_seconds().mean())
        if avg_steps_per_sec > 0:
            remaining_steps = status['total_steps'] - status['step']
            remaining_seconds = remaining_steps / avg_steps_per_sec
            eta_hours = remaining_seconds / 3600
            st.metric("Estimated Time Remaining", f"{eta_hours:.1f} Hours", f"{avg_steps_per_sec:.1f} steps/s")
        else:
            st.metric("ETA", "Calculating...")
    else:
        st.metric("ETA", "Gathering Speed Data...")

# --- CHARTS ---
st.markdown("---")
st.subheader("real-Time Performance")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Equity Curve (Session)")
    if not st.session_state.history.empty:
        st.line_chart(st.session_state.history.set_index('Step')['Equity'])
    else:
        st.info("Equity chart will build here as data arrives...")

with chart_col2:
    st.markdown("### Trade Volume Accumulation")
    if not st.session_state.history.empty:
         st.area_chart(st.session_state.history.set_index('Step')['Trades'])

# --- AUTO REFRESH ---
if st.checkbox("Auto-Refresh (5s)", value=True):
    time.sleep(5)
    st.rerun()

# --- FOOTER ---
st.markdown("---")
st.caption(f"Last Pulse: {status.get('last_updated', 'Unknown')} | Keep this tab open to build history chart.")
