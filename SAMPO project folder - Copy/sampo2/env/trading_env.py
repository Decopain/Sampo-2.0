
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment with Continuous Sizing and Multi-Metric Risk Framework.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission=0.0002, window_size=10, 
                 reward_scaling=1e-4, 
                 vol_target=0.01,
                 dd_limit=0.05,
                 dd_kill=0.15,
                 streak_kill=5,
                 lambda_tc=0.1,
                 lambda_vol=0.5,
                 lambda_dd=1.0,
                 lambda_streak=0.5,
                 # Execution Logic Config
                 conf_flip=0.6,
                 cooldown_steps=5,
                 profit_buffer=0.0005, # 5bps buffer above costs
                 t_profit_steps=20,    # Close if green & stagnant
                 max_hold_steps=100,   # Hard time exit
                 close_bonus=0.1,      # Reward bonus for profitable close
                 atr_window=14,
                 k_tp=2.0,
                 k_sl=1.0,
                 k_trl=1.25,
                 inactivity_penalty=0.0,
                 obs_cols=None):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.obs_cols = obs_cols
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.inactivity_penalty = inactivity_penalty
        
        # Risk Parameters
        self.vol_target = vol_target
        self.dd_limit = dd_limit
        self.dd_kill = dd_kill
        self.streak_kill = streak_kill
        
        # Penalty Weights
        self.lambda_tc = lambda_tc
        self.lambda_vol = lambda_vol
        self.lambda_dd = lambda_dd
        self.lambda_streak = lambda_streak
        
        # Execution Config
        self.conf_flip = conf_flip
        self.cooldown_steps = cooldown_steps
        self.profit_buffer = profit_buffer
        self.t_profit_steps = t_profit_steps
        self.max_hold_steps = max_hold_steps
        self.close_bonus = close_bonus
        
        # Dynamic Exit Config
        self.atr_window = atr_window
        self.k_tp = k_tp
        self.k_sl = k_sl
        self.k_trl = k_trl
        
        # --- Action Space: Continuous Target Exposure ---
        # -1.0 (Full Short) to 1.0 (Full Long)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # --- Observation Space ---
        # Features + Account State (Balance, Position, Drawdown, Streak) + Exec State
        # New: Unrealized PnL, Holding Steps, Costs Paid, Dist TP, Dist SL, Dist Trail, Chop Flag
        if self.obs_cols:
            self.n_features = len(self.obs_cols)
        else:
            self.n_features = len(df.columns)
            
        self.n_exec_features = 7 
        self.obs_shape = (self.window_size * self.n_features) + 4 + self.n_exec_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
        # Init State
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0 # Float exposure
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        
        # Track history for live metrics
        self.equity_history = [self.initial_balance]
        
        # Risk State
        self.loss_streak = 0
        self.running_returns = [] # Track returns for vol measurement
        
        # Execution State
        self.cooldown_remaining = 0
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.highest_price = 0.0 # For Trailing Stop
        self.lowest_price = 0.0
        self.holding_steps = 0
        self.costs_paid_current_trade = 0.0
        
        # Overrides Logging
        self.agent_action = 0.0
        self.executed_action = 0.0
        self.override_reason = "None"
        
        self.trade_log = []
        self.current_trade_info = None # Track active trade details
        
        observation = self._get_observation()
        info = { 
            "trade_log": self.trade_log, 
            "equity_history": self.equity_history 
        }
        return observation, info

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        
        end = self.current_step
        
        if self.obs_cols:
             window = self.df.iloc[start:end][self.obs_cols].values
        else:
             window = self.df.iloc[start:end].values
             
        flattened_window = window.flatten().astype(np.float32)
        
        # Risk Context
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        norm_streak = min(self.loss_streak / self.streak_kill, 1.0)
        
        state = np.array([self.balance, self.position, current_dd, norm_streak], dtype=np.float32)
        
        # Execution Context Features
        # 1. Unrealized PnL (Pct)
        if self.current_trade_info:
             entry_price = self.current_trade_info['entry_price']
             current_price = self.df.iloc[self.current_step]['Close']
             if self.position > 0:
                 unrealized_pnl = (current_price - entry_price) / entry_price
             else:
                 unrealized_pnl = (entry_price - current_price) / entry_price
        else:
            unrealized_pnl = 0.0
            
        # 2. Holding Steps (normalized)
        norm_holding = min(self.holding_steps / self.max_hold_steps, 1.0)
        
        # 3. Costs Paid (pct of equity estimate)
        cost_impact = self.costs_paid_current_trade / self.equity
        
        # 4. Dist to TP/SL/Trail (in ATR units)
        current_atr = self.df.iloc[self.current_step]['atr'] if 'atr' in self.df.columns else self.entry_atr
        if current_atr == 0: current_atr = 1.0 # Avoid Div0
        
        dist_tp = 0.0
        dist_sl = 0.0
        dist_trail = 0.0
        
        current_price = self.df.iloc[self.current_step]['Close']
        
        if self.position > 0:
             tp_price = self.entry_price + (self.k_tp * self.entry_atr)
             sl_price = self.entry_price - (self.k_sl * self.entry_atr)
             trail_price = self.highest_price - (self.k_trl * self.entry_atr)
             
             dist_tp = (tp_price - current_price) / current_atr
             dist_sl = (current_price - sl_price) / current_atr
             dist_trail = (current_price - trail_price) / current_atr
             
        elif self.position < 0:
             tp_price = self.entry_price - (self.k_tp * self.entry_atr)
             sl_price = self.entry_price + (self.k_sl * self.entry_atr)
             trail_price = self.lowest_price + (self.k_trl * self.entry_atr)
             
             dist_tp = (current_price - tp_price) / current_atr
             dist_sl = (sl_price - current_price) / current_atr
             dist_trail = (trail_price - current_price) / current_atr
             
        # 7. Chop Flag (From Renko Prob RY/BY)
        chop_prob = 0.0
        if 'renko_prob_ry' in self.df.columns and 'renko_prob_by' in self.df.columns:
            chop_prob = self.df.iloc[self.current_step]['renko_prob_ry'] + self.df.iloc[self.current_step]['renko_prob_by']
            
        exec_state = np.array([unrealized_pnl, norm_holding, cost_impact, dist_tp, dist_sl, dist_trail, chop_prob], dtype=np.float32)
        
        return np.concatenate((flattened_window, state, exec_state))

    def step(self, action):
        prev_equity = self.equity
        prev_position = self.position
        
        # 1. Action Interpretation
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        self.agent_action = raw_action
        self.override_reason = "None"
        
        # Risk Throttling (Legacy)
        risk_multiplier = 0.8 ** self.loss_streak
        target_exposure = raw_action * risk_multiplier
        
        # --- 2. EXECUTION OVERLAYS ---
        current_price = self.df.iloc[self.current_step]['Close']
        current_time = self.df.index[self.current_step]
        current_atr = self.df.iloc[self.current_step]['atr'] if 'atr' in self.df.columns else self.entry_atr
        if current_atr == 0: current_atr = current_price * 0.01 # Fallback
        
        # Market Confidence & Regime
        confidence = 1.0
        p_blue = 0.5
        p_red = 0.5
        if 'renko_prob_blue' in self.df.columns:
            p_blue = self.df.iloc[self.current_step]['renko_prob_blue']
            p_red = self.df.iloc[self.current_step]['renko_prob_red']
            confidence = max(p_blue, p_red)
            
        # Regime Flags
        chop_flag = False
        if 'renko_prob_ry' in self.df.columns: 
             chop_intensity = self.df.iloc[self.current_step]['renko_prob_ry'] + self.df.iloc[self.current_step]['renko_prob_by']
             if chop_intensity > 0.5: chop_flag = True
             
        # Regime Direction (derived from probs)
        regime_bullish = p_blue > p_red
        regime_bearish = p_red > p_blue
             
        # A) EXIT OVERLAYS (Force Close)
        force_close = False
        
        # Only check exits if we have a position
        if abs(self.position) > 0.01:
            # 1. Dynamic TP/SL
            if self.position > 0:
                if current_price >= self.entry_price + (self.k_tp * self.entry_atr): 
                    force_close = True; self.override_reason = "TP_Long"
                elif current_price <= self.entry_price - (self.k_sl * self.entry_atr):
                    force_close = True; self.override_reason = "SL_Long"
            else: # Short
                if current_price <= self.entry_price - (self.k_tp * self.entry_atr):
                    force_close = True; self.override_reason = "TP_Short"
                elif current_price >= self.entry_price + (self.k_sl * self.entry_atr):
                    force_close = True; self.override_reason = "SL_Short"
                    
            # 2. Trailing Stop
            if not force_close:
                if self.position > 0:
                    if current_price <= self.highest_price - (self.k_trl * self.entry_atr):
                         force_close = True; self.override_reason = "Trail_Long"
                else:
                    if current_price >= self.lowest_price + (self.k_trl * self.entry_atr):
                         force_close = True; self.override_reason = "Trail_Short"
                         
            # 3. Time Exit
            if not force_close:
                # Max Hold
                if self.holding_steps >= self.max_hold_steps:
                    force_close = True; self.override_reason = "Time_Max"
                # Green + Stagnant
                elif self.holding_steps >= self.t_profit_steps:
                    # Check Unrealized PnL vs Costs
                    unrealized_pnl = 0
                    if self.position > 0: unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                    else: unrealized_pnl = (self.entry_price - current_price) / self.entry_price
                    
                    # Accumulate estimate of closing cost to be safe
                    estimated_close_cost = abs(self.position) * self.commission
                    cost_thresh = (self.costs_paid_current_trade/self.equity) + estimated_close_cost + self.profit_buffer
                    
                    if unrealized_pnl > cost_thresh:
                        force_close = True; self.override_reason = "Time_Profit"
                        
            # 4. Regime Exits
            if not force_close:
                # A. Risk Exit: Regime against position
                if (self.position > 0 and regime_bearish) or (self.position < 0 and regime_bullish):
                     force_close = True; self.override_reason = "Regime_Risk"
                
                # B. Profit Capture in Chop
                if not force_close and chop_flag:
                     unrealized_pnl = 0
                     if self.position > 0: unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                     else: unrealized_pnl = (self.entry_price - current_price) / self.entry_price
                     
                     estimated_close_cost = abs(self.position) * self.commission
                     cost_thresh = (self.costs_paid_current_trade/self.equity) + estimated_close_cost + self.profit_buffer
                     
                     if unrealized_pnl > cost_thresh:
                         force_close = True; self.override_reason = "Regime_Profit"
        
        # Apply Forced Exit
        if force_close:
            target_exposure = 0.0
            
        # B) FLIP GATE & COOLDOWN (Only if NOT forced close)
        if not force_close:
            is_reversal = (np.sign(target_exposure) != np.sign(self.position)) and (abs(target_exposure) > 0.01) and (abs(self.position) > 0.01)
            
            if is_reversal:
                if self.cooldown_remaining > 0:
                    target_exposure = self.position # Block flip, Hold In
                    self.override_reason = "Cooldown_Block"
                elif confidence < self.conf_flip:
                    target_exposure = self.position # Block flip, Hold In
                    self.override_reason = "Conf_Block"
        
        self.executed_action = target_exposure
        
        # --- 3. MARKET UPDATE & ACCOUNTING ---
        
        # Decrement Cooldown
        self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
        
        # EOE Check omitted for brevity (handled by done=True later)

        next_price = self.df.iloc[self.current_step + 1]['Close']
        price_change_pct = (next_price - current_price) / current_price
        
        gross_return = target_exposure * price_change_pct
        gross_pnl = self.equity * gross_return
        
        # Transaction Costs (On Turnover)
        delta_pos = target_exposure - self.position
        turnover = abs(delta_pos)
        step_cost = turnover * self.equity * self.commission
        
        # Net PnL
        net_pnl = gross_pnl - step_cost
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # Update Balance
        self.balance = self.equity 
        
        # --- TRADE MANAGEMENT (Proportionate Attribution) ---
        tolerance = 0.01
        prev_flat = abs(self.position) < tolerance
        curr_flat = abs(target_exposure) < tolerance
        flipped = (np.sign(target_exposure) != np.sign(self.position)) and (not prev_flat) and (not curr_flat)
        
        trade_closed = False
        realized_pnl_net = 0.0
        
        # Calculate Fractions of Step Cost
        close_fraction = 0.0
        open_fraction = 0.0
        if turnover > 1e-9:
             if flipped or (not prev_flat and curr_flat):
                 # Closing existing
                 close_amt = abs(self.position)
                 close_fraction = close_amt / turnover
                 open_fraction = 1.0 - close_fraction
             elif prev_flat and not curr_flat:
                 open_fraction = 1.0
             else:
                 # Resizing in same direction, treat as 'open' or maintenance
                 open_fraction = 1.0

        # 1. Handle CLOSE (Closing Old)
        if (not prev_flat) and (curr_flat or flipped):
            # Allocate cost
            close_cost_allocated = step_cost * close_fraction
            
            # Update accumulated costs (final tally for this trade)
            self.costs_paid_current_trade += close_cost_allocated
            total_trade_costs = self.costs_paid_current_trade
            
            if self.current_trade_info:
                # Reconstruct Exit Equity:
                # Equity NOW has (Close+Open) deducted (step_cost).
                # We add back Open Cost portion to isolate "Equity at Close".
                open_cost_allocated = step_cost * open_fraction
                equity_post_close = self.equity + open_cost_allocated 
                
                trade_pnl_net = equity_post_close - self.current_trade_info['entry_equity']
                realized_pnl_net = trade_pnl_net
                
                self.trade_log.append({
                    'Entry Time': self.current_trade_info['entry_time'],
                    'Exit Time': current_time,
                    'Type': self.current_trade_info['type'],
                    'Entry Price': self.current_trade_info['entry_price'],
                    'Exit Price': current_price,
                    'PnL': trade_pnl_net,
                    'Return': trade_pnl_net / self.current_trade_info['entry_equity'],
                    'Duration': self.holding_steps,
                    'Size': close_amt,
                    'Reason': self.override_reason if force_close else ('Flip' if flipped else 'Close')
                })
                trade_closed = True
                
                # Close Bonus Check
                entry_equity = self.current_trade_info['entry_equity']
                trade_ret = realized_pnl_net / entry_equity
                trade_cost_pct = total_trade_costs / entry_equity
                
                if trade_ret >= (trade_cost_pct + self.profit_buffer):
                    self.override_reason += "|BONUS" 
                
                # RESET STATE
                self.current_trade_info = None
                self.entry_price = 0.0
                self.entry_atr = 0.0
                self.highest_price = 0.0
                self.lowest_price = 0.0
                self.holding_steps = 0
                self.costs_paid_current_trade = 0.0

        # 2. Handle OPEN (Opening New)
        if (not curr_flat) and (prev_flat or flipped):
            # Allocate Cost
            open_cost_allocated = step_cost * open_fraction
            
            # Initialize New Trade
            self.current_trade_info = {
                'entry_step': self.current_step,
                'entry_time': current_time,
                'entry_equity': self.equity, # Equity implies "post open cost" basis (correct)
                'entry_price': current_price,
                'type': 'Long' if target_exposure > 0 else 'Short'
            }
            
            # Init Execution State
            self.entry_price = current_price
            self.entry_atr = current_atr
            self.highest_price = current_price
            self.lowest_price = current_price
            self.holding_steps = 0
            self.costs_paid_current_trade = open_cost_allocated 
            
            # Cooldown Trigger
            if flipped or prev_flat:
                self.cooldown_remaining = self.cooldown_steps
        
        # Accumulate costs if holding (resizing/maintenance)
        if not trade_closed and not ((not curr_flat) and (prev_flat or flipped)):
             self.costs_paid_current_trade += step_cost

        # Update Holding Stats
        if abs(target_exposure) > 0.01:
            self.holding_steps += 1
            # Update extremes using CURRENT price (Available for next step decision)
            if target_exposure > 0:
                 self.highest_price = max(self.highest_price, current_price)
            else:
                 self.lowest_price = min(self.lowest_price, current_price)
            
        self.position = target_exposure

        # 5. Risk Metrics Calculation
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        
        self.running_returns.append(net_pnl / prev_equity)
        if len(self.running_returns) > 20: self.running_returns.pop(0)
        rolling_vol = np.std(self.running_returns) if len(self.running_returns) > 5 else 0
        
        # Streak Logic (Simplied for Continuous)
        is_active = abs(prev_position) > 0.01
        if is_active and net_pnl < 0:
            self.loss_streak += 1
        elif is_active and net_pnl > 0:
            self.loss_streak = 0
            
        # 6. Reward Construction (Multi-Metric)
        step_return = np.log(self.equity / prev_equity)
        
        # Penalties
        pen_vol = self.lambda_vol * max(0, rolling_vol - self.vol_target)
        pen_dd = self.lambda_dd * max(0, current_dd - self.dd_limit)
        pen_streak = self.lambda_streak * (self.loss_streak / self.streak_kill)
        
        # Close Bonus (Reward for realizing profit > costs)
        bonus = 0.0
        if "BONUS" in self.override_reason:
            bonus = self.close_bonus
        
        reward = (step_return * self.reward_scaling * 100) - pen_vol - pen_dd - pen_streak + bonus
        
        # Inactivity Penalty
        if abs(target_exposure) < 0.01:
            reward -= self.inactivity_penalty * 10 # Scale up since reward is roughly -1 to 1
        
        # 7. Termination (Kill Switch)
        done = False
        truncated = False
        
        if current_dd > self.dd_kill:
            done = True
            reward -= 10 # Massive penalty
            print(f"Kill Switch: Max DD ({current_dd:.2%}) exceeded.")
            
        if self.loss_streak > self.streak_kill:
            done = True
            reward -= 10
            # print(f"Kill Switch: Loss Streak ({self.loss_streak}) exceeded.")
            
        # Advance
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            # Close any open trade at end
            if self.current_trade_info:
                 trade_pnl = self.equity - self.current_trade_info['entry_equity']
                 self.trade_log.append({
                    'Entry Time': self.current_trade_info['entry_time'],
                    'Exit Time': current_time,
                    'Type': self.current_trade_info['type'],
                    'Entry Price': self.current_trade_info['entry_price'],
                    'Exit Price': current_price,
                    'PnL': trade_pnl,
                    'Return': trade_pnl / self.current_trade_info['entry_equity'],
                    'Duration': self.current_step - self.current_trade_info['entry_step'],
                    'Reason': 'EOE'
                })
            
        info = {
            'equity': self.equity,
            'position': self.position,
            'dd': current_dd,
            'streak': self.loss_streak,
            'trades': self.trade_log,
            'equity_history': self.equity_history
        }
        
        # Track Equity
        self.equity_history.append(self.balance)

        return self._get_observation(), reward, done, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, DD: {(self.peak_equity-self.equity)/self.peak_equity:.1%}, Streak: {self.loss_streak}")
