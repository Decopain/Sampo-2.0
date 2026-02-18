
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sampo2.env.trading_env import TradingEnv
from sampo2.env.actions import *

class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='H')
        self.df = pd.DataFrame({
            'Open': np.random.rand(100) + 100,
            'High': np.random.rand(100) + 105,
            'Low': np.random.rand(100) + 95,
            'Close': np.random.rand(100) + 100,
            'volume': np.random.rand(100) * 1000
        }, index=dates)
        
        self.env = TradingEnv(self.df, window_size=5)

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(len(obs), self.env.obs_shape)
        self.assertEqual(self.env.current_step, 5)
        self.assertEqual(self.env.balance, 10000)

    def test_step_buy(self):
        self.env.reset()
        action = ACTION_BUY
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertEqual(self.env.position, 1)
        self.assertLess(self.env.balance, 10000) # Commission paid
        self.assertFalse(done)

    def test_full_episode(self):
        self.env.reset()
        done = False
        steps = 0
        while not done:
            action = self.env.action_space.sample()
            obs, reward, done, truncated, info = self.env.step(action)
            steps += 1
            if steps > 200: # Safety break
                break
        
        self.assertTrue(done)
        self.assertEqual(self.env.current_step, len(self.df) - 1)

if __name__ == '__main__':
    unittest.main()
