
import unittest
import pandas as pd
import numpy as np
from . import backtest


class TestResult(unittest.TestCase):
    def test_fill_estimate(self):
        self.assertEqual(2, backtest.Result.estimate_price(0, 2, 4, 1, 3))
        self.assertEqual(1, backtest.Result.estimate_price(0.2, 2, 4, 1, 3))
        self.assertEqual(4, backtest.Result.estimate_price(0.8, 2, 4, 1, 3))
        self.assertEqual(3, backtest.Result.estimate_price(1, 2, 4, 1, 3))
        self.assertEqual(1.5, backtest.Result.estimate_price(0.1, 2, 4, 1, 3))
        self.assertEqual(3.5, backtest.Result.estimate_price(0.7, 2, 4, 1, 3))
        self.assertEqual(3.25, backtest.Result.estimate_price(0.95, 2, 4, 1, 3))

    def test_single_asset_cost(self):
        # generate data
        n_samples = 1000
        intraday = 20
        annual_sd = 0.35
        sample_sd = annual_sd / np.sqrt(252 * intraday)
        bdays = pd.bdate_range('2019-01-01', periods=n_samples)
        long_short = (2 * (np.random.rand(n_samples) > 0.5) - 1).reshape(-1, 1)

        inputs = {}
        price = np.cumprod(1 + np.random.normal(0, sample_sd, n_samples * intraday)).reshape(-1, intraday) * 200
        inputs['open'] = pd.DataFrame(price[:, 0], columns=['ABC'], index=bdays)
        inputs['high'] = pd.DataFrame(np.max(price, axis=1), columns=['ABC'], index=bdays)
        inputs['low'] = pd.DataFrame(np.min(price, axis=1), columns=['ABC'], index=bdays)
        inputs['close'] = pd.DataFrame(price[:, -1], columns=['ABC'], index=bdays)

        # test cost assumption
        costs = {'spreads': np.array([0.02 / 100]), 'comm': 0.005, 'delay': 0}
        res = backtest.Result('test', long_short, lookback=0, inputs=inputs, capital=50000, costs=costs)
        self.assertAlmostEqual(res.stats['netSLP'], 0, 9)
        self.assertNotAlmostEqual(res.stats['Spread'], 0, 3)
        self.assertNotAlmostEqual(res.stats['Comm'], 0, 3)

        costs = {'spreads': np.array([0.02 / 100]), 'comm': 0, 'delay': 15 / 6.5 / 60}
        res = backtest.Result('test', long_short, lookback=0, inputs=inputs, capital=50000, costs=costs)
        self.assertNotAlmostEqual(res.stats['netSLP'], 0, 3)
        self.assertNotAlmostEqual(res.stats['Spread'], 0, 3)
        self.assertAlmostEqual(res.stats['Comm'], 0, 9)

        costs = {'spreads': np.array([0]), 'comm': 0.005, 'delay': 15 / 6.5 / 60}
        res = backtest.Result('test', long_short, lookback=0, inputs=inputs, capital=50000, costs=costs)
        self.assertNotAlmostEqual(res.stats['netSLP'], 0, 3)
        self.assertAlmostEqual(res.stats['Spread'], 0, 9)
        self.assertNotAlmostEqual(res.stats['Comm'], 0, 3)

        costs = {'spreads': np.array([0.02 / 100]), 'comm': 0.005, 'delay': 15 / 6.5 / 60}
        res = backtest.Result('test', long_short, lookback=0, inputs=inputs, capital=50000, costs=costs)
        self.assertTrue(np.all(res.comm_cost >= 0))
        self.assertTrue(np.all(res.spread_cost >= 0))

    def test_single_asset_position(self):
        # generate data
        n_samples = 1000
        capital = 50000
        bdays = pd.bdate_range('2019-01-01', periods=n_samples)

        inputs = {}
        price = np.ones((n_samples, 1)) * 200
        inputs['open'] = pd.DataFrame(price[:, 0], columns=['ABC'], index=bdays)
        inputs['high'] = pd.DataFrame(np.max(price, axis=1), columns=['ABC'], index=bdays)
        inputs['low'] = pd.DataFrame(np.min(price, axis=1), columns=['ABC'], index=bdays)
        inputs['close'] = pd.DataFrame(price[:, -1], columns=['ABC'], index=bdays)

        buy_and_hold = np.ones(n_samples).reshape(-1, 1)
        costs = {'spreads': np.array([0]), 'comm': 0, 'delay': 0}
        res = backtest.Result('test', buy_and_hold, lookback=0, inputs=inputs, capital=capital, costs=costs)
        self.assertTrue(np.all(res.equity == capital))
        self.assertEqual(np.sum(res.traded_shares), 250)


if __name__ == '__main__':
    unittest.main()