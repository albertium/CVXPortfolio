from cvxport import utils
import riskparity
import numpy as np
import pandas as pd
from cvxport import backtest


# assets = ['eq_us', 'eq_exus', 'eq_em', 'tn_us', 'tb_us', 'bond_em', 'reit']
# rets = utils.get_prices(assets)
# print('Last update: %s' % rets.last_valid_index())
# # utils.plot_returns(rets)
#
# base_weights = riskparity.get_simple_parity_weights(rets)
# corr_weights = riskparity.get_correlation_adjustment(rets)
# final_weights = riskparity.adjust_weights(base_weights, [corr_weights])
# utils.plot_area([base_weights, final_weights])

n_samples = 1000
bdays = pd.bdate_range('2019-01-01', periods=n_samples)

inputs = {}
price = np.ones((n_samples, 1)) * 200
inputs['open'] = pd.DataFrame(price[:, 0], columns=['ABC'], index=bdays)
inputs['high'] = pd.DataFrame(np.max(price, axis=1), columns=['ABC'], index=bdays)
inputs['low'] = pd.DataFrame(np.min(price, axis=1), columns=['ABC'], index=bdays)
inputs['close'] = pd.DataFrame(price[:, -1], columns=['ABC'], index=bdays)

buy_and_hold = np.ones(n_samples).reshape(-1, 1)
costs = {'spreads': np.array([0]), 'comm': 0, 'delay': 0}
res = backtest.Result('test', buy_and_hold, lookback=0, inputs=inputs, capital=50000, costs=costs)
print(res.stats)
