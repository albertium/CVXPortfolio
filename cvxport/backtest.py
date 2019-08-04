
import pandas as pd
import numpy as np
from typing import Iterable
from . import utils
from .strategy import Strategy


class ResultSet:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.all_weights = {}
        self.lookbacks = {}
        self.returns = pd.DataFrame(index=data.index)
        self.performances = pd.DataFrame(index=data.index)
        self.stats = pd.DataFrame(columns=['Sharpe', 'Return', 'SD', 'maxDD'])

    def add_performance(self, name, lookback, weights):
        self.all_weights[name] = weights
        self.lookbacks[name] = lookback
        self.returns[name] = np.sum(self.data * weights, axis=1)
        self.performances[name] = (self.returns[name] + 1).cumprod()

        # add statistics
        series = self.performances[name].to_numpy()
        ret = np.power(series[-1] / series[0], 252 / len(series)) - 1
        sd = np.std(self.returns.to_numpy()) * np.sqrt(252)
        sharpe = ret / sd
        dd = self.calculate_drawdown(series)

        self.stats = self.stats.append(pd.Series([sharpe, ret, sd, dd], index=self.stats.columns, name=name))

    def plot(self):
        utils.plot_lines(self.performances.iloc[max(self.lookbacks.values()):])

    def show(self):
        utils.pretty_print(self.stats, ['{:.2f}', '{:.2%}', '{:.2%}', '{:.2%}'])

    @classmethod
    def calculate_drawdown(cls, ts):
        peak = ts[0]
        dd = 0
        for x in ts:
            peak = max(peak, x)
            dd = min(dd, min(x / peak - 1, 0))
        return dd


class BackTester:
    def __init__(self, data: pd.DataFrame, strategies: Iterable[Strategy]):
        self.data = data
        self.strategies = strategies
        self.indicators = list(set(sum([p.indicators for p in strategies], [])))

    def run(self):
        # prepare indicators
        inputs = {}
        for indicator in self.indicators:
            inputs[indicator.name] = indicator.process(self.data)

        # main loop
        print('\nRunning:')
        T = len(self.data)
        n_asset = self.data.shape[1]
        result = ResultSet(self.data)
        for strategy in self.strategies:
            weights = np.zeros((T, n_asset))
            for idx in range(strategy.lookback + 1, T):
                current_inputs = {k.alias: inputs[k.name][idx - 1] for k in strategy.indicators}
                current_inputs['last_weights'] = weights[idx - 1]
                current_inputs['n_asset'] = n_asset
                weights[idx] = strategy.generate_weights(current_inputs)

            result.add_performance(strategy.name, strategy.lookback, weights)
            print('\t%-20s: Done' % strategy.name)

        print()
        return result
