
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import Iterable
import time
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
        self.risk_contributions = {}

    def add_performance(self, name, lookback, weights):
        self.all_weights[name] = weights
        self.lookbacks[name] = lookback
        self.returns[name] = np.sum(self.data * weights, axis=1)
        self.performances[name] = (self.returns[name] + 1).cumprod()
        self.risk_contributions[name] = utils.get_risk_contribution(self.data, weights)

        # add statistics
        series = self.performances[name].to_numpy()
        ret = np.power(series[-1] / series[0], 252 / len(series)) - 1
        sd = np.std(self.returns.to_numpy()) * np.sqrt(252)
        sharpe = ret / sd
        dd = self.calculate_drawdown(series)

        self.stats = self.stats.append(pd.Series([sharpe, ret, sd, dd], index=self.stats.columns, name=name))

    def plot(self, plot_rc=False):
        chopped = self.returns.iloc[max(self.lookbacks.values()):].copy()
        chopped.iloc[0] = 0
        utils.plot_lines((chopped + 1).cumprod())

        if plot_rc:
            for name, data in self.risk_contributions.items():
                utils.plot_area(name, data)

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


def task(_args):
    _start = time.time()
    _strategy, _data, _inputs, _max_lev = _args
    weights = _strategy.run(_data, _inputs, _max_lev)
    print(f'{_strategy.name} finished [{time.time() - _start: 5.1f}s]')
    return _strategy.name, _strategy.lookback, weights


class BackTester:
    def __init__(self, data: pd.DataFrame, strategies: Iterable[Strategy], config=None):
        self.data = data
        self.strategies = strategies
        self.indicators = list(set(sum([p.indicators for p in strategies], [])))
        self.config = {'leverage': 1} if config is None else config

    def run(self):
        # prepare indicators
        inputs = {}
        for indicator in self.indicators:
            inputs[indicator.name] = indicator.process(self.data)

        # main loop
        print('\nRunning')
        max_leverage = self.config['leverage'] + 1e-4
        result = ResultSet(self.data)

        batches = [[x, self.data, inputs, max_leverage] for x in self.strategies]
        with Pool(processes=3) as p:
            result_all = p.map(task, batches)

        for res in result_all:
            name, lb, wgts = res
            result.add_performance(name, lb, wgts)

        # for strategy in self.strategies:
        #     print(f'{strategy.name}: ', end='')
        #     weights = strategy.run(self.data, inputs, max_leverage)
        #     result.add_performance(strategy.name, strategy.lookback, weights)
        #     print('done')

        print()
        return result
