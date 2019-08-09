
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
        print('\nRunning:')
        T = len(self.data)
        n_assets = self.data.shape[1]
        allowed_leverage = self.config['leverage'] + 1e-4
        timestamps = self.data.index
        result = ResultSet(self.data)
        for strategy in self.strategies:
            weights = np.zeros((T, n_assets))

            def run_strategy(idx):
                if strategy.timer.is_up(timestamps[idx]):
                    current_inputs = {k.alias: inputs[k.name][idx - 1] for k in strategy.indicators}
                    current_inputs['last_weights'] = weights[idx - 1]
                    current_inputs['n_assets'] = n_assets
                    for _ in range(strategy.rep):
                        weights[idx] += strategy.generate_weights(current_inputs)
                    if strategy.rep > 1:
                        weights[idx] /= strategy.rep
                else:
                    weights[idx] = weights[idx - 1]

                # check weights
                total_weight = np.sum(weights[idx])
                leverage = np.sum(np.abs(weights[idx]))
                if abs(total_weight - 1) > 1e-4:
                    raise RuntimeError(f'weights sum to {total_weight: .5f}')

                if leverage > allowed_leverage:
                    raise RuntimeError(f'leverage is {leverage: .5f}')

            utils.run_with_status(f'{strategy.name} rep={strategy.rep}',
                                  range(strategy.lookback + 1, T), T - strategy.lookback, run_strategy)
            result.add_performance(strategy.name, strategy.lookback, weights)

        print()
        return result
