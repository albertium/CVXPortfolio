
import numpy as np
import pandas as pd
import utils
from typing import Iterable, Dict
import abc


class Indicator(abc.ABC):
    def __init__(self, name, alias, lookback):
        self.name = name
        self.alias = alias
        self.lookback = lookback

    @abc.abstractmethod
    def process(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Strategy(abc.ABC):
    def __init__(self, name, indicators: Iterable[Indicator] = None):
        self.name = name
        self.lookback = 0 if indicators is None else max([x.lookback for x in indicators])
        self.indicators = indicators if indicators is not None else []

    @abc.abstractmethod
    def generate_weights(self, inputs: Dict):
        pass

    def __str__(self):
        return '<STRATEGY> lookback=%d; %s' % (self.lookback, self.name)


class Result:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.all_weights = {}
        self.lookbacks = {}
        self.returns = pd.DataFrame(index=data.index)
        self.performances = pd.DataFrame(index=data.index)
        self.stats = pd.DataFrame(columns=['Sharpe', 'Return', 'SD'])

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
        self.stats = self.stats.append(pd.Series([sharpe, ret, sd], index=self.stats.columns, name=name))

    def plot(self):
        utils.plot_lines(self.performances.iloc[max(self.lookbacks.values()):])

    def show(self):
        pass


class BackTester:
    def __init__(self, data: pd.DataFrame, strategies: Iterable[Strategy]):
        self.data = data
        self.strategies = strategies
        self.indicators = list(set(sum([p.indicators for p in strategies], [])))

    def run(self):
        print('Running backtest on:')
        [print('\t %s' % x) for x in self.strategies]
        # prepare indicators
        inputs = {}
        for indicator in self.indicators:
            inputs[indicator.name] = indicator.process(self.data)

        # main loop
        print('\nRunning:')
        T = len(self.data)
        n_asset = self.data.shape[1]
        result = Result(self.data)
        for strategy in self.strategies:
            weights = np.zeros((T, n_asset))
            for idx in range(strategy.lookback + 1, T):
                current_inputs = {k.alias: inputs[k.name][idx - 1] for k in strategy.indicators}
                current_inputs['last_weights'] = weights[idx - 1]
                current_inputs['n_asset'] = n_asset
                weights[idx] = strategy.generate_weights(current_inputs)

            result.add_performance(strategy.name, strategy.lookback, weights)
            print('\t%-20s: Done' % strategy.name)

        return result


class SingleAssetStrategy(Strategy):
    def __init__(self, asset_name, idx):
        self.idx = idx
        super(SingleAssetStrategy, self).__init__('single=%s' % asset_name)

    def generate_weights(self, inputs: Dict):
        weights = np.zeros(inputs['n_asset'])
        weights[self.idx] = 1
        return weights


class VolatilityIndicator(Indicator):
    def __init__(self, name, lookback=90):
        super(VolatilityIndicator, self).__init__('volatility=%d' % lookback, name, lookback)

    def process(self, data: pd.DataFrame):
        return data.rolling(window=self.lookback).std().to_numpy()


class InverseVolatilityStrategy(Strategy):
    def __init__(self, lookback=90):
        ind = VolatilityIndicator('vol', lookback)
        super(InverseVolatilityStrategy, self).__init__('inverse_vol=%d' % lookback, [ind])

    def generate_weights(self, inputs: Dict):
        inv = 1 / inputs['vol']
        return inv / np.sum(inv)
