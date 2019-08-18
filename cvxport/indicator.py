
import pandas as pd
import numpy as np
import abc


class Indicator(abc.ABC):
    def __init__(self, name, alias, lookback):
        self.name = name
        self.alias = alias
        self.lookback = lookback

    @abc.abstractmethod
    def process(self, inputs: dict) -> np.ndarray:
        pass

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class RollingVolatility(Indicator):
    def __init__(self, name, lookback=90):
        super(RollingVolatility, self).__init__('roll_vol=%d' % lookback, alias=name, lookback=lookback)

    def process(self, data: pd.DataFrame):
        return data.rolling(window=self.lookback).std().to_numpy()


class RollingCovariance(Indicator):
    def __init__(self, name, lookback=90):
        super(RollingCovariance, self).__init__(f'roll_cov={lookback}', alias=name, lookback=lookback)

    def process(self, inputs: dict) -> np.ndarray:
        data = inputs['ret']
        n_cols = data.shape[1]
        return data.rolling(window=self.lookback).cov().values.reshape(-1, n_cols, n_cols)


class LongTermDrift(Indicator):
    def __init__(self, name, lookback=90):
        super(LongTermDrift, self).__init__('con_drift', alias=name, lookback=lookback)

    def process(self, data: pd.DataFrame) -> np.ndarray:
        means = data.cumsum(axis=0).div(range(data.shape[0]), axis=0) # type: pd.DataFrame
        means.iloc[:self.lookback + 1] = np.nan
        return means.to_numpy()


class RateOfReturn(Indicator):
    def __init__(self, name, lookback=90, standardized=True):
        self.standardized = standardized
        super(RateOfReturn, self).__init__(f'roc={lookback}', alias=name, lookback=lookback)

    def process(self, inputs: dict) -> np.ndarray:
        ind = inputs['close'].pct_change(periods=self.lookback).values
        if self.standardized:
            return (ind - np.mean(ind, axis=1, keepdims=True)) / np.std(ind, axis=1, keepdims=True)
        return ind
