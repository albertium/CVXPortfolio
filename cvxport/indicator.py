
import pandas as pd
import numpy as np
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


class VolatilityIndicator(Indicator):
    def __init__(self, name, lookback=90):
        super(VolatilityIndicator, self).__init__('volatility=%d' % lookback, name, lookback)

    def process(self, data: pd.DataFrame):
        return data.rolling(window=self.lookback).std().to_numpy()
