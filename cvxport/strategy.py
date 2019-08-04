
import numpy as np
import abc
from typing import Dict, Iterable
from .indicator import Indicator, VolatilityIndicator


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


class SingleAssetStrategy(Strategy):
    def __init__(self, asset_name, idx):
        self.idx = idx
        super(SingleAssetStrategy, self).__init__('single=%s' % asset_name)

    def generate_weights(self, inputs: Dict):
        weights = np.zeros(inputs['n_asset'])
        weights[self.idx] = 1
        return weights


class InverseVolatilityStrategy(Strategy):
    def __init__(self, lookback=90):
        ind = VolatilityIndicator('vol', lookback)
        super(InverseVolatilityStrategy, self).__init__('inverse_vol=%d' % lookback, [ind])

    def generate_weights(self, inputs: Dict):
        inv = 1 / inputs['vol']
        return inv / np.sum(inv)