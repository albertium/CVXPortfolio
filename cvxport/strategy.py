
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import abc
from typing import Dict, Iterable
from .indicator import Indicator, VolatilityIndicator


class Timer:
    def __init__(self, freq: str = '1b'):
        self.unit = freq[-1]
        self.number = int(freq[:-1])
        if self.unit == 'b':
            self.delta = BDay(self.number)
        elif self.unit == 'w':
            self.delta = pd.DateOffset(weeks=self.number)
        elif self.unit == 'm':
            self.delta = pd.DateOffset(months=self.number)
        elif self.unit == 'q':
            self.delta = pd.DateOffset(months=self.number * 3)
        else:
            raise ValueError(f'Invalid frequency {freq}')
        self.freq = freq
        self.last_time = None

    def is_up(self, current_time: pd.Timestamp):
        if self.last_time is None or self.last_time + self.delta <= current_time:
            self.last_time = current_time
            return True
        return False

    def __str__(self):
        return f'<Timer> freq={self.freq}'


class Strategy(abc.ABC):
    def __init__(self, base_name, indicators: Iterable[Indicator] = None, timer: Timer = None):
        self.lookback = 0 if indicators is None else max([x.lookback for x in indicators])
        self.indicators = indicators if indicators is not None else []
        self.timer = Timer('1b') if timer is None else timer
        self.name = f'{base_name} freq={self.timer.freq}'

    @abc.abstractmethod
    def generate_weights(self, inputs: Dict):
        pass

    def __str__(self):
        return f'<STRATEGY> {self.name}'


class SingleAssetStrategy(Strategy):
    def __init__(self, asset_name, idx):
        self.idx = idx
        super(SingleAssetStrategy, self).__init__('single=%s' % asset_name)

    def generate_weights(self, inputs: Dict):
        weights = np.zeros(inputs['n_asset'])
        weights[self.idx] = 1
        return weights


class EqualWeightStrategy(Strategy):
    def __init__(self, timer=None):
        super(EqualWeightStrategy, self).__init__('eq_wgt', timer=timer)

    def generate_weights(self, inputs: Dict):
        n_asset = inputs['n_asset']
        return np.ones(n_asset) / n_asset


class InverseVolatilityStrategy(Strategy):
    def __init__(self, lookback=90, timer=None):
        ind = VolatilityIndicator('vol', lookback)
        super(InverseVolatilityStrategy, self).__init__('inv_vol=%d' % lookback, [ind], timer=timer)

    def generate_weights(self, inputs: Dict):
        inv = 1 / inputs['vol']
        return inv / np.sum(inv)
