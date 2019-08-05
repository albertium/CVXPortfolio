
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import cvxpy as cp
import abc
from typing import Dict, Iterable
from .indicator import Indicator
from . import indicator as idr


class Timer(abc.ABC):
    def __init__(self, name):
        self.name = name
        self.last_time = None

    @abc.abstractmethod
    def is_up(self, current_time):
        pass

    def __str__(self):
        return f'<Timer> {self.name}'


class IntervalTimer(Timer):
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
        super(IntervalTimer, self).__init__(f'freq={self.freq}')

    def is_up(self, current_time: pd.Timestamp):
        if self.last_time is None or self.last_time + self.delta <= current_time:
            self.last_time = current_time
            return True
        return False


class Strategy(abc.ABC):
    def __init__(self, base_name, indicators: Iterable[Indicator] = None, timer: Timer = None, rep=1):
        self.lookback = 0 if indicators is None else max([x.lookback for x in indicators])
        self.indicators = indicators if indicators is not None else []
        self.timer = IntervalTimer('1b') if timer is None else timer
        self.name = f'{base_name} freq={self.timer.freq}'
        self.rep = rep

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
        n_asset = inputs['n_assets']
        return np.ones(n_asset) / n_asset


class InverseVolatilityStrategy(Strategy):
    def __init__(self, lookback=90, timer=None):
        ind = idr.RollingVolatility('vol', lookback)
        super(InverseVolatilityStrategy, self).__init__('inv_vol=%d' % lookback, [ind], timer=timer)

    def generate_weights(self, inputs: Dict):
        inv = 1 / inputs['vol']
        return inv / np.sum(inv)


class SimpleRiskParityStrategy(Strategy):
    def __init__(self, gamma=2, use_mean=False, lookback=90, timer=None, rep=3):
        self.gamma = gamma
        self.use_mean = use_mean
        inds = [
            idr.RollingCovariance('cov', lookback=lookback),
            idr.LongTermDrift('mean', lookback=lookback)
        ]

        timer = IntervalTimer('1w')
        name = f'srp={lookback} gamma={gamma}' if use_mean else f'srp={lookback} no_mean'
        super(SimpleRiskParityStrategy, self).__init__(name, inds, timer=timer, rep=rep)

    def generate_weights(self, inputs: Dict):
        n_assets = inputs['n_assets']
        mean = inputs['mean']
        cov = inputs['cov']

        weights = cp.Variable(n_assets)
        risk = cp.quad_form(weights, cov)

        if self.use_mean:
            gamma = cp.Constant(self.gamma)
            ret = mean.T * weights
            problem = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(weights) == 1, weights >= 0])
        else:
            problem = cp.Problem(cp.Minimize(risk), [cp.sum(weights) == 1, weights >= 0])

        problem.solve()
        return weights.value
