
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import List
import traceback
import configparser
from . import utils, const
from .strategy import Strategy


class Result:
    def __init__(self, name, weights: np.ndarray, lookback, inputs, capital, costs):
        if weights.shape != inputs['close'].shape:
            raise RuntimeError(f'{name} weights shape doesn''t match')

        self.name = name
        self.T, self.n_assets = weights.shape

        self.index = inputs['close'].index
        self.open = inputs['open'].values
        self.high = inputs['high'].values
        self.low = inputs['low'].values
        self.close = inputs['close'].values

        self.init_capital = capital
        self.spreads, self.delay, self.comm = costs['spreads'], costs['delay'], costs['comm']
        self.lookback = lookback
        self.weights = weights

        # records
        self.equity = None
        self.cash = None
        self.slippage_cost = None
        self.spread_cost = None
        self.comm_cost = None
        self.traded_shares = None
        self.total_shares = None
        self.ret = None
        self.stats = None

        self.calculate_performance()
        self.calculate_statistics()

    def calculate_performance(self):
        cash = np.zeros(self.T)
        cash[self.lookback - 1] = self.init_capital
        positions = np.zeros((self.T, self.n_assets))
        spread_cost = np.zeros(self.T)
        slippage = np.zeros(self.T)
        comm = np.zeros(self.T)
        traded_shares = np.zeros(self.T)
        equity = np.zeros(self.T)
        for idx in range(self.lookback, self.T):
            prev_idx = idx - 1
            # if need re-balance
            if not np.isnan(self.weights[idx][0]):
                opening = self.open[idx]
                curr_capital = cash[prev_idx] + positions[prev_idx].dot(opening)  # opening capital
                pos_deltas = np.fix(curr_capital * self.weights[idx] / opening - positions[prev_idx])
                abs_deltas = np.abs(pos_deltas)
                traded_shares[idx] = np.sum(abs_deltas)
                delayed_prices = self.estimate_fills(idx)
                estimated_fill = (delayed_prices * (1 + np.sign(pos_deltas) * self.spreads)).dot(pos_deltas)
                spread_cost[idx] = delayed_prices.dot(self.spreads * abs_deltas)
                slippage[idx] = estimated_fill - opening.dot(pos_deltas)
                comm[idx] = self.comm * abs_deltas
                cash[idx] = cash[idx - 1] - estimated_fill - comm[idx]
                positions[idx] = positions[prev_idx] + pos_deltas
                equity[idx] = cash[idx] + positions[idx].dot(self.close[idx])
            else:
                cash[idx] = cash[prev_idx]
                positions[idx] = positions[prev_idx]
                equity[idx] = cash[idx] + positions[idx].dot(self.close[idx])

        self.equity = pd.Series(data=equity, index=self.index, name=self.name)
        self.cash = pd.Series(data=cash, index=self.index, name=self.name)
        self.slippage_cost = pd.Series(data=slippage, index=self.index, name=self.name)
        self.spread_cost = pd.Series(data=spread_cost, index=self.index, name=self.name)
        self.comm_cost = pd.Series(data=comm, index=self.index, name=self.name)
        self.traded_shares = pd.Series(data=traded_shares, index=self.index, name=self.name)
        self.total_shares = pd.Series(data=np.sum(np.abs(positions), axis=1), index=self.index, name=self.name)
        self.ret = self.equity.pct_change().dropna()

    def calculate_statistics(self):
        series = self.equity.to_numpy()

        # performance
        ret = np.power(series[-1] / series[0], 252 / len(series)) - 1
        sd = np.std(self.ret.to_numpy()) * np.sqrt(252)
        sharpe = ret / sd if sd > 0 else np.nan
        dd = self.calculate_drawdown(series)

        # costs
        normalizer = 252 / self.init_capital
        spread = np.mean(self.spread_cost) * normalizer
        net_slippage = np.mean(self.slippage_cost) * normalizer - spread  # slippage net of spread
        comm = np.mean(self.comm_cost) * normalizer
        traded_shares = np.mean(self.traded_shares) * 252 / np.mean(self.total_shares)
        self.stats = pd.Series([sharpe, ret, sd, dd, net_slippage, spread, comm, traded_shares],
                               index=['Sharpe', 'Return', 'SD', 'maxDD', 'netSLP', 'Spread', 'Comm', 'Traded'],
                               name=self.name)

    def estimate_fills(self, idx):
        op, hi, lo, co = self.open[idx], self.high[idx], self.low[idx], self.close[idx]
        return np.fromiter(
            (self.estimate_price(self.delay, op[i], hi[i], lo[i], co[i]) for i in range(self.n_assets)),
            np.float,
            self.n_assets
        )

    @classmethod
    def estimate_price(cls, slippage, op, hi, lo, co):
        data = np.array([op, hi, lo, co]) if co < op else np.array([op, lo, hi, co])
        sign = data[1:] - data[:-1]
        interval = np.cumsum(np.abs(sign))
        slippage *= interval[-1]  # scale delay so that we don't need to scale all interval numbers
        idx = np.argmin(slippage > interval)
        if idx == 0:
            return data[idx] + np.sign(sign[idx]) * slippage
        return data[idx + 1] + np.sign(sign[idx]) * (slippage - interval[idx])

    @classmethod
    def calculate_drawdown(cls, ts):
        peak = ts[0]
        dd = 0
        for x in ts:
            peak = max(peak, x)
            dd = min(dd, min(x / peak - 1, 0))
        return dd


class ResultSet:
    def __init__(self, inputs):
        self.inputs = inputs
        self.results = []

        config = configparser.ConfigParser()
        config.read('config')

        # prepare spreads
        if config.getint('Account', 'use_spread'):
            spreads = []
            default_spread = config.getfloat('Spreads', 'DEFAULT')
            for ticker in inputs['tickers']:
                if ticker in config['Data Map']:
                    ticker = config.get('Data Map', ticker).split(',')[-1]
                spreads.append(config.getfloat('Spreads', ticker) if ticker in config['Spreads'] else default_spread)
            spreads = np.array(spreads)
        else:
            spreads = np.zeros(inputs['n_asset'])

        # prepare cost
        self.costs = {
            'comm': config.getfloat('Account', 'comm'),
            'slippage': config.getint('Account', 'delay') / 6.5 / 60,  # 6.5 hours per trading day
            'spreads': spreads / 100  # spreads in percentage point
        }
        self.capital = config.getfloat('Account', 'capital')

        self.lookbacks = {}
        self.returns = pd.DataFrame(index=data.index)
        self.performances = pd.DataFrame(index=data.index)
        self.stats = pd.DataFrame(columns=['Sharpe', 'Return', 'SD', 'maxDD'])
        self.risk_contributions = {}

    def add_result(self, name, lookback, weights):
        self.results[name] = Result(weights, lookback, self.inputs, self.capital, self.spreads, self.costs)

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
    def __init__(self, data: pd.DataFrame, strategies: List[Strategy], config=None):
        self.data = data
        self.strategies = strategies
        self.indicators = list(set(sum([p.indicators for p in strategies], [])))
        self.config = {'leverage': 1, 'pool': mp.cpu_count() - 1} if config is None else config
        self.progresses = {strategy.name: 0 for strategy in self.strategies}

    @classmethod
    def run_remote_backtest(cls, pid, qin: mp.Queue, qout: mp.Queue):
        qout.put((const.Msg.START, pid))
        while True:
            msg_type, msg_data = qin.get()
            if msg_type == const.Msg.END:
                break

            if msg_type != const.Msg.INPUT:
                print(f'Unrecognized message type {msg_type}')

            strategy, inputs, max_lev = msg_data
            data = inputs['ret']
            try:
                weights = strategy.run(data, inputs, max_lev, qout=qout)
                qout.put((const.Msg.RESULT, (strategy.name, strategy.lookback, weights)))
            except:
                qout.put((const.Msg.ERROR, (strategy.name, traceback.format_exc())))

        qout.put((const.Msg.END, pid))

    def print_progress(self, name, progress):
        self.progresses[name] = progress
        ls = [[k, v] for k, v in self.progresses.items()]
        ls.sort(reverse=True, key=lambda x: x[1])
        out = '| '.join([f'{x[0]}: {x[1]: 4.0%}' for x in ls])
        print('\r' + out, end='')

    def run(self):
        # reset progress
        self.progresses = {strategy.name: 0 for strategy in self.strategies}

        # prepare indicators
        inputs = dict({'ret': self.data['close'].pct_change(), 'tickers': self.data['close'].columns}, **self.data)
        for indicator in self.indicators:
            inputs[indicator.name] = indicator.process(inputs)

        # main loop
        print('\nRunning')
        max_leverage = self.config['leverage'] + 1e-4
        n_processors = min(self.config['pool'], len(self.strategies))

        result = ResultSet(inputs)
        qin = mp.Queue()
        qout = mp.Queue()
        processes = []
        for pid in range(n_processors):
            process = mp.Process(target=BackTester.run_remote_backtest, args=(pid, qin, qout))
            process.start()
            processes.append(process)

        for strategy in self.strategies:
            qin.put((const.Msg.INPUT, (strategy, inputs, max_leverage)))

        for _ in range(n_processors):
            qin.put((const.Msg.END, ()))

        counter = n_processors
        while counter:
            msg_type, msg_data = qout.get()
            if msg_type == const.Msg.PROGRESS:
                self.print_progress(msg_data[0], msg_data[1])
            elif msg_type == const.Msg.RESULT:
                result.add_performance(msg_data[0], msg_data[1], msg_data[2])
                self.print_progress(msg_data[0], 1)
                print(f'\r{msg_data[0]} result received')
            elif msg_type == const.Msg.START:
                print(f'\rProcess {msg_data} started')
            elif msg_type == const.Msg.ERROR:
                print(f'\r{msg_data[0]} error: {msg_data[1]}')
            elif msg_type == const.Msg.END:
                print(f'\rProcess {msg_data} returned')
                counter -= 1

        for process in processes:
            process.join()

        print()
        return result
