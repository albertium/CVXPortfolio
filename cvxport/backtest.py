
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import List
import traceback
from . import utils, const
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

            strategy, data, inputs, max_lev = msg_data
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
        inputs = {}
        for indicator in self.indicators:
            inputs[indicator.name] = indicator.process(self.data)

        # main loop
        print('\nRunning')
        max_leverage = self.config['leverage'] + 1e-4
        n_processors = min(self.config['pool'], len(self.strategies))

        result = ResultSet(self.data)
        qin = mp.Queue()
        qout = mp.Queue()
        processes = []
        for pid in range(n_processors):
            process = mp.Process(target=BackTester.run_remote_backtest, args=(pid, qin, qout))
            process.start()
            processes.append(process)

        for strategy in self.strategies:
            qin.put((const.Msg.INPUT, (strategy, self.data, inputs, max_leverage)))

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
