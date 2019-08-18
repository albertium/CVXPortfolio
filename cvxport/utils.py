
import pandas as pd
import time
import plotly.graph_objs as go
import configparser


def get_prices_loader(config):
    def loader(asset):
        if asset not in config['Data Map']:
            raise ValueError('asset class "%s" doesn''t exist' % asset)

        # read data and pre-process
        root = config['Default']['data_root']
        tickers = config['Data Map'][asset].split(',')
        data = {}
        for ticker in tickers:
            panel = pd.read_csv('%s/%s.csv' % (root, ticker), parse_dates=['Date'], index_col=0)
            panel = panel.rename(lambda x: x.strip().lower(), axis=1)
            for col in ['open', 'high', 'low', 'close']:
                data.setdefault(col, []).append(panel[col].rename(asset).dropna())

        # splice series if needed
        if len(tickers) == 2:
            last_date = data['close'][1].first_valid_index()
            adj_ratio = data['close'][1][0] / data['close'][0][last_date]  # adjustment based on closed price
            for col, series in data.items():
                data[col] = [pd.concat([series[0][: last_date - pd.Timedelta('1d')] * adj_ratio, series[1]])]

        return {k: v[0] for k, v in data.items()}

    return loader


def get_prices(assets, config_name='config') -> pd.DataFrame:
    config = configparser.ConfigParser()
    config.read(config_name)
    loader = get_prices_loader(config)

    ts = {}
    for asset in assets:
        data = loader(asset)
        for col, series in data.items():
            ts.setdefault(col, []).append(series)

    return {col: pd.concat(series, axis=1, join='inner') for col, series in ts.items()}


def plot_area(title, dfs: [list, pd.DataFrame]):
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    fig = go.Figure()

    for group, df in enumerate(dfs):
        df = df.div(df.sum(axis=1), axis=0)
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], stackgroup=group, name='%s_%d' % (col, group), opacity=0.5))

    fig.update_layout(showlegend=True, xaxis={'hoverformat': '%d%b%Y'}, yaxis={'hoverformat': '.1%'})
    fig.update_layout(title=go.layout.Title(text=title))
    fig.show()


def plot_lines(df):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))

    fig.update_layout(showlegend=True, xaxis={'hoverformat': '%d%b%Y'}, yaxis={'hoverformat': '.1%'})
    fig.show()


def pretty_print(df: pd.DataFrame, formats):
    new = pd.DataFrame(index=df.index)
    for col, fmt in zip(df.columns, formats):
        new[col] = df[col].apply(lambda x: fmt.format(x))
    print(new)


def run_with_status(_msg, _iterable, _size, _func, _update=1):
    _start = time.time()
    _lap = _start
    print(f'[  0%] {_msg}', end='')
    for _idx, _data in enumerate(_iterable):
        _func(_data)
        _now = time.time()
        if _now - _lap > _update:
            print(f'\r[{_idx / _size: 4.0%}] {_msg}', end='', flush=True)
            _lap = _now
    print(f'\r[100%,{time.time() - _start: 5.1f}s] {_msg}')


def get_risk_contribution(df: pd.DataFrame, weights, lookback=90):
    n_col = df.shape[1]
    cov = df.rolling(window=lookback).cov().values.reshape(-1, n_col, n_col)
    w = weights.reshape(-1, n_col, 1)
    marginal_rc = (cov * w).sum(axis=1)
    return pd.DataFrame(marginal_rc * weights, index=df.index, columns=df.columns).dropna()
