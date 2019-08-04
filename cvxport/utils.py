
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import configparser


def get_prices_loader(config):
    def loader(asset):
        if asset not in config['Data Map']:
            raise ValueError('asset class "%s" doesn''t exist' % asset)

        # read data and pre-process
        root = config['Default']['data_root']
        tickers = config['Data Map'][asset].split(',')
        data = []
        for ticker in tickers:
            tmp = pd.read_csv('%s/%s.csv' % (root, ticker), parse_dates=['Date'], index_col=0)[' Close']
            data.append(tmp.rename(asset).pct_change().dropna())

        # splice series if needed
        if len(data) == 2:
            last_date = data[1].first_valid_index() - pd.Timedelta('1d')
            data[0] = pd.concat([data[0][: last_date], data[1]])

        return data[0]

    return loader


def get_price_returns(assets, config_name='config') -> pd.DataFrame:
    config = configparser.ConfigParser()
    config.read(config_name)
    loader = get_prices_loader(config)

    ts = []
    for asset in assets:
        ts.append(loader(asset))

    return pd.concat(ts, axis=1, join='inner')


def plot_area(dfs: [list, pd.DataFrame]):
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    fig = go.Figure()

    for group, df in enumerate(dfs):
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], stackgroup=group, name='%s_%d' % (col, group), opacity=0.5))

    fig.update_layout(showlegend=True, xaxis={'hoverformat': '%d%b%Y'}, yaxis={'hoverformat': '.1%'})
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
