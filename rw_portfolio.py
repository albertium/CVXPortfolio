
from scipy.stats import norm
from cvxport import utils


def standardize_data(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def get_simple_parity_weights(df, lookback=90):
    inv = 1 / df.rolling(lookback, min_periods=lookback).std().dropna()
    return inv.div(inv.sum(axis=1), axis=0)


def get_correlation_adjustment(df, lookback=90):
    corr = df.rolling(lookback, min_periods=lookback).corr().dropna().groupby(level=0).mean()
    # the higher the correlation, the lower the weight
    p_score = 1 - standardize_data(corr).apply(norm.cdf)
    # normalize to sum of 1 so that adjustment is on the same scale of original weight
    return p_score.div(p_score.sum(axis=1), axis=0) - 1 / df.shape[1]


def get_momentum_adjustment(df, lookback=90):
    momo = (1 + df).cumprod().pct_change(periods=lookback)
    return standardize_data(momo)


def get_momentum_speed_adjustment(df, lookback=90):
    speed = (1 + df).cumprod().rolling(window=lookback).mean().pct_change().shift(1)
    return standardize_data(speed)


def adjust_weights(weights, adjustment_weights, multipliers, cutoffs):
    for adj, multiplier, cutoff in zip(adjustment_weights, multipliers, cutoffs):
        if cutoff is None:
            weights = weights * (1 + multiplier * adj)
        else:
            weights = weights * (1 + multiplier * (adj >= cutoff))
        weights = weights.div(weights.sum(axis=1), axis=0)
    return weights


if __name__ == "__main__":
    # emerging bond, bitcoin, put write, 20yr bond, 2x 10yr bond, FTSE, ex-US REIT, TR US stock, emerging stock
    tickers = ['EMB', 'GBTC', 'PUTW', 'TLT', 'UST', 'VEA', 'VNQI', 'VTI', 'VWO']
    root_dir = 'C:/Users/Albert/Resilio Sync/FXBootcamp/RiskPremia'
    prices = utils.get_prices2(tickers, root_dir, end_date='2019-08-02')
    rets = prices['close'].pct_change()
    print(f"Date range: {prices['close'].first_valid_index()} to {prices['close'].last_valid_index()}")
    # utils.plot_lines(prices['close'], normalize=True)

    base_weights = get_simple_parity_weights(rets)
    corr_weights = get_correlation_adjustment(rets, lookback=84)
    momo_adjustments = [get_momentum_adjustment(rets, x * 22) for x in [3, 6, 9, 12]]
    speed_adjustments = [get_momentum_speed_adjustment(rets, x * 22) for x in [3, 6, 9, 12]]
    all_adjustments = [corr_weights] + momo_adjustments + speed_adjustments
    multipliers = [1] + 8 * [.1]
    cutoffs = [None] + 8 * [0]
    final_weights = adjust_weights(base_weights, all_adjustments, multipliers, cutoffs)
    # utils.plot_area([base_weights, final_weights])
    print(final_weights.tail())
    pass