
from scipy.stats import norm


def get_simple_parity_weights(df, lookback=90):
    inv = 1 / df.rolling(lookback, min_periods=lookback).std().dropna()
    return inv.div(inv.sum(axis=1), axis=0)


def get_correlation_adjustment(df, lookback=90):
    corr = df.rolling(lookback, min_periods=lookback).corr().dropna().groupby(level=0).mean()
    # the higher the correlation, the lower the weight
    p_score = 1 - corr.sub(corr.mean(axis=1), axis=0).div(corr.std(axis=1), axis=0).apply(norm.cdf)
    # normalize to sum of 1 so that adjustment is on the same scale of original weight
    return p_score.div(p_score.sum(axis=1), axis=0)


def adjust_weights(base_weight, adjustment_weights, multipliers=None):
    if multipliers is None:
        multipliers = [0.1] * len(adjustment_weights)

    dampening = 1 - sum(multipliers)
    if dampening < 0.5:
        raise ValueError('Adjustment too much. Dampening is %.3f' % dampening)
    new_weight = dampening * base_weight

    for m, adj in zip(multipliers, adjustment_weights):
        new_weight += m * adj

    return new_weight
