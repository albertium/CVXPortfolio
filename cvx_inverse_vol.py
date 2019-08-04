
import numpy as np
import cvxpy as cp
import plotly.offline as py
from plotly import graph_objs as go


def get_optimal_portfolio(mean, cov, gamma):
    n_assets = len(mean)
    weights = cp.Variable(n_assets)
    ret = mean.T * weights
    risk = cp.quad_form(weights, cov)

    problem = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(weights) == 1, weights >= 0])
    problem.solve()

    return float(ret.value), float(risk.value), np.array(weights.value)


def get_risk_parity(mean, cov, gamma):
    n_assets = len(mean)
    weights = cp.Variable(n_assets)
    ret = mean.T * weights

    L = np.linalg.cholesky(cov)
    risk = cp.norm(L.T * weights, 2)

    problem = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(weights) == 1, weights >= 0])
    problem.solve()

    return float(ret.value), float(risk.value), np.array(weights.value)


K = 2  # number of assets

np.random.seed(1)

mu = np.array([0.01, 0.01])
sigma = np.array([[0.25, 0], [0, 0.0625]])
ret, risk, w = get_optimal_portfolio(mu, sigma, 0.5)
print(ret)
print(risk)
print(w)


