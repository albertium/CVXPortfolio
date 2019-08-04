
import numpy as np
import cvxpy as cp
import plotly.offline as py
from plotly import graph_objs as go


def get_optimal_portfolio(mean, cov, n_samples=100, with_cash=False):
    n_assets = len(mean)
    weights = cp.Variable(n_assets)
    gamma = cp.Parameter(nonneg=True)
    ret = mean.T * weights
    risk = cp.quad_form(weights, cov)

    cond = weights[: n_assets - 1] >= 0 if with_cash else weights >= 0
    problem = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(weights) == 1, cond])

    ret_data = np.zeros(n_samples)
    risk_data = np.zeros(n_samples)
    multiplier = np.logspace(-2, 3, n_samples)
    for i in range(n_samples):
        gamma.value = multiplier[i]
        problem.solve()
        ret_data[i] = ret.value
        risk_data[i] = cp.sqrt(risk).value

    return ret_data, risk_data


K = 2  # number of assets

np.random.seed(1)

# no cash
mus = np.abs(np.random.randn(K, 1))
sigmas = np.random.randn(K, K)
sigmas = sigmas.T.dot(sigmas)

# with cash
mus1 = np.vstack([mus, min(mus)])
sigmas1 = np.zeros((K + 1, K + 1))
sigmas1[:K, :K] = sigmas

fig = go.Figure()
ret, risk = get_optimal_portfolio(mus, sigmas)
fig.add_trace(go.Scatter(x=risk, y=ret, name='no cash'))

ret, risk = get_optimal_portfolio(mus1, sigmas1, with_cash=True)
fig.add_trace(go.Scatter(x=risk, y=ret, name='w cash'))

fig.update_layout(showlegend=True)
fig.update_xaxes(range=[0, 1.1])
fig.update_yaxes(range=[0, 2])

py.iplot(fig)

mu = np.array([0.01, 0.01])
sigma = np.array([[0.5, 0], [0, 1.5]])