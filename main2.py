from cvxport import utils
import cvxport as xp

print(xp.backtest.Result.estimate_price(0, 2, 4, 1, 3))
print(xp.backtest.Result.estimate_price(0.2, 2, 4, 1, 3))
print(xp.backtest.Result.estimate_price(0.8, 2, 4, 1, 3))
print(xp.backtest.Result.estimate_price(0.95, 2, 4, 1, 3))

# if __name__ == '__main__':
#     assets = ['eq_us', 'eq_exus', 'eq_em', 'tn_us', 'tb_us', 'bond_em', 'reit']
#     prices = utils.get_prices(assets)
#     print('Last update: %s' % prices['close'].last_valid_index())
#     test = xp.backtest.Result(prices, 1)
#
#     strats = [
#         xp.strategy.InverseVolatilityStrategy(),
#         # xp.strategy.TwoStageRiskParityStrategy(),
#         # xp.strategy.SimpleRiskParityStrategy(),
#         # xp.strategy.RiskParityStrategy(rep=1),
#         # xp.strategy.MeanVarianceStrategy(rep=1),
#         # xp.strategy.EqualWeightStrategy()
#     ]
#     backtester = xp.BackTester(prices, strats)
#     result = backtester.run()
#     result.show()
#     result.plot()
