from cvxport import utils
import cvxport as xp


if __name__ == '__main__':
    assets = ['eq_us', 'eq_exus', 'eq_em', 'tn_us', 'tb_us', 'bond_em', 'reit']
    prices = utils.get_prices(assets)
    print('Last update: %s' % prices['close'].last_valid_index())

    strats = [
        # xp.strategy.InverseVolatilityStrategy(),
        xp.strategy.TwoStageRiskParityStrategy(),
        xp.strategy.SimpleRiskParityStrategy(),
        # xp.strategy.RiskParityStrategy(rep=1),
        # xp.strategy.MeanVarianceStrategy(rep=1),
        # xp.strategy.EqualWeightStrategy()
    ]
    backtester = xp.BackTester(prices, strats)
    result = backtester.run()
    result.show()
    result.plot(True)

