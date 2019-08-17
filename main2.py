from cvxport import utils
import cvxport as xp


if __name__ == '__main__':
    assets = ['eq_us', 'eq_exus', 'eq_em', 'tn_us', 'tb_us', 'bond_em', 'reit']
    rets = utils.get_price_returns(assets)
    print('Last update: %s' % rets.last_valid_index())

    strats = [
        # xp.strategy.InverseVolatilityStrategy(),
        xp.strategy.TwoStageRiskParityStrategy(),
        xp.strategy.SimpleRiskParityStrategy(),
        # xp.strategy.RiskParityStrategy(rep=1),
        # xp.strategy.MeanVarianceStrategy(rep=1),
        # xp.strategy.EqualWeightStrategy()
    ]
    backtester = xp.BackTester(rets, strats)
    result = backtester.run()
    result.show()
    result.plot(True)

