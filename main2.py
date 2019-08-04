from cvxport import utils
import cvxport as xp


assets = ['eq_us', 'eq_exus', 'eq_em', 'tn_us', 'tb_us', 'bond_em', 'reit']
rets = utils.get_price_returns(assets)
print('Last update: %s' % rets.last_valid_index())

# strats = [xp.SingleAssetStrategy(name, idx) for idx, name in enumerate(assets)]
strats = [
    xp.strategy.InverseVolatilityStrategy(),
    xp.strategy.InverseVolatilityStrategy(timer=xp.strategy.Timer('1w')),
    xp.strategy.InverseVolatilityStrategy(timer=xp.strategy.Timer('1m')),
    xp.strategy.InverseVolatilityStrategy(timer=xp.strategy.Timer('1q')),
    xp.strategy.EqualWeightStrategy()
]
backtester = xp.BackTester(rets, strats)
result = backtester.run()
result.show()
result.plot()

