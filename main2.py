
import utils
import cvx_framework as xp


assets = ['eq_us', 'eq_exus', 'eq_em', 'tn_us', 'tb_us', 'bond_em', 'reit']
rets = utils.get_price_returns(assets)
print('Last update: %s' % rets.last_valid_index())

# strats = [xp.SingleAssetStrategy(name, idx) for idx, name in enumerate(assets)]
strats = [xp.InverseVolatilityStrategy()]
backtester = xp.BackTester(rets, strats)
result = backtester.run()
# result.plot()
print(result.stats)