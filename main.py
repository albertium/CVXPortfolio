from cvxport import utils
import riskparity


assets = ['eq_us', 'eq_exus', 'eq_em', 'tn_us', 'tb_us', 'bond_em', 'reit']
rets = utils.get_prices(assets)
print('Last update: %s' % rets.last_valid_index())
# utils.plot_returns(rets)

base_weights = riskparity.get_simple_parity_weights(rets)
corr_weights = riskparity.get_correlation_adjustment(rets)
final_weights = riskparity.adjust_weights(base_weights, [corr_weights])
utils.plot_area([base_weights, final_weights])

