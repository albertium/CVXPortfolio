
import pandas_datareader.data as web
from matplotlib import pyplot as plt

data = web.DataReader(['SHY', 'IEF', 'TLT'], 'tiingo', '2004-01-01', '2019-07-01')
panel = data.reset_index().pivot(index='date', columns='symbol', values='adjClose').pct_change().dropna()
# final = 2 * panel['IEF'] - panel['SHY'] - panel['TLT']
final = panel['IEF'] - 7.49 / 17.66 * panel['TLT']
final = (final + 1).cumprod()
final.plot()
plt.show()