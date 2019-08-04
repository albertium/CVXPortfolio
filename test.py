
import pandas_datareader.data as web
from datetime import datetime

start = datetime(2010, 7, 1)
end = datetime(2019, 7, 30)

data = web.DataReader('TLT', 'av-daily-adjusted', start, end)
print(data.head())