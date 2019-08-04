
import ib_insync as ibs


ib = ibs.IB()
ib.connect('127.0.0.1', 4001, clientId=1)
print(ib.positions())
ib.disconnect()