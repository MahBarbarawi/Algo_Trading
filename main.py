from datetime import datetime, timedelta, timezone

from MT5_TOOL.AI import AImodelForex
import MetaTrader5 as mt

mt.initialize()

Login=58636136
Password="VxBj!qZ2"
server ="AdmiralsGroup-Demo"
info = mt.login(Login,Password,server)


dash="-"*30
model = AImodelForex(RootPath="DATA2",
                     Symbol='USD', )
historyOverAll = []


now = datetime.now()

formatted_full = now.strftime("%Y-%m-%d %H:%M:%S")
print(dash, f"start trading,-{formatted_full}-")

if info:
    model.__SaveLog("login success", typeOFlog="account_log")
else:
    model.__SaveLog("login FAILED", typeOFlog="account_log")


model.__SaveLog("start trading", typeOFlog="trading")
# some_file_management_function()
while True:
    ActiveTrades = mt.positions_get()


