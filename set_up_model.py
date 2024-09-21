from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt

from MT5_TOOL.AI import AImodelForex

mt.initialize()

Login=58636136
Password="VxBj!qZ2"
server ="AdmiralsGroup-Demo"
info = mt.login(Login,Password,server)

print("info = ",info)

setUp =AImodelForex(
    RootPath = "models",
    Symbol   = "EURUSD",
)

setUp.model_setup(
                    intervalTime = 120 ,
                    nextTime = 60,
                    timeframe = mt.TIMEFRAME_M1,
                    pip_value = 0.00003, # 1 pip for EUR/USD
                    deviation = 20,
                    sl_pips = 15,
                    tp_pips = 30,
                    skipFunctionBoundaries= .60,
                    preprocesMethods={
                            "Candle" : True,
                            "MinMax" : True,
                            "rsi" : True,
                            "bollinger_bands" : True,
                     }
)

print(setUp.GetCurrentSetUp())

setUp.SaveSetUp(name="TEST_2")

names = setUp.GetSetUpNames()

print(names)

setUp.GetSetUp(names[0])

print(setUp.GetCurrentSetUp())










