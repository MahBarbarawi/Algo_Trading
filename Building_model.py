from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt
import pandas as pd

from MT5_TOOL.model_setup import train_model

# mt.initialize()
#
# Login=42664415
# Password="s$6y3C50%mJ5dT"
#
# server ="AdmiralsGroup-Demo"
# info = mt.login(Login,Password,server)
#
# current_datetime_utc = datetime.utcnow()
#
# # Calculate the datetime 6 days ago
# datetime_6_days_ago = current_datetime_utc - timedelta(days=69)
#
# print(f"Current datetime (UTC): {current_datetime_utc}")
# print(f"Datetime 6 days ago (UTC): {datetime_6_days_ago}")
#
#
#
# print("info = ",info)
Symbol = "EURUSD"
timeMT =1


setUp = train_model(
    RootPath = "models",
    Symbol   = Symbol,
)
setUp.model_setup(
                    intervalTime = 120,
                    nextTime = 60,
                    timeframe = timeMT,
                    pip_value = 0.00003,# 1 pip for EUR/USD
                    deviation = 20,
                    sl_pips = 15,
                    tp_pips = 30,
                    skipFunctionBoundries = .60,
                    preprocesMethods = {
                            "Candle" : True,
                            "MinMax" : True,
                            "rsi" : True,
                            "bollinger_bands" : True,

                    }
)

# df =pd.read_csv("models/EURUSD/stocks/stock_EURUSD_1.csv")
# df.sort_values("time")
# re = setUp.data_preparing_rows(df)
# print(re)
setUp.train_model_Tp_Sl()

