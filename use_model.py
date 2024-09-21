import pandas as pd

from MT5_TOOL.model_use import model_load

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


setUp = model_load(
    RootPath = "models",
    Symbol   = Symbol,
)
test = setUp.GetSetUpNames()
print(test)
setUp.GetSetUp(test[0])
print(setUp.GetCurrentSetUp())
re = setUp.loadModels()
print(re)
re = setUp.loadModels()
print(re)
data = pd.read_csv("./models/EURUSD/stocks/stock_EURUSD_1.csv").iloc[:240]
results = setUp.predictions(data)
print(results)




