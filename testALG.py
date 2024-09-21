import pandas as pd
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


from tqdm import tqdm
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # Your code here that might trigger warnings



df =pd.read_csv("models/EURUSD/stocks/stock_EURUSD_1.csv")

df['datetime'] = pd.to_datetime(df['time'])
df = df.sort_values("time")


df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute

df['time_diff'] = df['datetime'].diff()

# Check if the difference is greater than 1 minute (60 seconds)
df['more_than_1_min'] = df['time_diff'] > pd.Timedelta(minutes=1)

mask = df["open"] < df['close']
df["Candle"] = mask.astype("int")


def targetExport(dffeatures, dfTarget,
                 lengthFeatures=60, lengthtarget=60
                 , ) -> "contnious , discret":
    if len(dffeatures) != lengthFeatures:
        raise f"length of data do not match {len(dffeatures)}== {lengthFeatures} "

    if len(dfTarget) != lengthtarget:
        raise f"length of data do not match {len(dfTarget)}== {lengthtarget}"

    BuyPoint = dffeatures['close'].values[-1]

    # next Postion
    Target = dfTarget - BuyPoint
    #     print(Target)
    #     AvgOfTarget = Target.mean().mean()

    #     rowMean  = dfTarget.mean(axis=1)
    upper = 0 < Target
    lower = 0 > Target
    high = dfTarget[upper].max().max() - BuyPoint
    low = dfTarget[lower].min().min() - BuyPoint

    prediction = high > abs(low)
    #     print(high)
    #     print(low)
    #     print(prediction)

    # detirmening SL / TP
    if prediction:
        diff = high
        tp = dfTarget[upper].max().max()

        sl = dfTarget[lower].min().min()

    else:
        diff = low

        #         print(dfTarget[ BuyPoint < rowMean ])
        sl = dfTarget[upper].max().max()

        tp = dfTarget[lower].min().min()
    if tp == None:
        return True

    return (diff,
            int(prediction),
            BuyPoint,
            tp,
            sl,

            )


def calculate_rsi(df, period=10):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_stoch_rsi(df, window=14):
    rsi = calculate_rsi(df, window)
    min_rsi = rsi.rolling(window=window, min_periods=window).min()
    max_rsi = rsi.rolling(window=window, min_periods=window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    return stoch_rsi


def calculate_bollinger_bands(df, window=120):
    """
    Calculate Bollinger Bands for a given DataFrame with stock price data using a specified window.

    Parameters:
    - df: DataFrame containing historical stock price data with a 'close' column.
    - window: The number of periods for the moving average (e.g., 120 days).

    Returns:
    - DataFrame with columns for the middle band, upper band, and lower band.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    df['Middle Band'] = df['close'].rolling(window=window).mean()
    df['Rolling Std Dev'] = df['close'].rolling(window=window).std()
    df['Upper Band'] = df['Middle Band'] + (df['Rolling Std Dev'] * 2)
    df['Lower Band'] = df['Middle Band'] - (df['Rolling Std Dev'] * 2)

    return df[['Middle Band', 'Upper Band', 'Lower Band']]

def Val(df):
    return df[["open","high","low","close"]].describe().T.iloc[:,1:]
def preprocessData(df, intervalTime=120, NextTime=60):
    values = df[["open", "high", "low", "close", "Candle", ]].values
    timeST = df[["day_of_week_num", "hour", "minute", ]].values
    DataLength = len(values)
    rows, targetBin, targetCon, tpItems, slItems, DesItems, rsiItems, BollItems, timeFeature = [], [], [], [], [], [], [], [], []
    strInd = 0
    for ind in tqdm(range(DataLength)):
        #         print(ind)
        if ind > DataLength - (intervalTime * 2 + NextTime):
            break

        # Feature
        row = values[ind: ind + intervalTime]
        timeRow = timeST[ind: ind + intervalTime]
        # TECHNIACAL Analysis

        Des = Val(pd.DataFrame(row[:, :4], columns=["open", "high", "low", "close", ])).values
        ExtendedRow = values[ind: ind + intervalTime * 2]

        rsi = calculate_rsi(pd.DataFrame(ExtendedRow[:, :4], columns=["open", "high", "low", "close", ]),
                            period=10).values
        Boll = calculate_bollinger_bands(pd.DataFrame(ExtendedRow[:, :4], columns=["open", "high", "low", "close", ]),
                                         window=121).values

        # target values
        AvgOfTarget, Prediction, Price, tp, sl = targetExport(

            dffeatures=pd.DataFrame(row[:, :4], columns=["open", "high", "low", "close", ]), lengthFeatures=intervalTime
            ,
            dfTarget=pd.DataFrame(values[intervalTime + ind:ind + intervalTime + NextTime, :4]), lengthtarget=NextTime
        )

        # Feature Prepreation
        row = list(row)

        Max, Min = -np.inf, np.inf
        MaxInd, MinInd = 0, 0
        for i in range(len(row)):
            # min

            subRow2 = row[i][:4]

            if min(subRow2) < Min:
                Min = min(subRow2)
                MinInd = i

            # max
            elif max(subRow2) > Max:
                Max = max(subRow2)
                MaxInd = i

        row[0] = np.append(row[0], 0)
        row[intervalTime - 1] = np.append(row[intervalTime - 1], 0)
        # local min max
        for i in range(1, len(row) - 1):
            temp = 0
            subRow1 = row[i - 1][:4]
            subRow2 = row[i][:4]
            subRow3 = row[i + 1][:4]
            if max(subRow1) < max(subRow2) and max(subRow3) < max(subRow2):
                temp += 1
            elif min(subRow1) > min(subRow2) and min(subRow3) > min(subRow2):
                temp -= 1
            row[i] = np.append(row[i], temp)

        row[MinInd][5] = -2
        row[MaxInd][5] = 2

        #         print(row)

        #         per, diff = persentage(values[ind +intervalTime-1][3],targetVal[0])
        #         print(per, diff)
        rows.append(row)
        #         print(targetVal[0])
        #         break
        #         if 46596==ind:
        #             print("Erorr")
        #             return row,targetVal[1],targetVal[0],values[intervalTime+ind:ind+intervalTime+NextTime,:4]

        #         if targetVal[0] == np.nan:
        #             print("Erorr")
        #             return row,targetVal[1],targetVal[0]

        if tp == None:
            print(ind)
            raise f"Error TP {tp}"

        targetBin.append(np.array(Prediction))
        targetCon.append(np.array(AvgOfTarget))
        timeFeature.append(np.array(timeRow))
        tpItems.append(np.array(tp))
        slItems.append(np.array(sl))
        DesItems.append(np.array(Des))
        rsiItems.append(np.array(rsi))
        BollItems.append(np.array(Boll))

    tpItems = np.array(tpItems)
    slItems = np.array(slItems)
    targetBin = np.array(targetBin)

    DesItems = np.array(DesItems)
    rsiItems = np.array(rsiItems)
    BollItems = np.array(BollItems)

    targetCon = np.array(targetCon)
    rows = np.array(rows)
    timeFeature = np.array(timeFeature)
    return rows, targetBin, targetCon, tpItems, slItems, DesItems, rsiItems, BollItems, timeFeature



x , Bin ,con, tp, sl, Des, rsi, Boll, timeFeature = [],[],[],[],[],[],[],[],[],

df['segment_id'] = df['more_than_1_min'].cumsum()

segments = [group for _, group in df.groupby('segment_id')]
count = 0
df['datetime'] = pd.to_datetime(df['time'])

df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
# Get the day of the week (full name)
df['day_of_week'] = df['datetime'].dt.strftime('%A')

# Alternatively, you can get the integer representation of the day (0=Monday, 6=Sunday)
df['day_of_week_num'] = df['datetime'].dt.dayofweek
for i in segments:
    count+=1
    print(f'{count}/ {len(segments)}')
    xtTemp, BintTemp,contTemp, tptTemp, sltTemp, DestTemp, rsitTemp, BolltTemp, timeFeaturetTemp= preprocessData(df.iloc[:300])# DIFF, PerDiff

    if len(xtTemp) == 0:
        continue

    x.append(xtTemp)
    Bin.append(BintTemp)
    con.append(contTemp)
    tp.append(tptTemp)
    sl.append(sltTemp)
    Des.append(DestTemp)
    rsi.append(rsitTemp)
    Boll.append(BolltTemp)
    timeFeature.append(timeFeaturetTemp)
    break
feature=x

from MT5_TOOL.AI import AImodelForex

Symbol = "EURUSD"
timeMT = 1

setUp =AImodelForex(
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
                    skipFunctionBoundaries= .60,
                    preprocesMethods={

                            "Candle" : True,
                            "MinMax" : True,
                            "rsi" : True,
                            "bollinger_bands" : True,

                    }
)

re = setUp.data_preparing(df.iloc[:120] )
print(feature)
print("---"*10)
print(re['preProcessed'])

print((re['preProcessed']==feature[0][0]).all())

#
