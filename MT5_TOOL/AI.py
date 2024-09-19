from .filesMangment import SymbolInfo

import MetaTrader5 as mt

import pandas as pd
import numpy as np

import time
from datetime import datetime, timedelta, timezone

from tqdm import tqdm

import os

import warnings

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class AImodelForex(SymbolInfo):
    def __init__(self,
                 RootPath,
                 Symbol,


                 openColumn="open",
                 highColumn="high",
                 lowColumn="low",
                 closeColumn="close",
                 ):

        super().__init__(
            mainPath=RootPath,
            Symbol=Symbol,
        )

        self.SetupReady = False

        self.openColumn = openColumn
        self.highColumn = highColumn
        self.lowColumn = lowColumn
        self.closeColumn = closeColumn

        self.modelPath = os.path.join(self.mainPath, "model")

        self.dataSetItems =['preProcessed', 'time','Des','TechANA']


    def __checkData(self, df):
        required_columns = [self.openColumn, self.highColumn, self.lowColumn, self.closeColumn]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} is not in the DataFrame")

    def __dataLength(self, data ,length=None ):
        if length == None:
            length=  self.DataLength

        if length != len(data):
            raise "error data length do not match"

    def __CreateCandels(self, data):
        mask = data[self.openColumn] < data[self.closeColumn]
        Candle = mask.astype(int)
        return Candle

    def __MinMax_GloLoc(self, data, RangeOfData = 1):
        row = list(data.values)
        newCol = []
        intervalTime = len(data.values)

        Min, Max = np.inf, -np.inf
        MinInd, MaxInd = 0, 0
        # Global Min Max
        for i in range(intervalTime):
            # min
            subRow2 = row[i][:4]
            if min(subRow2) < Min:
                Min = min(subRow2)
                MinInd = i

            # max
            elif max(subRow2) > Max:
                Max = max(subRow2)
                MaxInd = i

        # first row
        newCol.append(0)
        # local min max
        for i in range(1, len(row) - 1):
            temp = 0
            subRow1 = row[i - RangeOfData][:4]
            subRow2 = row[i][:4]
            subRow3 = row[i + RangeOfData][:4]
            if max(subRow1) < max(subRow2) and max(subRow3) < max(subRow2):
                temp += 1
            elif min(subRow1) > min(subRow2) and min(subRow3) > min(subRow2):
                temp -= 1
            newCol.append(temp)
        newCol.append(0)


        newCol[MinInd] = -2
        newCol[MaxInd] = 2

        # last Row


        return newCol

    def __calculate_rsi(self, df, period=10):
        delta = df[self.closeColumn].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values.reshape(-1,1)[-self.intervalTime:,:]

    def __calculate_bollinger_bands(self, df, window=120):

        if self.closeColumn not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column.")

        df['Middle Band'] = df[self.closeColumn].rolling(window=window).mean()
        df['Rolling Std Dev'] = df[self.closeColumn].rolling(window=window).std()
        df['Upper Band'] = df['Middle Band'] + (df['Rolling Std Dev'] * 2)
        df['Lower Band'] = df['Middle Band'] - (df['Rolling Std Dev'] * 2)

        return df[['Middle Band', 'Upper Band', 'Lower Band']]

    def __Val(self, df):
        return df[["open", "high", "low", "close"]].describe().T.iloc[:, 1:]
    def __Main_data_preproces(self, data,datalength= None):
        self.__checkData(data)
        self.__dataLength( data ,length=datalength )

        data = data[[self.openColumn,
                     self.highColumn,
                     self.lowColumn,
                     self.closeColumn, ]]

        ###

        #         print("lengthOfData",len(data))

        data["Candle"] = self.__CreateCandels(data)

        ###
        data["MinMax"] = self.__MinMax_GloLoc(data)

        ###
        row = data.values

        for i in range(len(row)):
            if (row[i].shape != (6,)):
                raise "error"

        #         result = np.concatenate((row, np.array(Candle).reshape(-1, 1)), axis=1)

        return np.array(row)

    def __Time(self, data):
        data['hour'] = data['datetime'].dt.hour
        data['minute'] = data['datetime'].dt.minute
        data['day_of_week_num'] = data['datetime'].dt.dayofweek

        return data[[
            'day_of_week_num',
            'hour',
            'minute',

        ]]

    def __targetExport(self,
                       BuyPoint ,
                       dfTarget ,
                       lengthtarget=60,
                     ) -> "contnious , discret":

        if len(dfTarget) != lengthtarget:
            raise f"length of data do not match {len(dfTarget)}== {lengthtarget}"


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
    def data_preparing_row(self,
                           row,
                           period=10,
                           window=120

                           ):
        self.DataLength =self.intervalTime + max(period,window)


        self.__dataLength(row)

        read_data={}

        read_data["preProcessed"] = self.__Main_data_preproces(row.iloc[:-self.intervalTime].reset_index(drop=True),
                                                               datalength=self.intervalTime)
        # Time
        read_data["time"] = self.__Time(row).values[-self.intervalTime:]

        # DES
        read_data["Des"] = self.__Val(row)

        # rsi
        TechAna =[]
        if self.preprocesMethods['rsi']:
            TechAna.append(self.__calculate_rsi(row,period)
                                     )

        # bollinger_bands

        if self.preprocesMethods['bollinger_bands']:
            TechAna.append(self.__calculate_bollinger_bands(row,window).values[-self.intervalTime:]
                                    )
        read_data["TechANA"]=np.concatenate(TechAna,axis=1)

        return read_data


    def data_preparing_rows(self,
                            rows,
                            period=10,
                            window=120,
                            ):

        dataLength = self.intervalTime + max(period,window)
        rows = rows.sort_values("time",ascending = True)
        rows['datetime'] = pd.to_datetime(rows['time'],unit='s')

        rows['time_diff'] = rows['datetime'].diff()
        rows['more_than_1_min'] = rows['time_diff'] > pd.Timedelta(minutes=1)
        print(rows[['more_than_1_min']].value_counts())
        rows['segment_id'] = rows['more_than_1_min'].cumsum()

        segments = [group for _, group in rows[['segment_id',
                                                'datetime',
                                                "open",
                                                "high",
                                                "low",
                                                "close"
                                                ]].groupby('segment_id')]
        Data = {key: [] for key in self.dataSetItems}
        count= 0
        targets = {key: [] for key in ['diff',
            'prediction',
            'BuyPoint',
            'tp',
            'sl',]}
        ValuesTarget = 0

        for SubData in segments:
            count+=1
            print(f"{count}/{len(segments)}")
            for length in tqdm(range(len(SubData))):

                if  len(SubData)-(dataLength+self.nextTime)  < length:
                    break
                subRow = SubData.iloc[length:length+dataLength]
                subTarget = SubData[["open", "high", "low", "close"]].iloc[length+dataLength:length+dataLength+self.nextTime]
                re = self.data_preparing_row(
                    row = subRow,
                    period=period,
                    window=window,

                )

                diff,prediction, BuyPoint, tp, sl, = self.__targetExport(
                    BuyPoint=subRow.values[-1][-1],
                    dfTarget=subTarget,

                )
                for DataSetName in self.dataSetItems:
                    Data[DataSetName].append(re[DataSetName])
                for DataSetName,val in zip(['diff','prediction','BuyPoint','tp','sl',]
                                       ,[diff,prediction, BuyPoint, tp, sl,]):
                    ValuesTarget+=1
                    targets[DataSetName].append(val)
            break

        print(f"ValuesTarget = {ValuesTarget}|| ValuesTarget = {ValuesTarget/5}")
        for DataSetName in self.dataSetItems:
            Data[DataSetName] = np.array(Data[DataSetName])

        for DataSetName in ['diff', 'prediction', 'BuyPoint', 'tp', 'sl', ]:
            targets[DataSetName] =np.array(targets[DataSetName] )

        return Data,targets

        # print(Data.keys())
        # print("preProcessed",Data['preProcessed'])
        # print("time",Data['time'])
        # print("rsi",Data['rsi'])
        # print("bollinger_bands",Data['bollinger_bands'])
        # print(re,target)


