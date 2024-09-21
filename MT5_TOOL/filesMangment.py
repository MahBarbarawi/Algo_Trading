import MetaTrader5 as mt

import pandas as pd

from datetime import datetime, timedelta

import json
import csv
import os

import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

class SymbolInfo:

    def intiateFiles(self):
        path = os.path.join(self.mainPath, )
        os.makedirs(path, exist_ok=True)
        for i in ["setUp", "tradesHistory", "stocks", "LOGS", "model"]:
            path = os.path.join(self.mainPath, i)
            os.makedirs(path, exist_ok=True)

    def __init__(self,
                 mainPath,
                 Symbol,
                 ):

        self.mainPath = os.path.join(mainPath, Symbol)
        self.Symbol = Symbol
        self.currentTime = lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.id = datetime.now().strftime("%Y%m%d%H%M%S")

        self.intiateFiles()

    def SetUpLoaded(self):
        if not self.SetupReady :
            raise "SetUp Is NOT loaded or created use the Set_SetUp or GetSetUp functions"

    def model_setup(self,
                    intervalTime,
                    nextTime,
                    timeframe,
                    pip_value,
                    deviation,
                    sl_pips,
                    tp_pips,
                    skipFunctionBoundries,
                    preprocesMethods,
                    **kwargs
                    ):
        # kwargs Check
        for i in list(preprocesMethods.keys()):
            if i not in [
                "Candle", "MinMax", "rsi", "bollinger_bands",
            ]:
                raise f"error in {i} keys it does not exists"

        # preprocess function used
        for i in [
            "Candle", "MinMax", "rsi", "bollinger_bands"
        ]:
            if preprocesMethods[i] not in [True, False]:
                raise f"error in {preprocesMethods[i]} value it should be only True or False"

        # id Check

        if 'id' in  list(kwargs.keys()):
            self.id = kwargs['id']

        # model SetUp
        self.intervalTime = intervalTime
        self.nextTime = nextTime

        # model Data
        self.preprocesMethods = preprocesMethods
        self.timeframe = timeframe  #

        # model Trades
        self.pip_value = pip_value  # 1 pip for EUR/USD
        self.deviation = deviation
        ## set to determine the TP and SL static or dynamic
        self.sl_pips = sl_pips  # Stop Loss in pips
        self.tp_pips = tp_pips  # Take Profit in pips

        # threshold to skip under it.
        self.skipFunctionBoundries = skipFunctionBoundries

        # Can BUILD MODEL
        self.__SaveLog(
            message="A New Setup is Created",
            typeOFlog= 0
        )
        self.SetupReady = True


    def SaveSetUp(self, name="" , **kwargs):
        self.SetUpLoaded()



        if name == "":
            setUpPath = os.path.join(self.mainPath, "setUp", self.id)
        else :
            setUpPath = os.path.join(self.mainPath, "setUp", name )

        if os.path.exists(setUpPath):
            raise f"the name {name} the Path Name do exists"

        data = {
            "symbol": self.Symbol,
            "intervalTime":self.intervalTime,
            "nextTime":self.nextTime,
            "timeframe": self.timeframe,
            "pip_value": self.pip_value,
            "deviation": self.deviation,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "skipFunctionBoundries": self.skipFunctionBoundries,
            "preprocesMethods": self.preprocesMethods,
            "id": self.id,
        }

        with open(fr'{setUpPath}.json', 'w') as json_file:
            json.dump(data, json_file)
        self.__SaveLog(f"setup Saved {name}", typeOFlog=0)

    def GetCurrentSetUp(self):
        return      {
            "symbol": self.Symbol,
            "intervalTime":self.intervalTime,
            "nextTime":self.nextTime,
            "timeframe": self.timeframe,
            "pip_value": self.pip_value,
            "deviation": self.deviation,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "skipFunctionBoundries": self.skipFunctionBoundries,
            "preprocesMethods ": self.preprocesMethods,

            "id": self.id,
        }

    def GetSetUpNames(self):
        path = os.path.join(self.mainPath, "setUp", )
        return os.listdir(path)

    def GetSetUp(self, name):
        setUpPath = os.path.join(self.mainPath, "setUp", name)


        with open(fr'{setUpPath}', 'r') as json_file:
            retrieved_data = json.load(json_file)



        self.model_setup(

            intervalTime=retrieved_data['intervalTime'],
            nextTime=retrieved_data['nextTime'],
            timeframe=retrieved_data['timeframe'],
            pip_value=retrieved_data['pip_value'],
            deviation=retrieved_data['deviation'],
            sl_pips=retrieved_data['sl_pips'],
            tp_pips=retrieved_data['tp_pips'],
            skipFunctionBoundries=retrieved_data['skipFunctionBoundries'],
            preprocesMethods=retrieved_data['preprocesMethods'],
            id=retrieved_data['id'],

        )
        self.SetupReady = True

        self.__SaveLog(f"New SetUp is loaded {name}", typeOFlog=0)

    def SaveTrades(self, trades):
        self.SetUpLoaded()
        Path = os.path.join(self.mainPath, "tradesHistory", "trades")
        trades['id'] = self.id
        with open(fr'{Path}.json', 'w') as json_file:
            json.dump(trades, json_file)
        self.__SaveLog(f"New Trades is Saved", typeOFlog=1)


    def SaveStock(self, stocks, column_names=None):
        self.SetUpLoaded()


        Path = os.path.join(self.mainPath, "stocks", f"stock_{self.Symbol}_{self.timeframe}.csv")
        indexPath = os.path.join(self.mainPath, "stocks", f"index_{self.Symbol}_{self.timeframe}.josn")


        if not os.path.exists(Path) and column_names is not None:
            with open(Path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(column_names)  # Write column names

        if os.path.exists(indexPath):
            with open(indexPath, 'r') as json_file:
                indexs = json.load(json_file)

        else:
            indexs = {}
        for stock in stocks:
            # search Index
            indexVal =  str(stock[0])
            if not indexs.get(indexVal, False):
                indexs[indexVal]=True

                with open(Path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(stock)


        with open(indexPath, 'w') as json_file:
            json.dump(indexs, json_file, indent=4)
            
        self.__SaveLog(f"New Data is Saved", typeOFlog=1)

    def __saved_file_check(self, fileName , path, type):
        if os.path.exists(path):
            self.__SaveLog(f"file Saved Successfully {fileName}"  ,typeOFlog=type)
        else:
            self.__SaveLog(f"file failed {fileName}"  ,typeOFlog=type)


    def __SaveLog(self, message, typeOFlog):
        LOGStypes = ["setup", "Data_info", "model_building", "Extra",]

        Path = os.path.join(self.mainPath, "LOGS", f"{LOGStypes[typeOFlog]}")
        with open(Path, "a") as log:
            log.write(f"{self.currentTime()},{self.id},{message}\n")



