import json
import os
import tensorflow as tf

import numpy as np
from keras.src.saving import load_model

from .AI import AImodelForex
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class model_load(AImodelForex):
    def __init__(self,
                 RootPath,
                 Symbol,):
        super().__init__(
            RootPath,
            Symbol,
        )

        # model
    def loadModels(self,):
        loadPath = os.path.join(self.mainPath,'model',self.id,"models_setUp")
        with open(fr'{loadPath}.json', 'r') as json_file:
            main_models_setup = json.load( json_file)
        self.main_model    = load_model(main_models_setup['main_model'])
        self.bear_tp_model = load_model(main_models_setup['bear_tp'])
        self.bear_sl_model = load_model(main_models_setup['bear_sl'])
        self.bull_tp_model = load_model(main_models_setup['bull_tp'])
        self.bull_sl_model = load_model(main_models_setup['bull_sl'])

    def predictions(self, data):
        prepareRow = self.data_preparing_row(data)
        d1 ,d2, d3, d4  = prepareRow['preProcessed'],prepareRow['Des'],prepareRow['time'],prepareRow['TechANA'],
        print(d1.shape ,d2.shape, d3.shape, d4.shape )

        predictions = self.main_model.predict(
            [
               d1.reshape(-1,d1.shape[0], d1.shape[1]) ,
               d2.reshape(-1,d2.shape[0], d2.shape[1]) ,
               d3.reshape(-1,d3.shape[0], d3.shape[1]) ,
               d4.reshape(-1,d4.shape[0], d4.shape[1]) ,

                    ]
        )
        return predictions




