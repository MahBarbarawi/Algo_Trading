import os.path

import pandas as pd

from .AI import AImodelForex

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.utils import shuffle

from keras.models import Model

class train_model(AImodelForex):
    def __init__(self,
                 RootPath,
                 Symbol,
                 ):
        super().__init__(
                RootPath,
                 Symbol,
        )



    def __load_data(self,):
        Path = os.path.join(self.mainPath, "stocks", f"stock_{self.Symbol}_{self.timeframe}.csv")
        df = pd.read_csv(Path)
        preprocessedData , target  = self.data_preparing_rows(df)


        X       = preprocessedData['preProcessed']
        timeFeature    = preprocessedData['time']
        Des     = preprocessedData['Des']
        TechANA = preprocessedData['TechANA']

        yBin    = target['prediction']
        ydiff   = target['diff']
        ytp     = target['tp']
        ysl     = target['sl']

        #Data Shapes
        self.input_shape_X           = X.shape[1:]  # Shape of the first input
        self.input_shape_timeFeature = timeFeature.shape[1:] # Shape of the second input
        self.input_shape_Des         = Des.shape[1:] # Shape of the third input
        self.input_shape_TechANA     = TechANA.shape[1:]   # Shape of the fourth input

        #Number of Classes It got
        self.classesNum =len(np.unique(yBin))

        # Data Splits
        train_size = int(len(X) * 0.8)
        validation_size = int(len(X) * 0.1)

        # Split the preProcessed
        self.X_train_preProcessed = X[:train_size]
        self.X_val_preProcessed   = X[train_size:train_size + validation_size]
        self.X_test_preProcessed  = X[train_size + validation_size:]

        # Split the Description
        self.X_train_Description = Des[:train_size]
        self.X_val_Description   = Des[train_size:train_size + validation_size]
        self.X_test_Description  = Des[train_size + validation_size:]

        # Split the Time
        self.X_train_time = timeFeature[:train_size]
        self.X_val_time   = timeFeature[train_size:train_size + validation_size]
        self.X_test_time  = timeFeature[train_size + validation_size:]

        # Split the TechANA
        self.X_train_TechANA = TechANA[:train_size]
        self.X_val_TechANA   = TechANA[train_size:train_size + validation_size]
        self.X_test_TechANA  = TechANA[train_size + validation_size:]



        # Split the target points Binary
        self.y_train_Bin = yBin[:train_size]
        self.y_val_Bin   = yBin[train_size:train_size + validation_size]
        self.y_test_Bin  = yBin[train_size + validation_size:]


        # encoded Classes
        self.y_train_Bin_encoded = to_categorical(self.y_train_Bin, num_classes=self.classesNum)
        self.y_val_Bin_encoded   = to_categorical(self.y_val_Bin, num_classes=self.classesNum)
        self.y_test_Bin_encoded  = to_categorical(self.y_test_Bin, num_classes=self.classesNum)

        print(self.X_train_preProcessed.shape)
        print(self.y_train_Bin_encoded.shape)

        # Split the target points diff
        self.y_train_diff = ydiff[:train_size]
        self.y_val_diff   = ydiff[train_size:train_size + validation_size]
        self.y_test_diff  = ydiff[train_size + validation_size:]

        # Split the target points tp
        self.y_train_tp = ytp[:train_size]
        self.y_val_tp   = ytp[train_size:train_size + validation_size]
        self.y_test_tp  = ytp[train_size + validation_size:]

        # Split the target points sl
        self.y_train_sl = ysl[:train_size]
        self.y_val_sl   = ysl[train_size:train_size + validation_size]
        self.y_test_sl  = ysl[train_size + validation_size:]


    def __next_n_candles(self):
        pass

    def __next_Baer_SL(self):
        pass

    def __next_Bear_TP(self):
        pass

    def __next_Bull_SL(self):
        pass

    def __next_Bull_TP(self):
        pass

    def train_model(self):
        self.__load_data()

        # Define the Input Layers
        input_1 = Input(shape=self.input_shape_X  )
        input_2 = Input(shape=self.input_shape_Des)
        input_3 = Input(shape=self.input_shape_timeFeature)
        input_4 = Input(shape=self.input_shape_TechANA)

        # LSTM layers with Dropout for the first input
        x1 = LSTM(64)(input_1)
        x1 = Dropout(0.5)(x1)  # Dropout with 50% rate

        # LSTM layers with Dropout for the second input
        x2 = LSTM(64)(input_2)
        x2 = Dropout(0.5)(x2)  # Dropout with 50% rate

        # LSTM layers with Dropout for the third input
        x3 = LSTM(64)(input_3)
        x3 = Dropout(0.5)(x3)  # Dropout with 50% rate

        # LSTM layers with Dropout for the fourth input
        x4 = LSTM(64)(input_4)
        x4 = Dropout(0.5)(x4)  # Dropout with 50% rate

        # Concatenate the outputs of the LSTM streams
        merged = Concatenate()([x1, x2, x3, x4])

        # Further processing after merging with Dropout
        x = Dense(64, activation='relu')(merged)
        x = Dropout(0.5)(x)  # Dropout with 50% rate
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.5)(x)  # Dropout with 50% rate

        # Output layer
        output = Dense(self.classesNum, activation='sigmoid')(x)  # Assuming a binary classification task

        # Create the model
        model = Model(inputs=[input_1,
                              input_2,
                              input_3,
                              input_4], outputs=output)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', AUC(name='auc')])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            [self.X_train_preProcessed,
             self.X_train_Description,
             self.X_train_time,
             self.X_train_TechANA],

            self.y_train_Bin_encoded,
            validation_data=([self.X_val_preProcessed,
                              self.X_val_Description,
                              self.X_val_time,
                              self.X_val_TechANA],
                             self.y_val_Bin_encoded),           # Assuming you have X_test, Des_test, TechAna_test
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping]
        )

        model.summary()
        path = os.path.join(self.mainPath,
                            "model",
                            f'model_{self.intervalTime}_{self.nextTime}_{self.timeframe}_{self.currentTime()}.h5')
        # os.makedirs(os.path.dirname("), exist_ok=True)

        model.save(rf"{path}")
        self.AccurcyVal( model, self.y_train_Bin_encoded,
                         self.y_val_Bin_encoded,
                         self.y_test_Bin_encoded)

    def AccurcyVal(self,model,y,ytest,yval):
        train_loss, train_accuracy, train_auc = model.evaluate([ self.X_train_preProcessed,
                                                                 self.X_train_Description,
                                                                 self.X_train_time,
                                                                 self.X_train_TechANA],
                                                               y)
        test_loss, test_accuracy, test_auc = model.evaluate([self.X_test_preProcessed,
                                                             self.X_test_Description,
                                                             self.X_test_time,
                                                             self.X_test_TechANA],
                                                            yval)
        val_loss, val_accuracy, val_auc = model.evaluate([  self.X_val_preProcessed,
                                                            self.X_val_Description,
                                                            self.X_val_time,
                                                            self.X_val_TechANA],
                                                           ytest)

        print(f'Train Loss: {train_loss},` Train Accuracy: {train_accuracy}, Train AUC: {train_auc}')
        print(f'val Loss: {test_loss}, val Accuracy: {test_accuracy}, val AUC: {test_auc}')
        print(f'test Loss: {val_loss}, test Accuracy: {val_accuracy}, test AUC: {val_auc}')


    def retrain_Model(self):
        pass