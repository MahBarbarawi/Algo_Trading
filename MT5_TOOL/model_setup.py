import json
import os.path

import pandas as pd

from .AI import AImodelForex

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.utils import shuffle

from keras.models import Model


class model_building(AImodelForex):
    def __init__(self,
                 RootPath,
                 Symbol,
                 ):

        super().__init__(RootPath, Symbol)  # Ensure parent class is initialized

    def __load_data(self, ):
        Path = os.path.join(self.mainPath, "stocks", f"stock_{self.Symbol}_{self.timeframe}.csv")
        df = pd.read_csv(Path)
        preprocessedData, target = self.data_preparing_rows(df)

        self.X = preprocessedData['preProcessed']
        self.timeFeature = preprocessedData['time']
        self.Des = preprocessedData['Des']
        self.TechANA = preprocessedData['TechANA']

        self.yBin = target['prediction']
        self.ydiff = target['diff']
        self.ytp = target['tp']
        self.ysl = target['sl']

    def __drop_nan(self, arr):
        # Check if the input is a NumPy array
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a NumPy array")

        # Filter the array by keeping only non-NaN values
        clean_arr = arr[~np.isnan(arr)]
        indices = np.where(np.isnan(arr))

        return clean_arr, indices

    def remove_elements(arr, elements_to_remove):
        # Use np.isin to check if elements in arr are in elements_to_remove
        mask = ~np.isin(arr, elements_to_remove)

        # Filter the array to remove the elements
        filtered_arr = arr[mask]

        return filtered_arr

    def __binaryPreparing_spliting(self):

        #Data Shapes
        self.input_shape_X = self.X.shape[1:]  # Shape of the first input
        self.input_shape_timeFeature = self.timeFeature.shape[1:]  # Shape of the second input
        self.input_shape_Des = self.Des.shape[1:]  # Shape of the third input
        self.input_shape_TechANA = self.TechANA.shape[1:]  # Shape of the fourth input

        #Number of Classes It got
        self.classesNum = len(np.unique(self.yBin))
        # Data Splits
        train_size = int(len(self.X) * 0.8)
        validation_size = int(len(self.X) * 0.1)

        # Split the preProcessed
        self.X_train_preProcessed_binaryModel = self.X[:train_size]
        self.X_val_preProcessed_binaryModel = self.X[train_size:train_size + validation_size]
        self.X_test_preProcessed_binaryModel = self.X[train_size + validation_size:]

        # Split the Description
        self.X_train_Description_binaryModel = self.Des[:train_size]
        self.X_val_Description_binaryModel = self.Des[train_size:train_size + validation_size]
        self.X_test_Description_binaryModel = self.Des[train_size + validation_size:]

        # Split the Time
        self.X_train_time_binaryModel = self.timeFeature[:train_size]
        self.X_val_time_binaryModel = self.timeFeature[train_size:train_size + validation_size]
        self.X_test_time_binaryModel = self.timeFeature[train_size + validation_size:]

        # Split the TechANA
        self.X_train_TechANA_binaryModel = self.TechANA[:train_size]
        self.X_val_TechANA_binaryModel = self.TechANA[train_size:train_size + validation_size]
        self.X_test_TechANA_binaryModel = self.TechANA[train_size + validation_size:]

        # Split the target points Binary
        self.y_train_Bin_binaryModel = self.yBin[:train_size]
        self.y_val_Bin_binaryModel = self.yBin[train_size:train_size + validation_size]
        self.y_test_Bin_binaryModel = self.yBin[train_size + validation_size:]

        # encoded Classes
        self.y_train_Bin_encoded = to_categorical(self.y_train_Bin_binaryModel, num_classes=self.classesNum)
        self.y_val_Bin_encoded = to_categorical(self.y_val_Bin_binaryModel, num_classes=self.classesNum)
        self.y_test_Bin_encoded = to_categorical(self.y_test_Bin_binaryModel, num_classes=self.classesNum)

        # Split the target points diff
        self.y_train_diff = self.ydiff[:train_size]
        self.y_val_diff = self.ydiff[train_size:train_size + validation_size]
        self.y_test_diff = self.ydiff[train_size + validation_size:]

        # Split the target points tp
        self.y_train_tp = self.ytp[:train_size]
        self.y_val_tp = self.ytp[train_size:train_size + validation_size]
        self.y_test_tp = self.ytp[train_size + validation_size:]

        # Split the target points sl
        self.y_train_sl = self.ysl[:train_size]
        self.y_val_sl = self.ysl[train_size:train_size + validation_size]
        self.y_test_sl = self.ysl[train_size + validation_size:]

    def __Data_Preparing_Tp_Sl_all(self):
        self.targetValues = {}
        self.dataValues = {}
        for targetVal, typeOfRun in zip([0, 1], ['bear', 'bull']):
            ind = np.where(self.yBin == targetVal)[0]
            self.targetValues[f"{typeOfRun}_sl"], indices = self.__drop_nan(self.ysl[ind])
            self.targetValues[f"{typeOfRun}_tp"], indices = self.__drop_nan(self.ysl[ind])

            X = np.delete(self.X[ind], indices[0], axis=0)
            timeFeature = np.delete(self.timeFeature[ind], indices[0], axis=0)
            Des = np.delete(self.Des[ind], indices[0], axis=0)
            TechANA = np.delete(self.TechANA[ind], indices[0], axis=0)

            self.dataValues[typeOfRun] = (X, timeFeature, Des, TechANA)

    def custom_split(self, data, train_size=0.7, validation_size=0.15, test_size=0.15):
        assert np.isclose(train_size + validation_size + test_size,
                          1.0), "Train, validation, and test sizes must add up to 1"

        # Get the length of the data
        data_length = len(data)

        # Calculate indices for splitting
        train_end = int(train_size * data_length)
        val_end = int((train_size + validation_size) * data_length)

        # Split the data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    def train_model_Tp_Sl(self,  ):
        self.__Data_Preparing_Tp_Sl_all()
        model_name = {}
        for typeOfRun in ['bear', 'bull']:

            for targetVal in [f"{typeOfRun}_sl", f"{typeOfRun}_tp"]:
                self._SaveLog(message=f"start building {targetVal}", typeOFlog=2)

                features, timeFeature, Des, TechAna = self.dataValues[typeOfRun]

                X_train, X_val, X_test = self.custom_split(features, 0.7, 0.15, 0.15)

                X_train2, X_val2, X_test2 = self.custom_split(Des, 0.7, 0.15, 0.15)

                X_train3, X_val3, X_test3 = self.custom_split(TechAna, 0.7, 0.15, 0.15)

                X_train4, X_val4, X_test4 = self.custom_split(timeFeature, 0.7, 0.15, 0.15)

                y_train, y_val, y_test = self.custom_split(self.targetValues[targetVal], 0.7, 0.15, 0.15)

                # Assuming the shape of the four input datasets
                input_shape_1 = features.shape[1:]  # Shape of the first input
                input_shape_2 = Des.shape[1:]  # Shape of the second input
                input_shape_3 = TechAna.shape[1:]  # Shape of the third input
                input_shape_4 = timeFeature.shape[1:]  # Shape of the fourth input

                # Define the Input Layers
                input_1 = Input(shape=input_shape_1)
                input_2 = Input(shape=input_shape_2)
                input_3 = Input(shape=input_shape_3)
                input_4 = Input(shape=input_shape_4)

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

                # Output layer for regression
                output = Dense(1, activation='linear')(x)  # Linear activation for continuous output

                # Create the model
                model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=output)

                # Compile the model for regression
                model.compile(optimizer=Adam(), loss='mean_squared_error',
                              metrics=['mean_squared_error', 'mean_absolute_error'])

                # Define early stopping callback
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                history = model.fit(
                    [X_train, X_train2, X_train3, X_train4], y_train,  # Continuous values for y_train
                    validation_data=([X_test, X_test2, X_test3, X_test4], y_test),  # Continuous values for y_test
                    epochs=1,
                    batch_size=32,
                    callbacks=[early_stopping]
                )

                path_model = os.path.join(path, f"{targetVal}.h5")
                model.save(path_model)
                self._saved_file_check(fileName=f"{targetVal}", path=path_model, typeOFlog=2)
                model_name[targetVal] = path_model
                # Make predictions on the test set
                print("-" * 20, "train")
                self.__reg_evaluation(model, X_train, X_train2, X_train3, X_train4, y_train)

                print("-" * 20, "val")
                self.__reg_evaluation(model, X_val, X_val2, X_val3, X_val4, y_val)

                print("-" * 20, "test")
                self.__reg_evaluation(model, X_test, X_test2, X_test3, X_test4, y_test)

        return model_name

    def __reg_evaluation(self, model, X_test, X_test2, X_test3, X_test4, y_test):
        y_pred = model.predict([X_test, X_test2, X_test3, X_test4])

        # Calculate R² score
        r2 = r2_score(y_test, y_pred)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)

        # Print the evaluation metrics
        print(f'R² Score: {r2:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')

    def train_model(self, path):
        self.__binaryPreparing_spliting()
        self._SaveLog(message="building Binary model", typeOFlog=2)

        # Define the Input Layers
        input_1 = Input(shape=self.input_shape_X)
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
            [self.X_train_preProcessed_binaryModel,
             self.X_train_Description_binaryModel,
             self.X_train_time_binaryModel,
             self.X_train_TechANA_binaryModel],

            self.y_train_Bin_encoded,
            validation_data=([
                            self.X_val_preProcessed_binaryModel,
                            self.X_val_Description_binaryModel,
                            self.X_val_time_binaryModel,
                            self.X_val_TechANA_binaryModel,
                ],
                            self.y_val_Bin_encoded),  # Assuming you have X_test, Des_test, TechAna_test
            epochs=1,
            batch_size=32,
            callbacks=[early_stopping]
        )

        model.summary()
        # os.makedirs(os.path.dirname("), exist_ok=True)

        path = os.path.join(path, f"main_binary_model.h5")
        model.save(path)
        self._saved_file_check(fileName=f"Binary_model", path=path, typeOFlog=2)

        self.__Cat_evaluation(model, self.y_train_Bin_encoded,
                              self.y_val_Bin_encoded,
                              self.y_test_Bin_encoded)
        return path

    def __Cat_evaluation(self, model, y, ytest, yval):
        train_loss, train_accuracy, train_auc = model.evaluate([self.X_train_preProcessed_binaryModel,
                                                                self.X_train_Description_binaryModel,
                                                                self.X_train_time_binaryModel,
                                                                self.X_train_TechANA_binaryModel],
                                                               y)
        test_loss, test_accuracy, test_auc = model.evaluate([self.X_test_preProcessed_binaryModel,
                                                             self.X_test_Description_binaryModel,
                                                             self.X_test_time_binaryModel,
                                                             self.X_test_TechANA_binaryModel],
                                                            yval)
        val_loss, val_accuracy, val_auc = model.evaluate([self.X_val_preProcessed_binaryModel,
                                                          self.X_val_Description_binaryModel,
                                                          self.X_val_time_binaryModel,
                                                          self.X_val_TechANA_binaryModel],
                                                         ytest)

        print(f'Train Loss: {train_loss},` Train Accuracy: {train_accuracy}, Train AUC: {train_auc}')
        print(f'val Loss: {test_loss}, val Accuracy: {test_accuracy}, val AUC: {test_auc}')
        print(f'test Loss: {val_loss}, test Accuracy: {val_accuracy}, test AUC: {val_auc}')

    def build_AI(self):
        self.__load_data()

        path = os.path.join(self.mainPath,
                            "model",
                            self.id)


        main_model_path = self.train_model(path)
        models_path = self.train_model_Tp_Sl(path)

        main_models_setup = {
            "id":1,
            "path": path,
            "main_model": main_model_path,
            "bear_tp": models_path["bear_tp"],
            "bear_sl": models_path["bear_sl"],
            "bull_tp": models_path["bull_tp"],
            "bull_sl": models_path["bull_sl"],
            "model_preprocess": self.preprocesMethods,
            "model_parameters": {
                "intervalTime": self.intervalTime,
                "nextTime": self.nextTime,
                "timeframe": self.timeframe,
                "skipFunctionBoundaries": self.skipFunctionBoundries,

            }
        }
        setPath = os.path.join(path,"models_setUp.json")
        with open(fr'{setPath}.json', 'w') as json_file:
            json.dump(main_models_setup, json_file)

        return main_models_setup

    def retrain_Model(self):
        pass
