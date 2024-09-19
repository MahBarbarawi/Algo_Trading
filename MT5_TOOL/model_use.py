import numpy as np
from keras.src.saving import load_model

from .AI import AImodelForex


class model_load(AImodelForex):
    def __init__(self):
        super().__init__()

        # model
    def loadModel(self, Model_path):
        self.model = load_model(Model_path)

    #
    def PredictProb(self, data):
        dataPre = self.preproces(data)
        dataPre = np.expand_dims(dataPre, axis=0)
        prediction = self.model.predict(dataPre, verbose=0)
        return prediction

    def Predict(self, data, thershould=0.5):
        prePro = self.PredictProb(data)
        return prePro > thershould


