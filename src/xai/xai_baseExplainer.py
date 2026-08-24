import os

# Reduz mensagens informativas do TensorFlow; deve vir antes do import abaixo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

class BaseExplainer():
    def __init__(self, model_path, x_path, y_path):
        self.model = tf.keras.models.load_model(model_path)

        x_data = np.load(x_path)
        self.X_test = x_data['dados'] 

        if 'features' in x_data:
            self.feature_names = x_data['features']
        else:
            self.feature_names = [f"Sensor {i+1}" for i in range(self.X_test.shape[2])]
        
        y_data = np.load(y_path)
        self.y_test = y_data.get('dados', y_data.get('arr_0')) 

    def get_instance(self, index):
        return self.X_test[index:index+1]

    def get_prediction(self, index):
        instance = self.get_instance(index)
        prediction = self.model.predict(instance, verbose=0)
        return prediction[0][0]

    def get_true_rul(self, index):
        return self.y_test[index]
