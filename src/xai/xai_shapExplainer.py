import os
import warnings

# Deve ser definido antes de qualquer importacao do TensorFlow/SHAP.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import shap
import numpy as np
import matplotlib.pyplot as plt
from src.xai.xai_baseExplainer import BaseExplainer 


warnings.filterwarnings(
    "ignore",
    message=r"The structure of `inputs` doesn't match the expected structure\.",
    category=UserWarning,
    module=r"keras\.src\.models\.functional",
)

class ShapExplainer(BaseExplainer):
    def __init__(self, model_path, x_path, y_path, background_size=100):
        super().__init__(model_path, x_path, y_path)
        
        self.background_data = self.X_test[:background_size].astype(np.float32)

        self.explainer = shap.GradientExplainer(self.model, self.background_data)

    def explain_instance(self, index):

        instance = self.get_instance(index)

        shap_values = self.explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            return shap_values[0]
        return shap_values

    def plot_temporal_heatmap(self, index):
        """Plota o mapa de calor (Tempo x Sensor) para uma amostra."""
        shap_values = self.explain_instance(index)

        shap_matrix_2d = shap_values[0, :, :, 0]
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)
        num_sensores = len(self.feature_names)

        plt.figure(figsize=(10, 6))
        plt.imshow(shap_matrix_2d.T, aspect='auto', cmap='coolwarm') 
        plt.title(f'SHAP Temporal Heatmap - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}')
        plt.xlabel('Janela de Tempo (30 ciclos)')
        plt.ylabel('Sensores')
        plt.colorbar(label='Valor SHAP (Impacto no RUL)')
        plt.yticks(ticks=np.arange(num_sensores), labels=self.feature_names)
        plt.tight_layout()
        plt.show()

    def plot_sensor_importance_bar(self, index):
        """Plota um gráfico de barras com a importância total de cada sensor na amostra."""
        shap_values = self.explain_instance(index)
        shap_matrix_2d = shap_values[0, :, :, 0]
        
        sensor_importance = np.sum(np.abs(shap_matrix_2d), axis=0)
        sensores_labels = self.feature_names
        sorted_idx = np.argsort(sensor_importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(sensores_labels)[sorted_idx], sensor_importance[sorted_idx], color='steelblue')
        plt.title(f'Impacto Agregado por Sensor (SHAP) - Amostra {index}')
        plt.xlabel('Soma Absoluta dos Valores SHAP')
        plt.tight_layout()
        plt.show()

    def plot_global_summary(self, num_samples=100):
        """
        Plota o Summary Plot clássico do SHAP (Visão Global).
        Avalia um lote de amostras para entender quais sensores 
        são globalmente mais importantes para o modelo.
        """
        print(f"Calculando SHAP para {num_samples} amostras (isso pode levar alguns segundos)...")
        
        instances = self.X_test[:num_samples].astype(np.float32)
        shap_values = self.explainer.shap_values(instances)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        shap_values_3d = shap_values[:, :, :, 0]
        
        shap_values_2d = np.mean(shap_values_3d, axis=1)
        instances_2d = np.mean(instances, axis=1)
        
        sensores_labels = self.feature_names

        plt.figure(figsize=(10, 6))
        plt.title(f"Visão Global do Modelo ({num_samples} Amostras)")

        shap.summary_plot(shap_values_2d, instances_2d, feature_names=sensores_labels, show=False)
        
        plt.tight_layout()
        plt.show()
