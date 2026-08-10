import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.xai.xai_baseExplainer import BaseExplainer

class SaliencyMapsExplainer(BaseExplainer):
    def __init__(self, model_path, x_path, y_path):
        super().__init__(model_path, x_path, y_path)

    def explain_instance(self, index):
        """Calcula o Saliency Map (Vanilla Gradient) para uma amostra."""
        instance = self.get_instance(index)
        inputs_tensor = tf.cast(instance, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(inputs_tensor)

            predictions = self.model(inputs_tensor, training=False)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, inputs_tensor)

        saliency = tf.abs(grads)
        saliency = saliency[0].numpy()

        max_val = np.max(saliency)
        if max_val > 0:
            saliency /= max_val
            
        return saliency

    def plot_temporal_heatmap(self, index):
        """Plota o mapa de calor (Tempo x Sensor) do Saliency Map para uma amostra."""
        saliency_matrix = self.explain_instance(index)
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)
        num_sensores = len(self.feature_names)

        plt.figure(figsize=(10, 6))
        # Transpõe a matriz para (Sensores, Tempo) mantendo o padrão visual
        plt.imshow(saliency_matrix.T, aspect='auto', cmap='coolwarm') 
        plt.title(f'Saliency Map (Vanilla Gradients) - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}')
        plt.xlabel('Janela de Tempo (30 ciclos)')
        plt.ylabel('Sensores')
        plt.colorbar(label='Sensibilidade (Gradiente Absoluto Normalizado)')
        plt.yticks(ticks=np.arange(num_sensores), labels=self.feature_names)
        plt.tight_layout()
        plt.show()

    def plot_sensor_importance_bar(self, index):
        """Plota um gráfico de barras com a sensibilidade total de cada sensor na amostra."""
        saliency_matrix = self.explain_instance(index)
        
        sensor_importance = np.sum(saliency_matrix, axis=0)
        sensores_labels = self.feature_names
        sorted_idx = np.argsort(sensor_importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(sensores_labels)[sorted_idx], sensor_importance[sorted_idx], color='steelblue')
        plt.title(f'Sensibilidade Agregada por Sensor (Saliency) - Amostra {index}')
        plt.xlabel('Soma dos Gradientes Absolutos')
        plt.tight_layout()
        plt.show()

    def plot_global_summary(self, num_samples=100):
        """
        Plota a visão global dos Saliency Maps.
        Avalia um lote de amostras para entender o comportamento geral dos gradientes.
        """
        print(f"Calculando Saliency Maps para {num_samples} amostras (isso pode levar alguns segundos)...")
        
        all_saliency = []
        for i in range(num_samples):
            all_saliency.append(self.explain_instance(i))

        mean_saliency = np.mean(all_saliency, axis=0)
        num_sensores = len(self.feature_names)

        plt.figure(figsize=(10, 6))
        plt.imshow(mean_saliency.T, aspect='auto', cmap='coolwarm')
        plt.title(f"Visão Global de Sensibilidade ({num_samples} Amostras)\nSaliency Map Médio")
        plt.xlabel('Janela de Tempo (30 ciclos)')
        plt.ylabel('Sensores')
        plt.colorbar(label='Sensibilidade Média')
        plt.yticks(ticks=np.arange(num_sensores), labels=self.feature_names)
        
        plt.tight_layout()
        plt.show()