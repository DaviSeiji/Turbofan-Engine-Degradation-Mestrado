import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage
from src.xai.xai_baseExplainer import BaseExplainer

class GradCamExplainer(BaseExplainer):
    def __init__(self, model_path, x_path, y_path, conv_layer_name=None):
        super().__init__(model_path, x_path, y_path)
        
        if conv_layer_name is None:
            self.conv_layer_name = self._find_last_conv_layer()
            print(f"Camada Convolucional identificada automaticamente: '{self.conv_layer_name}'")
        else:
            self.conv_layer_name = conv_layer_name

    def _find_last_conv_layer(self):
        """Varre a arquitetura do modelo buscando a última camada Conv."""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("Nenhuma camada convolucional encontrada. Verifique a arquitetura.")

    def explain_instance(self, index):
        """Calcula o mapa de calor (Grad-CAM) fazendo um forward pass manual."""
        instance = self.get_instance(index)
        inputs_tensor = tf.cast(instance, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(inputs_tensor)
            
            x = inputs_tensor
            conv_outputs = None
            
            for layer in self.model.layers:
                x = layer(x, training=False) 

                if layer.name == self.conv_layer_name:
                    conv_outputs = x
                    
            predictions = x
            loss = predictions[:, 0]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        
        max_heat = tf.math.reduce_max(heatmap)
        if max_heat > 0:
            heatmap /= max_heat
            
        heatmap = heatmap.numpy()

        total_timesteps = self.X_test.shape[1]
        zoom_factor = total_timesteps / len(heatmap)
        heatmap_resized = scipy.ndimage.zoom(heatmap, zoom_factor, order=1)
        
        return heatmap_resized

    def plot_temporal_heatmap(self, index):
        """Plota o mapa de calor (Grad-CAM) sobre o sinal dos sensores."""
        heatmap = self.explain_instance(index)
        instance = self.get_instance(index)[0]
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)
        total_timesteps = self.X_test.shape[1]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [0.5, 3]})
        
        extent = [0, total_timesteps - 1, 0, 1]
        ax1.imshow(heatmap[np.newaxis, :], cmap='coolwarm', aspect='auto', extent=extent)
        ax1.set_title(f'Grad-CAM Temporal Heatmap - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}')
        ax1.set_yticks([])
        ax1.set_xticks(range(total_timesteps))
        ax1.set_xticklabels([])
        
        for i in range(instance.shape[1]):
            ax2.plot(instance[:, i], alpha=0.6, linewidth=1.5)
            
        ax2.set_xlabel('Janela de Tempo (30 ciclos)')
        ax2.set_ylabel('Sensores (Normalizados)')
        ax2.set_xlim(0, total_timesteps - 1)
        ax2.set_xticks(range(total_timesteps))
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.show()

    def plot_time_importance_bar(self, index):
        """Plota um gráfico de barras com a importância de cada ciclo."""
        heatmap = self.explain_instance(index)
        
        ciclos_labels = [f"C {i}" for i in range(len(heatmap))]
        
        plt.figure(figsize=(10, 6))
        plt.bar(ciclos_labels, heatmap, color='steelblue')
        plt.title(f'Impacto Agregado por Ciclo de Tempo (Grad-CAM) - Amostra {index}')
        plt.xlabel('Janela de Tempo (30 ciclos)')
        plt.ylabel('Ativação Grad-CAM')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_global_summary(self, num_samples=100):
        """
        Plota a visão global (Summary Plot) do Grad-CAM.
        Avalia um lote de amostras para entender quais ciclos de tempo 
        são globalmente mais importantes para a CNN.
        """
        print(f"Calculando Grad-CAM para {num_samples} amostras (isso pode levar alguns segundos)...")
        
        all_heatmaps = []
        for i in range(num_samples):
            all_heatmaps.append(self.explain_instance(i))
            
        mean_heatmap = np.mean(all_heatmaps, axis=0)
        total_timesteps = self.X_test.shape[1]
        
        plt.figure(figsize=(10, 4))
        extent = [0, total_timesteps - 1, 0, 1]
        
        plt.imshow(mean_heatmap[np.newaxis, :], cmap='coolwarm', aspect='auto', extent=extent)
        plt.title(f"Visão Global do Modelo ({num_samples} Amostras)\nFoco Médio da CNN ao Longo do Tempo")
        plt.xlabel('Janela de Tempo (30 ciclos)')
        plt.yticks([])
        plt.xticks(range(total_timesteps))
        plt.colorbar(label='Ativação Média')
        
        plt.tight_layout()
        plt.show()