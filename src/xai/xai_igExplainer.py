import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from src.xai.xai_baseExplainer import BaseExplainer 

class IntegratedGradientsExplainer(BaseExplainer):
    def __init__(self, model_path, x_test_path, y_test_path, baseline=None):

        super().__init__(model_path, x_test_path, y_test_path)

        self.sample_shape = self.X_test[0].shape
        
        if baseline is None:
            self.baseline = tf.zeros(shape=self.sample_shape)
        else:
            self.baseline = tf.cast(baseline, dtype=tf.float32)

    @tf.function
    def compute_gradients(self, inputs):

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.model(inputs)
        return tape.gradient(predictions, inputs)

    def generate_interpolants(self, sample, m_steps=50):

        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
        alphas = tf.cast(alphas, dtype=tf.float32)

        alphas_expanded = tf.reshape(alphas, (-1, 1, 1))
        
        baseline_expanded = tf.expand_dims(self.baseline, axis=0)
        sample_expanded = tf.expand_dims(sample, axis=0)

        interpolated = baseline_expanded + alphas_expanded * (sample_expanded - baseline_expanded)
        return interpolated

    def explain_instance(self, index, m_steps=50):

        sample = tf.cast(self.X_test[index], dtype=tf.float32)

        interpolated_inputs = self.generate_interpolants(sample, m_steps)

        gradients = self.compute_gradients(interpolated_inputs)

        avg_gradients = tf.reduce_mean(gradients[:-1], axis=0)

        integrated_gradients = (sample - self.baseline) * avg_gradients
        
        return integrated_gradients.numpy()

    def plot_heatmap(self, index, m_steps=50):
        """
        Gera um heatmap temporal para visualizar o impacto ao longo dos 30 ciclos.
        Visualmente comparável ao Heatmap gerado pelo SHAP.
        """
        ig_attributions = self.explain_instance(index, m_steps)
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)
        
        plt.figure(figsize=(12, 8))

        max_val = np.max(np.abs(ig_attributions))
        num_sensores = len(self.feature_names)
        
        ax = sns.heatmap(
            ig_attributions, 
            cmap="RdBu_r", 
            center=0,
            vmin=-max_val, 
            vmax=max_val,
            cbar_kws={'label': 'Impacto no RUL (Atribuição IG)'}
        )
        
        ax.set_ylabel("Sensores", fontsize=12)
        ax.set_xlabel("Janela de Tempo (Ciclos T-29 a T)", fontsize=12)

        ax.set_xticks(np.arange(num_sensores) + 0.5)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        
        plt.title(f"Integrated Gradients Heatmap - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.show()

    def plot_local_bar(self, index, m_steps=50):
        """
        Gera um gráfico de barras resumindo o impacto de cada sensor 
        para uma amostra específica, achatando a dimensão temporal.
        Ideal para comparar diretamente com os resultados do LIME.
        """

        ig_attributions = self.explain_instance(index, m_steps)
        sensor_impact = np.sum(ig_attributions, axis=0)
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)
        
        sensor_labels = self.feature_names

        sorted_indices = np.argsort(sensor_impact)
        sorted_impacts = sensor_impact[sorted_indices]
        sorted_labels = np.array(sensor_labels)[sorted_indices]
        
        colors = ['red' if val < 0 else 'green' for val in sorted_impacts]
        
        plt.figure(figsize=(10, 8))
        plt.barh(sorted_labels, sorted_impacts, color=colors)
        plt.xlabel("Soma do Impacto no RUL (Atribuição IG)", fontsize=12)
        plt.title(f"Impacto Local das Features (IG) - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}", fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_global_importance(self, num_samples=100, m_steps=50):
        """
        Calcula a importância global dos sensores avaliando múltiplas amostras.
        Isso prova quais sensores o modelo mais utiliza de forma geral 
        para detectar a degradação, gerando um argumento forte para a dissertação.
        """
        print(f"Calculando IG global para {num_samples} amostras. Isso pode levar alguns segundos...")
        
        samples_to_use = min(num_samples, len(self.X_test))
        global_attributions = np.zeros(self.sample_shape) 
        
        for i in range(samples_to_use):
            global_attributions += np.abs(self.explain_instance(i, m_steps))

        global_attributions /= samples_to_use

        global_sensor_impact = np.sum(global_attributions, axis=0)
        sensor_labels = self.feature_names

        sorted_indices = np.argsort(global_sensor_impact)
        sorted_impacts = global_sensor_impact[sorted_indices]
        sorted_labels = np.array(sensor_labels)[sorted_indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(sorted_labels, sorted_impacts, color='royalblue')
        plt.xlabel("Média da Magnitude Absoluta do Impacto (IG)", fontsize=12)
        plt.title(f"Importância Global dos Sensores (Integrated Gradients)\nBaseado em {samples_to_use} amostras", fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_global_summary(self, num_samples=100):
        """
        Plota a visão global dos Integrated Gradients.
        Avalia um lote de amostras para entender o comportamento geral dos gradientes acumulados.
        """
        print(f"Calculando Integrated Gradients para {num_samples} amostras (isso pode levar um tempo)...")
        
        all_ig = []
        for i in range(num_samples):
            ig_matrix = self.explain_instance(i)
            all_ig.append(ig_matrix)

        mean_ig = np.mean(np.abs(all_ig), axis=0)

        plt.figure(figsize=(10, 6))
        plt.imshow(mean_ig.T, aspect='auto', cmap='coolwarm')

        num_sensores = mean_ig.shape[1]
        
        plt.title(f"Visão Global de Sensibilidade - IG ({num_samples} Amostras)\nIntegrated Gradients Médio")
        plt.xlabel('Janela de Tempo (30 ciclos)')
        plt.ylabel('Sensores')
        plt.colorbar(label='Magnitude Média do IG')
        plt.yticks(ticks=np.arange(num_sensores), labels=self.feature_names)
        
        plt.tight_layout()
        plt.show()