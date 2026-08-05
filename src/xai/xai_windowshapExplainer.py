import numpy as np
import shap
import matplotlib.pyplot as plt
from src.xai.xai_baseExplainer import BaseExplainer 

class WindowShapExplainer(BaseExplainer):
    def __init__(self, model_path, x_path, y_path, window_size=5, baseline_value=0.0):

        # Inicializa a classe base, que vai carregar os dados e criar self.X_test, self.y_test e self.model
        super().__init__(model_path, x_path, y_path)
        
        self.window_size = window_size
        self.baseline_value = baseline_value

        # CORREÇÃO: Usar self.X_test para acessar o atributo herdado da BaseExplainer
        self.timesteps = self.X_test.shape[1]  
        self.n_features = self.X_test.shape[2] 
        
        if self.timesteps % self.window_size != 0:
            raise ValueError(f"O tamanho da janela ({window_size}) deve ser um divisor exato do número total de ciclos ({self.timesteps}).")
            
        self.num_windows = self.timesteps // self.window_size

    def _mask_to_3d(self, mask_matrix, original_instance):
            n_samples = mask_matrix.shape[0]
            
            masked_data = np.full((n_samples, self.timesteps, self.n_features), self.baseline_value, dtype=np.float32)

            orig_squeezed = np.squeeze(original_instance)
            
            for i in range(n_samples):
                for w in range(self.num_windows):
                    if mask_matrix[i, w] == 1:
                        start_idx = w * self.window_size
                        end_idx = start_idx + self.window_size
                        
                        masked_data[i, start_idx:end_idx, :] = orig_squeezed[start_idx:end_idx, :]
                        
            return masked_data

    def get_explainer(self, index):

        original_instance = self.get_instance(index)

        def predict_wrapper(mask_matrix):
            X_3d = self._mask_to_3d(mask_matrix, original_instance)
            return self.model.predict(X_3d, verbose=0).flatten()

        background = np.zeros((1, self.num_windows))
        
        explainer = shap.KernelExplainer(predict_wrapper, background)
        return explainer

    def explain_windows(self, index, n_samples=1000):
        """Calcula os valores SHAP exatos para as janelas temporais."""

        explainer = self.get_explainer(index)

        instance_to_explain = np.ones((1, self.num_windows))
        
        shap_values = explainer.shap_values(instance_to_explain, nsamples=n_samples, silent=True)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        return shap_values.flatten()
        
    def plot_window_importance(self, index, n_samples=1000):
        """Plota a contribuição de cada bloco de tempo para o cálculo do RUL."""
        shap_values = self.explain_windows(index, n_samples)
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)

        window_labels = [f"Ciclos {w*self.window_size}-{(w+1)*self.window_size - 1}" for w in range(self.num_windows)]
        
        plt.figure(figsize=(12, 5))

        colors = ['red' if val < 0 else 'green' for val in shap_values]
        
        plt.bar(window_labels, shap_values, color=colors, alpha=0.8, edgecolor='black')
        plt.axhline(0, color='black', linewidth=1)
        
        plt.title(f"WindowSHAP - Impacto por Janela Temporal (Amostra {index})\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}", fontsize=14)
        plt.ylabel("Shapley Value (Impacto no RUL)", fontsize=12)
        plt.xlabel("Janela de Tempo", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def plot_global_window_importance(self, num_instances=20, n_samples=1000):
        """
        Calcula a importância global agregando o valor absoluto dos Shapley Values 
        de várias amostras para cada bloco de tempo (janela).
        """
        
        all_shap_values = []
        
        for i in range(num_instances):
            shap_vals = self.explain_windows(i, n_samples)
            all_shap_values.append(np.abs(shap_vals))
            
        mean_shap_values = np.mean(all_shap_values, axis=0)
        
        window_labels = [f"Ciclos {w*self.window_size}-{(w+1)*self.window_size - 1}" for w in range(self.num_windows)]
        
        plt.figure(figsize=(12, 5))
        plt.bar(window_labels, mean_shap_values, color='indigo', alpha=0.7, edgecolor='black')
        plt.plot(window_labels, mean_shap_values, color='black', marker='o', linestyle='-', alpha=0.5)
        
        plt.title(f"WindowSHAP - Importância Global das Janelas Temporais\n(Média Absoluta de {num_instances} amostras)", fontsize=14)
        plt.ylabel("Média Absoluta do Shapley Value", fontsize=12)
        plt.xlabel("Janela de Tempo", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()