import numpy as np
from lime import lime_tabular
import matplotlib.pyplot as plt
from src.xai.xai_baseExplainer import BaseExplainer
from IPython.display import HTML, display
import webbrowser
import os

class LimeExplainer(BaseExplainer):
    def __init__(self, model_path, x_path, y_path, background_size=500,
                 discretize_continuous=True, random_state=None):
        super().__init__(model_path, x_path, y_path)
        
        self.background_data = self.X_test[:background_size].astype(np.float32)
        
        # self.feature_names já foi populado dinamicamente pela BaseExplainer!
        num_features = len(self.feature_names)
        
        self.explainer = lime_tabular.RecurrentTabularExplainer(
            self.background_data,
            training_labels=self.y_test[:background_size],
            feature_names=self.feature_names,
            discretize_continuous=discretize_continuous,
            class_names=['RUL'],
            mode='regression',
            random_state=random_state,
        )
        
    def explain_instance(self, index, num_features=None, num_samples=5000):
        if num_features is None:
            num_features = len(self.feature_names)

        instance = self.get_instance(index)[0].astype(np.float32) 
        
        def predict_wrapper(data_3d_batch):
            preds = self.model.predict(data_3d_batch.astype(np.float32), verbose=0)
            return preds.flatten() 

        exp = self.explainer.explain_instance(
            instance, 
            predict_wrapper, 
            num_features=num_features,
            num_samples=num_samples,
        )
        
        return exp

    # ==========================================
    # FUNÇÕES DE VISUALIZAÇÃO ESPECÍFICAS (LIME)
    # ==========================================
    
    def plot_lime_explanation(self, index):
        """
        Plota o gráfico de barras padrão do LIME mostrando quais sensores
        aumentam (verde) ou diminuem (vermelho) o RUL previsto.
        """
        exp = self.explain_instance(index)
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)

        print(f"RUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}")
        
        fig = exp.as_pyplot_figure()
        plt.title(f"Impacto das Features (LIME) - Amostra {index}", pad=20)
        plt.tight_layout()
        plt.show()

    def save_and_open_html(self, index, filename="lime_explicacao.html"):
        """
        Salva a explicação do LIME em um arquivo HTML e abre automaticamente
        no navegador
        """
        
        exp = self.explain_instance(index)
        html_content = exp.as_html()
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)
        print(f"Amostra {index} | RUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"Arquivo HTML salvo em: {filename}")

        caminho_absoluto = 'file://' + os.path.realpath(filename)
        
        # Abre o arquivo no navegador da web padrão
        webbrowser.open(caminho_absoluto)
        print("Abrindo a explicação no navegador...")
