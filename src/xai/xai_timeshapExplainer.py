import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeshap.explainer import local_event, local_feat, local_pruning
from src.xai.xai_baseExplainer import BaseExplainer

class TimeShapExplainer(BaseExplainer):
    def __init__(self, model_path, x_path, y_path):
        super().__init__(model_path, x_path, y_path)

        self.baseline = np.zeros((1, self.X_test.shape[1], self.X_test.shape[2]), dtype=np.float32)
        
        # self.feature_names já foi populado dinamicamente pela BaseExplainer!

        self.predict_wrapper = lambda x: self.model.predict(x, verbose=0)

    def explain_features(self, index, n_samples=3200):

        instance = self.get_instance(index).astype(np.float32)

        feature_dict = {'rs': 42, 'nsamples': n_samples, 'feature_names': self.feature_names}

        feat_shap_df = local_feat(
            f=self.predict_wrapper,
            data=instance,
            feature_dict=feature_dict,
            entity_uuid=index,      
            entity_col="engine_id",   
            baseline=self.baseline,
            pruned_idx=0              
        )

        return feat_shap_df

    def explain_events(self, index, n_samples=3200):

        instance = self.get_instance(index).astype(np.float32)
        
        event_dict = {'rs': 42, 'nsamples': n_samples}
        
        event_shap_df = local_event(
            f=self.predict_wrapper,
            data=instance,
            event_dict=event_dict,   
            entity_uuid=index,            
            entity_col="engine_id",        
            baseline=self.baseline,
            pruned_idx=0                  
        )

        event_shap_df['Feature'] = event_shap_df['Feature'].astype(str).str.extract(r'(\d+)').astype(int)

        return event_shap_df


    def plot_feature_importance(self, index, n_samples=3200):
        """Plota o gráfico de barras dos Sensores mais impactantes."""

        df_feat = self.explain_features(index, n_samples)
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)
        
        features = df_feat['Feature'].values
        shap_values = df_feat['Shapley Value'].values
        
        sorted_idx = np.argsort(shap_values)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if val < 0 else 'green' for val in shap_values[sorted_idx]]
        plt.barh(features[sorted_idx], shap_values[sorted_idx], color=colors)
        
        plt.title(f"TimeSHAP (Importância do Sensor) - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}", fontsize=14)
        plt.xlabel("Shapley Value (Impacto no RUL)", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_event_importance(self, index, n_samples=3200):
        """Plota a linha do tempo indicando quais ciclos denunciaram a falha."""

        df_event = self.explain_events(index, n_samples)
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)

        # CORREÇÃO: Ordenação numérica correta dos ciclos
        df_event = df_event.sort_values(by='Feature')
        
        event_ids = df_event['Feature'].values
        shap_values = df_event['Shapley Value'].values
        
        plt.figure(figsize=(12, 6))

        colors = ['red' if val < 0 else 'green' for val in shap_values]
        plt.bar(event_ids, shap_values, color=colors, alpha=0.8)

        plt.plot(event_ids, shap_values, color='black', marker='o', linestyle='-', linewidth=1.5, alpha=0.5)
        
        plt.title(f"TimeSHAP (Linha do Tempo dos Ciclos) - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}", fontsize=14)
        plt.xlabel("Passo de Tempo (0 = Mais antigo, 29 = Mais recente)", fontsize=12)
        plt.ylabel("Shapley Value (Impacto no RUL)", fontsize=12)
        plt.xticks(event_ids)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_global_importance(self, num_instances=10, n_samples=3200):
        """
        Calcula a importância global agregando o valor absoluto dos Shapley Values 
        de várias amostras, tanto para Sensores quanto para Ciclos de Tempo.
        """
        print(f"Calculando TimeSHAP Global para {num_instances} amostras...")
        
        all_feat_shap = []
        all_event_shap = []
        
        for i in range(num_instances):
            df_feat = self.explain_features(i, n_samples)
            df_event = self.explain_events(i, n_samples)
            
            all_feat_shap.append(np.abs(df_feat['Shapley Value'].values))
            
            df_event = df_event.sort_values(by='Feature')
            all_event_shap.append(np.abs(df_event['Shapley Value'].values))
            
        mean_feat_shap = np.mean(all_feat_shap, axis=0)
        mean_event_shap = np.mean(all_event_shap, axis=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        features = df_feat['Feature'].values
        sorted_idx = np.argsort(mean_feat_shap)
        ax1.barh(features[sorted_idx], mean_feat_shap[sorted_idx], color='steelblue')
        ax1.set_title(f"Importância Global dos Sensores\n(Média de {num_instances} amostras)")
        ax1.set_xlabel("Média Absoluta do Shapley Value")
        ax1.grid(axis='x', linestyle='--', alpha=0.7)

        events = df_event['Feature'].values
        ax2.bar(events, mean_event_shap, color='darkorange', alpha=0.8)
        ax2.plot(events, mean_event_shap, color='black', marker='o', linestyle='-', alpha=0.5)
        ax2.set_title(f"Importância Global da Linha do Tempo\n(Média de {num_instances} amostras)")
        ax2.set_xlabel("Passo de Tempo (0 = Mais antigo, 29 = Mais recente)")
        ax2.set_ylabel("Média Absoluta do Shapley Value")
        ax2.set_xticks(events)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def plot_pruning(self, index, tolerance=0.05, n_samples=3200):
        """
        Plota o resultado do algoritmo de Poda (Pruning) do TimeSHAP.
        """
        print(f"Calculando Pruning Temporal para a Amostra {index}... (Tolerância: {tolerance*100}%)")
        instance = self.get_instance(index).astype(np.float32)
        pruning_dict = {'tol': tolerance, 'rs': 42, 'nsamples': n_samples}

        prune_data, coal_prun_idx = local_pruning(
            f=self.predict_wrapper, 
            data=instance, 
            pruning_dict=pruning_dict, 
            baseline=self.baseline,
            entity_uuid=index,
            entity_col="engine_id"
        )
        
        true_rul = self.get_true_rul(index)
        pred_rul = self.get_prediction(index)

        total_timesteps = self.X_test.shape[1]
        
        pruning_idx = total_timesteps + coal_prun_idx
        
        plt.figure(figsize=(12, 5))

        plt.axvspan(0, pruning_idx, color='red', alpha=0.1, label='Dados Descartáveis (Poda)')
        plt.axvspan(pruning_idx, total_timesteps - 1, color='green', alpha=0.1, label='Dados Críticos (Mantidos)')

        plt.axvline(x=pruning_idx, color='red', linestyle='--', linewidth=2.5, label=f'Ponto de Poda: Ciclo {pruning_idx}')
        
        plt.title(f"TimeSHAP Pruning (Poda Temporal) - Amostra {index}\nRUL Real: {true_rul} | RUL Previsto: {pred_rul:.2f}", fontsize=14)
        plt.xlabel("Passo de Tempo (Ciclos)", fontsize=12)
        plt.ylabel("Relevância para a Predição", fontsize=12)
        plt.xlim(0, total_timesteps - 1)
        plt.xticks(range(0, total_timesteps))
        plt.yticks([]) 
        plt.legend(loc="upper left")
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.show()
        
        print("-" * 60)
        print(f"RESULTADO DO PRUNING (Tolerância {tolerance*100}%):")
        print(f"-> Para manter a predição robusta, a CNN-LSTM precisa apenas dos ciclos {pruning_idx} a {total_timesteps - 1}.")
        print(f"-> Os ciclos iniciais de 0 a {max(0, pruning_idx - 1)} não carregam impacto suficiente e podem ser podados.")
        print("-" * 60)