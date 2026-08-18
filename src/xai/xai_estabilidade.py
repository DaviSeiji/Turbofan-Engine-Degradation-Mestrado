import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
import gc

from keras.models import Sequential
from keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

caminho_atual = os.path.dirname(os.path.abspath(__file__))
raiz_projeto = os.path.abspath(os.path.join(caminho_atual, '..', '..'))

if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from src.xai.xai_baseExplainer import BaseExplainer
from src.xai.xai_shapExplainer import ShapExplainer
from src.xai.xai_igExplainer import IntegratedGradientsExplainer
from src.xai.xai_saliencymapsExplainer import SaliencyMapsExplainer


def gerar_tabela_consenso(shap_explainer, saliency_explainer, ig_explainer, feature_names, num_samples=100):
    print(f"Extraindo dados de {num_samples} amostras para a Tabela de Consenso (aguarde)...")
    
    n_sensores = shap_explainer.X_test.shape[2]
    
    shap_importances = np.zeros(n_sensores)
    saliency_importances = np.zeros(n_sensores)
    ig_importances = np.zeros(n_sensores)
           
    for i in range(num_samples):

        sv = shap_explainer.explain_instance(i)
        if sv.ndim == 4:
            sv = sv[0, :, :, 0]
        shap_importances += np.sum(np.abs(sv), axis=0)

        sal = saliency_explainer.explain_instance(i)
        saliency_importances += np.sum(sal, axis=0) 

        ig = ig_explainer.explain_instance(i)
        ig_importances += np.sum(np.abs(ig), axis=0)

    def get_top10_scores(importances):
        top_indices = np.argsort(importances)[-10:][::-1]
        scores = {}
        ranking = {}
        for rank, sensor_idx in enumerate(top_indices):
            nome_sensor = feature_names[sensor_idx] 
            scores[nome_sensor] = 10 - rank
            ranking[nome_sensor] = rank + 1
        return scores, ranking

    shap_scores, shap_rank = get_top10_scores(shap_importances)
    sal_scores, sal_rank = get_top10_scores(saliency_importances)
    ig_scores, ig_rank = get_top10_scores(ig_importances)

    consenso = {}

    for sensor in feature_names:
        score_total = shap_scores.get(sensor, 0) + sal_scores.get(sensor, 0) + ig_scores.get(sensor, 0)
        
        if score_total > 0:
            consenso[sensor] = {
                "Pontuação Final": score_total,
                "Ranking (SHAP)": shap_rank.get(sensor, "-"),
                "Ranking (Saliency)": sal_rank.get(sensor, "-"),
                "Ranking (IG)": ig_rank.get(sensor, "-")
            }

    df_consenso = pd.DataFrame.from_dict(consenso, orient='index')
    if not df_consenso.empty:
        df_consenso = df_consenso.sort_values(by="Pontuação Final", ascending=False).head(5)
        
    return df_consenso


def treinar_modelo_temporario(X_train, y_train, best_params, temp_model_path):
    """
    Treina um modelo temporário refletindo EXATAMENTE o treinamento final 
    do campeão no Optuna.
    """
    from keras.models import Sequential
    from keras.layers import Input, Conv1D, LSTM, Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping

    filters = best_params['filters']
    kernel_size = best_params['kernel_size']
    lstm_units = best_params['lstm_units']
    dropout_rate = best_params['dropout_rate']
    lr = best_params['learning_rate']
    batch_size = best_params['batch_size']

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(LSTM(units=lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=lr), 
        loss='mse', 
        metrics=['mae']
    )

    # EarlyStopping idêntico ao modelo campeão do Optuna
    early_stop_final = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50, 
        batch_size=batch_size,
        callbacks=[early_stop_final],
        verbose=0
    )

    model.save(temp_model_path)
    return model


def avaliar_estabilidade_consenso_total(X_train, y_train, x_path, y_test_path, feature_names, best_params, num_runs=30, num_samples_xai=50):
    """
    Treina a arquitetura 'num_runs' vezes. A cada treino, extrai o Consenso (SHAP + Saliency + IG)
    e anota o Top 5 sensores daquela rodada.
    """
    historico_top5_consenso = []

    # Criação do diretório temporário
    temp_dir = os.path.join(raiz_projeto, "models", "temp_stability")
    os.makedirs(temp_dir, exist_ok=True)
    temp_model_path = os.path.join(temp_dir, "temp_model_run.keras")
    
    print(f"Iniciando Teste de Estabilidade do Consenso com {num_runs} iterações...")
    
    for i in range(num_runs):
        print(f"\n--- Rodada {i+1}/{num_runs} ---")
        
        # 1. Construir e Treinar o modelo com os MELHORES HIPERPARÂMETROS
        treinar_modelo_temporario(X_train, y_train, best_params, temp_model_path)
        
        # 2. Instanciar os Explainers
        shap_exp = ShapExplainer(temp_model_path, x_path, y_test_path, background_size=50)
        sal_exp = SaliencyMapsExplainer(temp_model_path, x_path, y_test_path)
        ig_exp = IntegratedGradientsExplainer(temp_model_path, x_path, y_test_path)

        # 3. Gerar a Tabela de Consenso
        df_consenso_rodada = gerar_tabela_consenso(shap_exp, sal_exp, ig_exp, feature_names, num_samples=num_samples_xai)

        # 4. Extrair e anotar o Top 5
        if not df_consenso_rodada.empty:
            sensores_top5_rodada = df_consenso_rodada.index.tolist()
            historico_top5_consenso.extend(sensores_top5_rodada)
            print(f"Top 5 desta rodada: {sensores_top5_rodada}")
        
        # 5. Liberar memória
        del shap_exp, sal_exp, ig_exp
        tf.keras.backend.clear_session()

    return Counter(historico_top5_consenso)


def plotar_estabilidade(contagem_sensores, num_runs, dataset_name):
    """
    Plota o gráfico de barras evidenciando a estabilidade da escolha de features.
    """
    sensores_unicos = list(contagem_sensores.keys())
    frequencias = list(contagem_sensores.values())

    indices_ordenados = np.argsort(frequencias)[::-1]
    sensores_plot = [sensores_unicos[idx] for idx in indices_ordenados]
    frequencias_plot = [frequencias[idx] for idx in indices_ordenados]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sensores_plot, frequencias_plot, color='teal', edgecolor='black', alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(y=num_runs, color='red', linestyle='--', linewidth=1.5, label=f'Consenso Absoluto ({num_runs} Treinos)')
    plt.title(f"Estabilidade das Explicações de Consenso ({dataset_name})", fontsize=14)
    plt.ylabel("Frequência de aparição no Top 5", fontsize=12)
    plt.xlabel("Sensores", fontsize=12)
    plt.ylim(0, num_runs + (num_runs * 0.15))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"estabilidade_{dataset_name}.png")
    #plt.show()
    plt.close()


if __name__ == "__main__":

    hiper_001 = {
        'filters': 32,
        'kernel_size': 3,
        'lstm_units': 128,
        'batch_size': 64,
        'dropout_rate': 0.2, 
        'learning_rate': 0.001 
    }

    hiper_002 = {
        'filters': 128,
        'kernel_size': 3,
        'lstm_units': 64,
        'batch_size': 32,
        'dropout_rate': 0.2, 
        'learning_rate': 0.001 
    }

    hiper_003 = {
        'filters': 64,
        'kernel_size': 3,
        'lstm_units': 128,
        'batch_size': 32,
        'dropout_rate': 0.2, 
        'learning_rate': 0.001 
    }

    hiper_004 = {
        'filters': 32,
        'kernel_size': 3,
        'lstm_units': 32,
        'batch_size': 32,
        'dropout_rate': 0.2, 
        'learning_rate': 0.001 
    }

    mapa_hiperparametros = {
        "FD001": hiper_001,
        "FD002": hiper_002,
        "FD003": hiper_003,
        "FD004": hiper_004
    }

    motores = ["FD001", "FD002", "FD003", "FD004"] 

    for DATASET_TARGET in motores:
        print(f"\n{'='*50}")
        print(f"INICIANDO TESTE DE ESTABILIDADE PARA O MOTOR: {DATASET_TARGET}")
        print(f"{'='*50}\n")

        x_train_path = os.path.join(raiz_projeto, f"data/processed_data/X_train_{DATASET_TARGET}.npz")
        y_train_path = os.path.join(raiz_projeto, f"data/processed_data/y_train_{DATASET_TARGET}.npz")
        x_test_path  = os.path.join(raiz_projeto, f"data/processed_data/X_test_{DATASET_TARGET}.npz")
        y_test_path  = os.path.join(raiz_projeto, f"data/processed_data/y_test_{DATASET_TARGET}.npz")

        print(f"Carregando dados para {DATASET_TARGET}...")
        X_train_data = np.load(x_train_path)['dados']
        y_train_data = np.load(y_train_path)['dados']
        
        dados_x_test = np.load(x_test_path)
        X_test_data = dados_x_test['dados']

        if 'features' in dados_x_test:
            feature_names_data = [str(f) for f in dados_x_test['features'].tolist()]
        else:
            feature_names_data = [f"s_{i}" for i in range(1, X_test_data.shape[2] + 1)] 

        contagem_estabilidade = avaliar_estabilidade_consenso_total(
            X_train=X_train_data, 
            y_train=y_train_data, 
            x_path=x_test_path, 
            y_test_path=y_test_path, 
            feature_names=feature_names_data, 
            best_params=mapa_hiperparametros[DATASET_TARGET], 
            num_runs=30, 
            num_samples_xai=100
        )

        plotar_estabilidade(contagem_estabilidade, num_runs=30, dataset_name=DATASET_TARGET)
        gc.collect() 