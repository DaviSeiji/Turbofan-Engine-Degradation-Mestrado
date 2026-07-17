import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_decoded_shap(model, X_test_2d, raw_file_path, sequence_length=30, sample_size=300):
    """
    Lê o arquivo bruto para descobrir os sensores ativos, decodifica as features 
    achatadas e gera os gráficos globais do SHAP.
    """
    # 1. Descobrindo as features dinamicamente
    colunas_originais = [
        'unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3',
        's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10',
        's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19',
        's_20', 's_21'
    ]
    
    df_raw = pd.read_csv(raw_file_path, sep=r'\s+', header=None, names=colunas_originais)
    colunas_controle = ['unit_nr', 'time_cycles']
    colunas_candidatas = [c for c in df_raw.columns if c not in colunas_controle]
    
    desvio_padrao = df_raw[colunas_candidatas].std()
    colunas_estaticas = desvio_padrao[desvio_padrao == 0].index
    
    features_list = [col for col in colunas_candidatas if col not in colunas_estaticas]
    
    print(f"Features detectadas matematicamente ({len(features_list)}): {features_list}")
    print(f"Gerando nomes para {sequence_length} ciclos e {len(features_list)} sensores...")
    
    # 2. Gerando os nomes decodificados exatos
    nomes_decodificados = []
    for t in range(sequence_length):
        for feat in features_list:
            # t=sequence_length-1 é o momento da previsão (t-0)
            tempo_relativo = (sequence_length - 1) - t
            nomes_decodificados.append(f"{feat}_(t-{tempo_relativo})")

    # 3. Amostragem para evitar travamento da memória
    np.random.seed(42)
    tamanho_real = X_test_2d.shape[0]
    idx = np.random.choice(tamanho_real, min(sample_size, tamanho_real), replace=False)
    X_test_sample = X_test_2d[idx]

    # 4. Calculando os valores SHAP
    print("Calculando contribuições marginais (SHAP)... ⏳")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # 5. Plot: Importância Global (Barras)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=nomes_decodificados, plot_type="bar", show=False)
    plt.title("Top 20 Features Mais Importantes (Decodificadas)")
    plt.tight_layout()
    plt.show()

    # 6. Plot: Impacto Direcional (Pontos)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=nomes_decodificados, show=False)
    plt.title("Impacto Direcional por Sensor e Tempo Relativo")
    plt.tight_layout()
    plt.show()