import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def load_nasa_data(file_path):

    #Pegando as colunas da documentação do dataset
    colunas = [
        'unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3',
        's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10',
        's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19',
        's_20', 's_21'
    ]

    try:
        data = pd.read_csv(file_path, sep=r'\s+', header=None, names=colunas)
        return data
    except Exception as e:
        print(f"Erro ao carregar os dados de {file_path}: {e}")
        return None


def load_rul_data(file_path):

    #Lendo os dados de RUL (Remaining Useful Life) do arquivo fornecido
    try:
        rul_data = pd.read_csv(file_path, sep=r'\s+', header=None, names=['RUL'])
        return rul_data
    except Exception as e:
        print(f"Erro ao carregar os dados de RUL de {file_path}: {e}")
        return None


def add_rul_train(train_data):

    #Precisamos criar a coluna Y (RUL) para o conjunto de treinamento.
    #Para isso, precisamos descobrir o ciclo máximo de cada motor e subtrair o ciclo atual.
    max_cycles = train_data.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_cycle']

    train_data = train_data.merge(max_cycles, on='unit_nr', how='left')
    train_data['RUL'] = train_data['max_cycle'] - train_data['time_cycles']
    train_data.drop(columns=['max_cycle'], inplace=True)

    return train_data


def add_rul_test(test_data, rul_data):
    #Precisamos criar a coluna Y (RUL) para o conjunto de teste.
    #Aqui, precisamos descobrir o ciclo máximo de cada motor no conjunto de teste e somar com o RUL fornecido para obter o ciclo final.
    max_cycles_test = test_data.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles_test.columns = ['unit_nr', 'max_test_cycle']

    rul_data = rul_data.copy()
    rul_data['unit_nr'] = rul_data.index + 1
    max_cycles_test = max_cycles_test.merge(rul_data, on='unit_nr')
    max_cycles_test['max_cycle_absolute'] = max_cycles_test['max_test_cycle'] + max_cycles_test['RUL']

    test_data = test_data.merge(max_cycles_test[['unit_nr', 'max_cycle_absolute']], on='unit_nr', how='left')
    test_data['RUL'] = test_data['max_cycle_absolute'] - test_data['time_cycles']
    test_data.drop(columns=['max_cycle_absolute'], inplace=True)

    return test_data


def apply_piecewise_rul(data, rul_cap=125):
    #Limitando o RUL a um valor máximo (rul_cap) para evitar valores muito altos que podem prejudicar o treinamento do modelo.
    data = data.copy()
    data['RUL'] = data['RUL'].clip(upper=rul_cap)
    return data


def detect_multi_regime(data, decimals=2):
    settings = ['setting_1', 'setting_2', 'setting_3']
    combinacoes_unicas = data[settings].round(decimals).drop_duplicates()
    return len(combinacoes_unicas) > 1


def preprocess_nasa_data(data, sequence_length=30, is_train=True,
                          rul_cap=125, scaler=None, features=None,
                          kmeans=None, regime_scalers=None, multi_regime=None,
                          last_window_only=False):

    #Tratamento de NULLs e valores ausentes
    data = data.dropna().copy()

    #Piecewise linear RUL (cap padrão de 125 ciclos)
    if rul_cap is not None:
        data = apply_piecewise_rul(data, rul_cap=rul_cap)

    settings = ['setting_1', 'setting_2', 'setting_3']
    colunas_controle = ['unit_nr', 'time_cycles', 'RUL']

    if is_train:
        multi_regime = detect_multi_regime(data)
        if multi_regime:
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            data['regime'] = kmeans.fit_predict(data[settings])
    else:
        if multi_regime:
            data['regime'] = kmeans.predict(data[settings])

    if is_train:
        #Removendo colunas irrelevantes/redundantes (desvio padrão zero).
        colunas_protegidas = colunas_controle + (['regime'] if multi_regime else [])
        colunas_candidatas = [c for c in data.columns if c not in colunas_protegidas]
        desvio_padrao = data[colunas_candidatas].std()
        colunas_estaticas = desvio_padrao[desvio_padrao == 0].index
        data.drop(columns=colunas_estaticas, inplace=True)

        features = [col for col in data.columns if col not in colunas_protegidas]
    else:
        colunas_manter = colunas_controle + features + (['regime'] if multi_regime else [])
        data = data[colunas_manter].copy()

    data[features] = data[features].astype(float)

    #Normalizando: MinMaxScaler global para datasets de condição única
    if not multi_regime:
        if is_train:
            scaler = MinMaxScaler()
            data[features] = scaler.fit_transform(data[features])
        else:
            #No teste, usamos apenas .transform() para não vazar dados
            data[features] = scaler.transform(data[features])
    else:
        if is_train:
            regime_scalers = {}
            for regime in sorted(data['regime'].unique()):
                mask = data['regime'] == regime
                regime_scaler = MinMaxScaler()
                data.loc[mask, features] = regime_scaler.fit_transform(data.loc[mask, features])
                regime_scalers[regime] = regime_scaler
        else:
            for regime, regime_scaler in regime_scalers.items():
                mask = data['regime'] == regime
                if mask.any():
                    data.loc[mask, features] = regime_scaler.transform(data.loc[mask, features])
        data.drop(columns=['regime'], inplace=True)

    #Criando janelas deslizantes para capturar a sequência temporal
    X_list = []
    y_list = []

    for unit_id, group in data.groupby('unit_nr'):
        group_features = group[features].values
        group_target = group['RUL'].values
        n = len(group)

        if last_window_only:
            seq = group_features[-sequence_length:]
            if len(seq) < sequence_length:
                pad = np.repeat(seq[0:1], sequence_length - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            X_list.append(seq)
            y_list.append(group_target[-1])
            continue

        if n < sequence_length:
            pad = np.repeat(group_features[0:1], sequence_length - n, axis=0)
            seq = np.vstack([pad, group_features])
            X_list.append(seq)
            y_list.append(group_target[-1])
        else:
            for i in range(n - sequence_length + 1):
                X_list.append(group_features[i:i + sequence_length])
                y_list.append(group_target[i + sequence_length - 1])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, scaler, features, kmeans, regime_scalers, multi_regime


if __name__ == "__main__":

    raw_path = "data/raw_data"
    processed_path = "data/processed_data"
    os.makedirs(processed_path, exist_ok=True)

    #Carregando os dados de treinamento e teste
    name_data = ['1', '2', '3', '4']

    for name in name_data:

        train_file = f"{raw_path}/train_FD00{name}.txt"
        test_file = f"{raw_path}/test_FD00{name}.txt"
        rul_file = f"{raw_path}/RUL_FD00{name}.txt"

        train_data = load_nasa_data(train_file)
        test_data = load_nasa_data(test_file)
        rul_data = load_rul_data(rul_file)

        train_data = add_rul_train(train_data)
        test_data = add_rul_test(test_data, rul_data)

        (X_train, y_train, scaler_treino, features,
         kmeans_treino, regime_scalers_treino, multi_regime) = preprocess_nasa_data(
            train_data, is_train=True, rul_cap=125
        )

        X_test, y_test, _, _, _, _, _ = preprocess_nasa_data(
            test_data, is_train=False, rul_cap=125,
            scaler=scaler_treino, features=features,
            kmeans=kmeans_treino, regime_scalers=regime_scalers_treino,
            multi_regime=multi_regime,
            last_window_only=False,  # mude para True se quiser bater com o protocolo oficial de benchmark
        )

        np.savez_compressed(f"{processed_path}/X_train_FD00{name}.npz", dados=X_train)
        np.savez_compressed(f"{processed_path}/y_train_FD00{name}.npz", dados=y_train)
        np.savez_compressed(f"{processed_path}/X_test_FD00{name}.npz", dados=X_test)
        np.savez_compressed(f"{processed_path}/y_test_FD00{name}.npz", dados=y_test)

        print(f"✅ Processamento do FD00{name} concluído e tensores salvos! "
              f"(multi-regime: {multi_regime}, features: {len(features)})")