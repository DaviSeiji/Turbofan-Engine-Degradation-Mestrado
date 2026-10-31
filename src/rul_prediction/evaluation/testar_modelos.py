import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import gc
from time import perf_counter
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.paths import ROOT as RAIZ_PROJETO

from src.rul_prediction.experiment_utils import MODELOS, carregar_best_params, salvar_csv_atomico


# Seleciona os modelos, datasets e seeds que serão avaliados.
MODELOS_ATIVOS = ['CNN-LSTM', 'Transformer']
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [1, 2, 3, 4, 5]
REEXECUTAR_EXISTENTES = False


PASTA_DADOS = RAIZ_PROJETO / 'data' / 'processed_data'
CSV_DETALHADO = RAIZ_PROJETO / 'models' / 'benchmark_seeds_detalhado.csv'
CSV_RESUMO = RAIZ_PROJETO / 'models' / 'benchmark_seeds_resumo.csv'
CHAVES = ['Model', 'Dataset', 'Seed']
METRICAS = ['MAE', 'RMSE', 'MedianAE', 'P95AE', 'MaxAE']
COLUNAS = CHAVES + METRICAS + ['Epochs', 'TrainTimeSeconds']




# Carrega os NPZ existentes, sem refazer o pré-processamento.
def carregar_dados(dataset):
    arrays = []
    for prefixo in ['X_train', 'y_train', 'X_test', 'y_test']:
        caminho = PASTA_DADOS / f'{prefixo}_{dataset}.npz'
        with np.load(caminho, allow_pickle=False) as arquivo:
            arrays.append(arquivo['dados'])
    X_train, y_train, X_test, y_test = arrays
    if X_train.ndim != 3 or X_test.ndim != 3 or X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(f'{dataset}: shapes de entrada incompatíveis.')
    for X, y in [(X_train, y_train), (X_test, y_test)]:
        if len(X) != len(y) or y.size != len(X):
            raise ValueError(f'{dataset}: esperado um target escalar por janela.')
    return X_train, y_train, X_test, y_test


# Compara previsões e RUL real com os formatos alinhados.
def calcular_metricas(y_true, y_pred):
    real = np.asarray(y_true).reshape(-1)
    previsto = np.asarray(y_pred).reshape(-1)
    if real.size == 0 or real.shape != previsto.shape:
        raise ValueError('Targets e previsões devem ter o mesmo número de elementos.')
    if not np.isfinite(real).all() or not np.isfinite(previsto).all():
        raise ValueError('Targets ou previsões contêm NaN/Inf.')
    erros = np.abs(real - previsto)
    return {
        'MAE': float(mean_absolute_error(real, previsto)),
        'RMSE': float(np.sqrt(mean_squared_error(real, previsto))),
        'MedianAE': float(np.median(erros)),
        'P95AE': float(np.percentile(erros, 95)),
        'MaxAE': float(np.max(erros)),
    }


# Treina um modelo do zero e avalia o teste para uma seed.
def executar_seed(nome_modelo, dataset, seed, params, dados):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = history = None
    try:
        X_train, y_train, X_test, y_test = dados
        model = MODELOS[nome_modelo]['builder'](X_train.shape[1:], params)
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True,
        )
        inicio = perf_counter()
        history = model.fit(
            X_train, y_train, validation_split=0.2, epochs=50,
            batch_size=params['batch_size'], callbacks=[early_stop], verbose=0,
        )
        tempo = perf_counter() - inicio
        y_pred = model.predict(X_test, verbose=0)
        return {
            'Model': nome_modelo, 'Dataset': dataset, 'Seed': seed,
            **calcular_metricas(y_test, y_pred),
            'Epochs': len(history.epoch), 'TrainTimeSeconds': tempo,
        }
    finally:
        del history, model
        keras.backend.clear_session()
        gc.collect()


# Confere os resultados já salvos antes de retomar o benchmark.
def carregar_resultados():
    if not CSV_DETALHADO.exists():
        return pd.DataFrame(columns=COLUNAS)
    tabela = pd.read_csv(CSV_DETALHADO)
    faltantes = set(COLUNAS) - set(tabela.columns)
    if faltantes:
        raise ValueError(f'CSV detalhado incompatível; colunas ausentes: {sorted(faltantes)}')
    if tabela[CHAVES].isna().any().any() or tabela.duplicated(CHAVES).any():
        raise ValueError('CSV contém chaves ausentes/duplicadas; revise antes de retomar.')
    numeros = tabela[METRICAS + ['Seed', 'Epochs', 'TrainTimeSeconds']].to_numpy(dtype=float)
    if not np.isfinite(numeros).all() or (numeros < 0).any():
        raise ValueError('CSV contém resultados inválidos; revise antes de retomar.')
    if (tabela['Seed'] % 1 != 0).any() or (tabela['Epochs'] < 1).any():
        raise ValueError('CSV contém Seed ou Epochs inválidos.')
    tabela['Seed'] = tabela['Seed'].astype(int)
    return tabela


def resumir_resultados(tabela):
    # Pandas usa desvio padrão amostral (ddof=1); uma seed resulta em std NaN.
    resumo = tabela.groupby(['Model', 'Dataset'])[
        METRICAS + ['TrainTimeSeconds']
    ].agg(['mean', 'std'])
    resumo.columns = ['_'.join(coluna) for coluna in resumo.columns]
    return resumo.reset_index()




def main():
    for valores, nome in [(MODELOS_ATIVOS, 'MODELOS_ATIVOS'), (DATASETS, 'DATASETS'), (SEEDS, 'SEEDS')]:
        if not valores or len(set(valores)) != len(valores):
            raise ValueError(f'{nome} deve ser uma lista não vazia, sem repetições.')
    if set(MODELOS_ATIVOS) - set(MODELOS):
        raise ValueError('Modelo ativo sem entrada no registro MODELOS.')
    if set(DATASETS) - {'FD001', 'FD002', 'FD003', 'FD004'}:
        raise ValueError('Datasets válidos: FD001 a FD004.')
    if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in SEEDS):
        raise ValueError('Seeds devem ser inteiros entre 0 e 2**32 - 1.')

    tabela = carregar_resultados()
    concluidos = set(tabela[CHAVES].itertuples(index=False, name=None))
    # Pula combinações concluídas, exceto quando a reexecução está habilitada.
    pendentes = [
        (modelo, dataset, seed)
        for dataset in DATASETS for modelo in MODELOS_ATIVOS for seed in SEEDS
        if REEXECUTAR_EXISTENTES or (modelo, dataset, seed) not in concluidos
    ]
    print(f'Execuções pendentes: {len(pendentes)}; concluídas serão preservadas/puladas.')
    # Verifica todos os estudos necessários antes de iniciar o primeiro treino.
    parametros = {}
    for modelo, dataset, seed in pendentes:
        if (modelo, dataset) not in parametros:
            parametros[modelo, dataset] = carregar_best_params(modelo, dataset)

    for dataset in DATASETS:
        selecionados = [item for item in pendentes if item[1] == dataset]
        if not selecionados:
            continue
        dados = carregar_dados(dataset)
        try:
            for modelo, _, seed in selecionados:
                print(f'\n{"=" * 40}\n{modelo} | {dataset} | seed={seed}\n{"=" * 40}', flush=True)
                try:
                    resultado = executar_seed(modelo, dataset, seed, parametros[modelo, dataset], dados)
                    # Só substitui uma execução anterior após sucesso da nova.
                    mesma = (tabela['Model'] == modelo) & (tabela['Dataset'] == dataset) & (tabela['Seed'] == seed)
                    tabela = pd.concat([tabela.loc[~mesma], pd.DataFrame([resultado])], ignore_index=True)
                    salvar_csv_atomico(tabela, CSV_DETALHADO)
                    salvar_csv_atomico(resumir_resultados(tabela), CSV_RESUMO)
                    for metrica in METRICAS:
                        print(f'{metrica}: {resultado[metrica]:.4f}')
                except Exception as exc:
                    raise RuntimeError(f'Falha: {modelo} | {dataset} | seed={seed}. Resultados já salvos foram preservados.') from exc
        finally:
            del dados
            gc.collect()

    if not tabela.empty:
        resumo = resumir_resultados(tabela)
        salvar_csv_atomico(resumo, CSV_RESUMO)
        exibicao = resumo[['Model', 'Dataset']].copy()
        for metrica in ['MAE', 'RMSE']:
            exibicao[f'{metrica} mean ± std'] = resumo.apply(
                lambda row: f'{row[f"{metrica}_mean"]:.4f} ± {row[f"{metrica}_std"]:.4f}', axis=1,
            )
        print('\nResumo disponível (std indefinido com apenas uma seed):')
        print(exibicao.to_string(index=False))


if __name__ == '__main__':
    main()
