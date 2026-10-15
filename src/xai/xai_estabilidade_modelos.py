import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

# Registro único: novos modelos são adicionados em testar_modelos.MODELOS.
from src.ml.testar_modelos import MODELOS, carregar_best_params, salvar_csv_atomico
from src.xai.xai_estabilidade import gerar_tabela_consenso
from src.xai.xai_shapExplainer import ShapExplainer
from src.xai.xai_igExplainer import IntegratedGradientsExplainer
from src.xai.xai_saliencymapsExplainer import SaliencyMapsExplainer


# Seleciona os modelos, datasets e seeds do experimento.
MODELOS_ATIVOS = ['CNN-LSTM', 'Transformer']
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
NUM_RUNS = 30
NUM_SAMPLES_XAI = 100
SEEDS = list(range(1, 31))
REEXECUTAR_EXISTENTES = False

PASTA_DADOS = RAIZ_PROJETO / 'data' / 'processed_data'
PASTA_RESULTADOS = RAIZ_PROJETO / 'results' / 'xai_estabilidade_modelos'
CHAVES = ['Model', 'Dataset', 'Seed']
COLUNAS = CHAVES + ['Rank', 'Feature', 'ConsensusScore']


def identificador(modelo):
    # Prefixo único do registro, sem caminhos arbitrários.
    slug = MODELOS[modelo]['study_prefix']
    if not slug or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in slug):
        raise ValueError(f'Prefixo de modelo inválido: {slug!r}')
    return slug


# Lê os resultados salvos para permitir a retomada.
def carregar_detalhado(caminho, modelo):
    if not caminho.exists():
        return pd.DataFrame(columns=COLUNAS)
    tabela = pd.read_csv(caminho, dtype={'Feature': str})
    if set(COLUNAS) - set(tabela.columns):
        raise ValueError(f'CSV incompatível: {caminho}')
    if tabela[COLUNAS].isna().any().any() or not tabela['Model'].eq(modelo).all():
        raise ValueError(f'CSV com dados ausentes ou modelo incompatível: {caminho}')
    for coluna in ['Seed', 'Rank', 'ConsensusScore']:
        valores = pd.to_numeric(tabela[coluna], errors='raise')
        if not np.isfinite(valores).all() or (valores % 1 != 0).any():
            raise ValueError(f'{coluna} inválido em {caminho}')
        tabela[coluna] = valores.astype(int)
    return tabela


# Considera concluída a seed que possui cinco features válidas e distintas.
def grupos_completos(tabela):
    completos = set()
    for chave, grupo in tabela.groupby(CHAVES):
        if (len(grupo) == 5 and set(grupo['Rank']) == {1, 2, 3, 4, 5}
                and grupo['Feature'].nunique() == 5
                and grupo['ConsensusScore'].between(1, 30).all()):
            completos.add(chave)
    return completos


def carregar_treino_e_features(dataset):
    with np.load(PASTA_DADOS / f'X_train_{dataset}.npz') as arquivo:
        X_train = arquivo['dados']
    with np.load(PASTA_DADOS / f'y_train_{dataset}.npz') as arquivo:
        y_train = arquivo['dados']
    with np.load(PASTA_DADOS / f'X_test_{dataset}.npz') as arquivo:
        X_test = arquivo['dados']
        features = ([str(f) for f in arquivo['features'].tolist()]
                    if 'features' in arquivo else
                    [f's_{i}' for i in range(1, X_test.shape[2] + 1)])
    if X_train.ndim != 3 or X_test.ndim != 3 or X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(f'{dataset}: formatos de entrada incompatíveis.')
    if len(X_train) != len(y_train) or y_train.size != len(X_train):
        raise ValueError(f'{dataset}: targets incompatíveis.')
    if len(X_test) < max(NUM_SAMPLES_XAI, 50):
        raise ValueError(f'{dataset}: amostras insuficientes para XAI/background.')
    if len(features) != X_train.shape[2] or len(set(features)) != len(features) or len(features) < 5:
        raise ValueError(f'{dataset}: nomes de features inválidos.')
    return X_train, y_train, features


# Treina do zero e calcula o consenso para uma seed.
def executar_seed(modelo, dataset, seed, params, X_train, y_train, features):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    model = history = shap_exp = sal_exp = ig_exp = None
    pasta_temp = RAIZ_PROJETO / 'models' / f'temp_stability_{identificador(modelo)}'
    pasta_temp.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='run_', dir=pasta_temp) as pasta:
            caminho_modelo = Path(pasta) / 'temporary_model.keras'
            model = MODELOS[modelo]['builder'](X_train.shape[1:], params)
            history = model.fit(
                X_train, y_train, validation_split=0.2, epochs=50,
                batch_size=params['batch_size'], verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=7, restore_best_weights=True,
                )],
            )
            model.save(caminho_modelo)
            # Libera o modelo de treino antes das três instâncias dos explainers.
            del history, model
            history = model = None
            keras.backend.clear_session()
            gc.collect()
            x_path = str(PASTA_DADOS / f'X_test_{dataset}.npz')
            y_path = str(PASTA_DADOS / f'y_test_{dataset}.npz')
            shap_exp = ShapExplainer(str(caminho_modelo), x_path, y_path, background_size=50)
            sal_exp = SaliencyMapsExplainer(str(caminho_modelo), x_path, y_path)
            ig_exp = IntegratedGradientsExplainer(str(caminho_modelo), x_path, y_path)
            consenso = gerar_tabela_consenso(
                shap_exp, sal_exp, ig_exp, features, num_samples=NUM_SAMPLES_XAI,
            )
            if len(consenso) != 5 or not consenso.index.is_unique:
                raise ValueError('O consenso não retornou cinco features distintas.')
            linhas = [
                {'Model': modelo, 'Dataset': dataset, 'Seed': seed,
                 'Rank': rank, 'Feature': feature, 'ConsensusScore': row['Pontuação Final']}
                for rank, (feature, row) in enumerate(consenso.iterrows(), start=1)
            ]
            return pd.DataFrame(linhas, columns=COLUNAS)
    finally:
        del model, history, shap_exp, sal_exp, ig_exp
        keras.backend.clear_session()
        gc.collect()


def atualizar_resumo_e_graficos(tabela, modelo, features_por_dataset):
    completos = grupos_completos(tabela)
    linhas = []
    slug = identificador(modelo)
    pasta_graficos = PASTA_RESULTADOS / slug
    pasta_graficos.mkdir(parents=True, exist_ok=True)
    # Inclui datasets anteriores; somente grupos completos entram no denominador.
    for dataset in sorted(tabela['Dataset'].unique()):
        sementes = [seed for m, d, seed in completos if m == modelo and d == dataset]
        if not sementes:
            continue
        grupo = tabela[(tabela['Dataset'] == dataset) & tabela['Seed'].isin(sementes)]
        features = features_por_dataset.get(dataset)
        if features is None:
            with np.load(PASTA_DADOS / f'X_test_{dataset}.npz') as arquivo:
                features = ([str(f) for f in arquivo['features'].tolist()] if 'features' in arquivo
                            else [f's_{i}' for i in range(1, arquivo['dados'].shape[2] + 1)])
        if not set(grupo['Feature']).issubset(features):
            raise ValueError(f'{dataset}: CSV contém features ausentes nos dados atuais.')
        contagem = grupo['Feature'].value_counts().reindex(features, fill_value=0)
        contagem = contagem.sort_values(ascending=False, kind='stable')
        # A frequência usa o número de seeds concluídas neste dataset.
        n = len(sementes)
        for feature, count in contagem.items():
            linhas.append({'Model': modelo, 'Dataset': dataset, 'Feature': feature,
                           'Count': int(count), 'Frequency': count / n, 'CompletedRuns': n})
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            ax.bar(contagem.index, contagem.values, color='teal', edgecolor='black', alpha=0.8)
            ax.axhline(NUM_RUNS, color='red', linestyle='--',
                       label=f'Full consensus ({NUM_RUNS} training runs)')
            titulo = f'Consensus Explanation Stability - {modelo} ({dataset})'
            if n != NUM_RUNS:
                titulo += f'\nCompleted training runs: {n} / {NUM_RUNS}'
            ax.set(title=titulo, xlabel='Features', ylabel='Appearances in the Consensus Top-5',
                   ylim=(0, max(NUM_RUNS, n) * 1.15))
            ax.tick_params(axis='x', labelrotation=60)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(pasta_graficos / f'stability_{dataset.lower()}.png', dpi=300, bbox_inches='tight')
        finally:
            plt.close(fig)
    resumo = pd.DataFrame(linhas, columns=['Model', 'Dataset', 'Feature', 'Count', 'Frequency', 'CompletedRuns'])
    salvar_csv_atomico(resumo, PASTA_RESULTADOS / f'{slug}_stability_resumo.csv')


def main():
    if not MODELOS_ATIVOS or len(set(MODELOS_ATIVOS)) != len(MODELOS_ATIVOS):
        raise ValueError('Selecione modelos sem repetições.')
    if set(MODELOS_ATIVOS) - set(MODELOS):
        raise ValueError('Modelo não encontrado no registro compartilhado.')
    if not DATASETS or len(set(DATASETS)) != len(DATASETS) or set(DATASETS) - {'FD001', 'FD002', 'FD003', 'FD004'}:
        raise ValueError('Selecione datasets válidos, sem repetições.')
    if not SEEDS or len(set(SEEDS)) != len(SEEDS) or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in SEEDS):
        raise ValueError('Seeds devem ser inteiros distintos entre 0 e 2**32 - 1.')
    if NUM_RUNS < len(SEEDS) or NUM_RUNS < 1 or NUM_SAMPLES_XAI < 1:
        raise ValueError('NUM_RUNS deve cobrir as seeds selecionadas e NUM_SAMPLES_XAI deve ser positivo.')

    for modelo in MODELOS_ATIVOS:
        slug = identificador(modelo)
        caminho = PASTA_RESULTADOS / f'{slug}_stability_detalhado.csv'
        tabela = carregar_detalhado(caminho, modelo)
        # Identifica as execuções completas antes de decidir o que falta rodar.
        completos = grupos_completos(tabela)
        pendentes = [(d, s) for d in DATASETS for s in SEEDS
                     if REEXECUTAR_EXISTENTES or (modelo, d, s) not in completos]
        params = {d: carregar_best_params(modelo, d) for d in DATASETS if any(p[0] == d for p in pendentes)}
        features_por_dataset = {}
        print(f'{modelo}: {len(pendentes)} execuções pendentes.', flush=True)
        for dataset in DATASETS:
            seeds_dataset = [s for d, s in pendentes if d == dataset]
            if not seeds_dataset:
                continue
            X_train, y_train, features = carregar_treino_e_features(dataset)
            features_por_dataset[dataset] = features
            try:
                for seed in seeds_dataset:
                    print(f'\n{modelo} | {dataset} | seed={seed}', flush=True)
                    try:
                        novo = executar_seed(modelo, dataset, seed, params[dataset], X_train, y_train, features)
                        mesma = (tabela['Model'] == modelo) & (tabela['Dataset'] == dataset) & (tabela['Seed'] == seed)
                        tabela = pd.concat([tabela.loc[~mesma], novo], ignore_index=True)
                        salvar_csv_atomico(tabela, caminho)
                        atualizar_resumo_e_graficos(tabela, modelo, features_por_dataset)
                    except Exception as exc:
                        raise RuntimeError(f'Falha em {modelo} | {dataset} | seed={seed}; resultados salvos preservados.') from exc
            finally:
                del X_train, y_train
                gc.collect()
        if not tabela.empty:
            atualizar_resumo_e_graficos(tabela, modelo, features_por_dataset)


if __name__ == '__main__':
    main()
