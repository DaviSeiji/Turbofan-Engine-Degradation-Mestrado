"""Registro de modelos e utilitários usados por RUL e XAI, sem executar experimentos."""
import os
import sqlite3
import tempfile
from pathlib import Path

import optuna

from src.common.paths import ROOT as RAIZ_PROJETO
from src.rul_prediction.models.model_builders import build_cnn_lstm, build_transformer


# Liga cada modelo ao seu builder e ao estudo Optuna correspondente.
MODELOS = {
    'CNN-LSTM': {
        'builder': build_cnn_lstm,
        'optuna_dir': RAIZ_PROJETO / 'models' / 'optuna_logs',
        'study_prefix': 'cnn_lstm',
    },
    'Transformer': {
        'builder': build_transformer,
        'optuna_dir': RAIZ_PROJETO / 'models' / 'optuna_logs_transformer',
        'study_prefix': 'transformer',
    },
}


# Lê os melhores parâmetros já encontrados, sem executar nova busca.
def carregar_best_params(nome_modelo, dataset):
    config = MODELOS[nome_modelo]
    banco = Path(config['optuna_dir']) / f'optuna_study_{dataset}.db'
    estudo = f"{config['study_prefix']}_{dataset}"
    if not banco.is_file():
        raise FileNotFoundError(f'{nome_modelo} | {dataset}: banco ausente: {banco}')

    # Optuna pode inicializar metadados no RDBStorage. Fazemos isso só na cópia.
    with tempfile.TemporaryDirectory(prefix='benchmark_optuna_') as pasta:
        copia = Path(pasta) / 'study.db'
        origem = sqlite3.connect(banco.resolve().as_uri() + '?mode=ro', uri=True)
        destino = sqlite3.connect(copia)
        try:
            origem.backup(destino)
        finally:
            destino.close()
            origem.close()
        storage = optuna.storages.RDBStorage(f'sqlite:///{copia.as_posix()}')
        try:
            try:
                study = optuna.load_study(study_name=estudo, storage=storage)
            except KeyError as exc:
                raise ValueError(f'Estudo {estudo!r} não encontrado em {banco}.') from exc
            try:
                params = dict(study.best_params)
            except ValueError as exc:
                raise ValueError(f'Estudo {estudo!r} sem trial concluído utilizável.') from exc
            if 'batch_size' not in params:
                raise ValueError(f'Estudo {estudo!r}: best_params não contém batch_size.')
            return params
        finally:
            storage.remove_session()
            storage.engine.dispose()



# Só substitui o CSV depois que a nova versão foi escrita por completo.
def salvar_csv_atomico(tabela, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='', suffix='.tmp',
            dir=caminho.parent, delete=False,
        ) as arquivo:
            temporario = Path(arquivo.name)
            tabela.to_csv(arquivo, index=False)
        os.replace(temporario, caminho)
    finally:
        if temporario is not None and temporario.exists():
            temporario.unlink()
