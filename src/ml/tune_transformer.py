"""Transformer para RUL com o protocolo de otimização de tune_hybrid.py."""

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna_integration.tfkeras import TFKerasPruningCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras


# Habilite os demais datasets depois de validar FD001.
motores = ['FD001', 'FD002', 'FD003', 'FD004']
SEED = 42
N_TRIALS = 30
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DADOS = RAIZ_PROJETO / 'data' / 'processed_data'
PASTA_MODELOS = RAIZ_PROJETO / 'models' / 'best_models_transformer'
PASTA_ESTUDOS = RAIZ_PROJETO / 'models' / 'optuna_logs_transformer'
CAMINHO_CSV = RAIZ_PROJETO / 'models' / 'resultados_finais_transformer_optuna.csv'


@keras.utils.register_keras_serializable(package='RUL')
class PositionalEmbedding(keras.layers.Layer):
    """Embedding temporal treinável, somado à projeção das features."""

    def __init__(self, window_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_size)
        self.d_model = int(d_model)

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name='position_embeddings',
            shape=(self.window_size, self.d_model),
            initializer='glorot_uniform',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position_embeddings

    def get_config(self):
        return {
            **super().get_config(),
            'window_size': self.window_size,
            'd_model': self.d_model,
        }


def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout):
    """Mantém d_model nas duas conexões residuais."""
    if d_model % num_heads != 0:
        raise ValueError('d_model deve ser divisível por num_heads.')
    attention = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        output_shape=d_model, dropout=dropout,
    )(inputs, inputs)
    attention = keras.layers.Dropout(dropout)(attention)
    x = keras.layers.Add()([inputs, attention])
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ff = keras.layers.Dense(ff_dim, activation='relu')(x)
    ff = keras.layers.Dropout(dropout)(ff)
    ff = keras.layers.Dense(d_model)(ff)
    ff = keras.layers.Dropout(dropout)(ff)
    x = keras.layers.Add()([x, ff])
    return keras.layers.LayerNormalization(epsilon=1e-6)(x)


def build_transformer(input_shape, params):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Dense(params['d_model'])(inputs)
    x = PositionalEmbedding(input_shape[0], params['d_model'])(x)
    for _ in range(params['num_layers']):
        x = transformer_encoder(
            x, params['d_model'], params['num_heads'],
            params['ff_dim'], params['dropout_rate'],
        )
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(params['dropout_rate'])(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name='transformer_rul')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse', metrics=['mae'],
    )
    return model


def create_objective(X_train, y_train):
    def objective(trial):
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + trial.number)
        model = None
        try:
            params = {
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
                'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
                'ff_dim': trial.suggest_categorical('ff_dim', [64, 128, 256]),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            }
            # Todas as combinações do espaço atual são divisíveis.
            if params['d_model'] % params['num_heads'] != 0:
                raise optuna.TrialPruned('Número de heads incompatível com d_model.')
            model = build_transformer(X_train.shape[1:], params)
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True,
            )
            history = model.fit(
                X_train, y_train, validation_split=0.2,
                epochs=40, batch_size=params['batch_size'],
                callbacks=[early_stop, TFKerasPruningCallback(trial, 'val_loss')],
                verbose=0,
            )
            return min(history.history['val_loss'])
        finally:
            # Executado também quando o Optuna interrompe um trial.
            del model
            keras.backend.clear_session()
            gc.collect()

    return objective


def carregar_array(nome):
    # Somente leitura; não muda features, janelas ou normalização.
    with np.load(PASTA_DADOS / nome) as arquivo:
        return arquivo['dados']


def main():
    if not motores or any(m not in {'FD001', 'FD002', 'FD003', 'FD004'} for m in motores):
        raise ValueError('Selecione ao menos um dataset entre FD001 e FD004.')
    if len(set(motores)) != len(motores):
        raise ValueError('A lista de datasets não deve conter repetições.')
    # Não sobrescreve campeões existentes, nem mesmo os Transformers.
    for motor in motores:
        destino = PASTA_MODELOS / f'best_transformer_{motor}.keras'
        if destino.exists():
            raise FileExistsError(f'Modelo já existente, preservado: {destino}')
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    PASTA_ESTUDOS.mkdir(parents=True, exist_ok=True)
    resultados_finais = []

    for motor in motores:
        print(f'\nINICIANDO OTIMIZAÇÃO TRANSFORMER: {motor}')
        X_train = carregar_array(f'X_train_{motor}.npz')
        y_train = carregar_array(f'y_train_{motor}.npz')

        db_path = PASTA_ESTUDOS / f'optuna_study_{motor}.db'
        study = optuna.create_study(
            direction='minimize', study_name=f'transformer_{motor}',
            storage=f'sqlite:///{db_path.as_posix()}', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        # Como na referência, cada execução solicita 30 trials adicionais.
        study.optimize(create_objective(X_train, y_train), n_trials=N_TRIALS)
        best_hps = study.best_params
        print(f'Melhor validation MSE: {study.best_value:.4f}\n{best_hps}')

        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        best_model = None
        try:
            best_model = build_transformer(X_train.shape[1:], best_hps)
            early_stop_final = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True,
            )
            best_model.fit(
                X_train, y_train, validation_split=0.2,
                epochs=50, batch_size=best_hps['batch_size'],
                callbacks=[early_stop_final], verbose=0,
            )

            # O teste só é usado após a seleção e o treinamento final.
            X_test = carregar_array(f'X_test_{motor}.npz')
            y_test = carregar_array(f'y_test_{motor}.npz')
            y_pred = best_model.predict(X_test, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            best_model.save(PASTA_MODELOS / f'best_transformer_{motor}.keras')
            resultados_finais.append({'Motor': motor, 'RMSE': rmse, 'MAE': mae, **best_hps})

            # Preserva os resultados de outros datasets de execuções anteriores.
            tabela = pd.DataFrame(resultados_finais)
            if CAMINHO_CSV.exists():
                anteriores = pd.read_csv(CAMINHO_CSV)
                anteriores = anteriores[~anteriores['Motor'].isin(tabela['Motor'])]
                tabela = pd.concat([anteriores, tabela], ignore_index=True)
            tabela.to_csv(CAMINHO_CSV, index=False)
            print(f'{motor}: RMSE = {rmse:.4f} | MAE = {mae:.4f}')
        finally:
            del best_model
            keras.backend.clear_session()
            gc.collect()

    print('\nRESUMO DOS CAMPEÕES - TRANSFORMER (OPTUNA)')
    print(pd.DataFrame(resultados_finais).to_string(index=False))


if __name__ == '__main__':
    main()
