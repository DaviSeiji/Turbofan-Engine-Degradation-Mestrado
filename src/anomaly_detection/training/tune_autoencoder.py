"""Optuna do AE usando somente janelas nominais de treino e validação."""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import gc
import json

import numpy as np
import pandas as pd
import optuna
from optuna_integration.tfkeras import TFKerasPruningCallback
import tensorflow as tf
from tensorflow import keras

from src.anomaly_detection.preprocessing import (
    ROOT, DATASETS, SEED, NOMINAL_FRACTION, SEQUENCE_LENGTH, load_or_prepare, file_hash,
)
from src.anomaly_detection.models.autoencoder_models import build_lstm_autoencoder
from src.anomaly_detection.evaluation.detect_anomalies import score_windows, calibrate_threshold, THRESHOLD_PERCENTILE

N_TRIALS = 30
MAX_EPOCHS = 50
MODEL_DIR = ROOT / 'models' / 'best_models_autoencoder'
STUDY_DIR = ROOT / 'models' / 'optuna_logs_autoencoder'
SUMMARY_CSV = MODEL_DIR / 'resultados_autoencoder_optuna.csv'


def create_objective(X_train, X_val):
    def objective(trial):
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + trial.number)
        model = history = None
        try:
            params = {
                'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32, 64]),
                'encoder_units': trial.suggest_categorical('encoder_units', [32, 64, 128]),
                'num_layers': trial.suggest_categorical('num_layers', [1, 2]),
                'dropout_rate': trial.suggest_categorical('dropout_rate', [0.0, 0.1, 0.2, 0.3]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            }
            model = build_lstm_autoencoder(X_train.shape[1:], params)
            history = model.fit(
                X_train, X_train, validation_data=(X_val, X_val), epochs=MAX_EPOCHS,
                batch_size=params['batch_size'], verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True,
                ), TFKerasPruningCallback(trial, 'val_loss')],
            )
            return float(min(history.history['val_loss']))
        finally:
            del history, model
            keras.backend.clear_session()
            gc.collect()
    return objective


def tune_dataset(dataset, n_trials=N_TRIALS, nominal_fraction=NOMINAL_FRACTION,
                 threshold_percentile=THRESHOLD_PERCENTILE):
    model_path = MODEL_DIR / f'best_autoencoder_{dataset}.keras'
    threshold_path = MODEL_DIR / f'threshold_{dataset}.json'
    if model_path.exists() or threshold_path.exists():
        raise FileExistsError(f'{dataset}: modelo/threshold existente preservado. Arquive antes de refazer.')
    _, X_train, X_val, manifest = load_or_prepare(dataset, nominal_fraction, SEQUENCE_LENGTH, SEED)
    print(f'{dataset}: treino nominal {X_train.shape}; validação nominal {X_val.shape}')
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f'lstm_autoencoder_{dataset}', direction='minimize',
        storage=f"sqlite:///{(STUDY_DIR / f'optuna_study_{dataset}.db').as_posix()}",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    signature = {'fingerprint': manifest['fingerprint'], 'architecture_version': 1,
                 'epochs': MAX_EPOCHS, 'patience': 5, 'seed': SEED,
                 'processed_sha256': manifest['sha256']['nominal_windows.npz']}
    stored = study.user_attrs.get('protocol')
    if stored is None:
        if study.trials:
            raise ValueError('Estudo existente sem identificação de protocolo; não será misturado.')
        study.set_user_attr('protocol', signature)
    elif stored != signature:
        raise ValueError('Protocolo mudou; use outro estudo/pasta antes de continuar.')
    # Na retomada, completa o orçamento total em vez de adicionar mais 30 trials.
    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(create_objective(X_train, X_val), n_trials=remaining)
    params = study.best_params
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = history = None
    try:
        model = build_lstm_autoencoder(X_train.shape[1:], params)
        history = model.fit(
            X_train, X_train, validation_data=(X_val, X_val), epochs=MAX_EPOCHS,
            batch_size=params['batch_size'], verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True,
            )],
        )
        errors, _ = score_windows(model, X_val, params['batch_size'])
        threshold = calibrate_threshold(errors, threshold_percentile)
        model.save(model_path)
        calibration = {
            'Dataset': dataset, 'threshold': threshold, 'percentile': threshold_percentile,
            'source': 'validation_nominal', 'validation_windows': len(X_val),
            'fingerprint': manifest['fingerprint'], 'model_sha256': file_hash(model_path),
            'FinalValidationReconstructionMSE': float(errors.mean()),
        }
        threshold_path.write_text(json.dumps(calibration, indent=2), encoding='utf-8')
        row = {'Dataset': dataset, 'BestValidationReconstructionMSE': study.best_value,
               **params, 'epochs': len(history.epoch),
               'FinalValidationReconstructionMSE': float(errors.mean())}
        result = pd.DataFrame([row])
        if SUMMARY_CSV.exists():
            previous = pd.read_csv(SUMMARY_CSV)
            result = pd.concat([previous[previous.Dataset != dataset], result], ignore_index=True)
        result.to_csv(SUMMARY_CSV, index=False)
        print(f'{dataset}: modelo salvo; threshold nominal = {threshold:.6g}')
    finally:
        del history, model
        keras.backend.clear_session()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--nominal-fraction', type=float, default=NOMINAL_FRACTION)
    parser.add_argument('--threshold-percentile', type=float, default=THRESHOLD_PERCENTILE)
    args = parser.parse_args()
    if args.trials < 1 or not 0 < args.nominal_fraction <= 1 or not 0 < args.threshold_percentile < 100:
        parser.error('Trials, fração nominal ou percentil inválidos.')
    print('GPUs detectadas:', tf.config.list_physical_devices('GPU'))
    for dataset in args.datasets:
        tune_dataset(dataset, args.trials, args.nominal_fraction, args.threshold_percentile)


if __name__ == '__main__':
    main()
