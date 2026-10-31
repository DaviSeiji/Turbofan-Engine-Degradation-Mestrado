"""Reconstrução e alertas persistentes; não estima o início real da falha."""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.anomaly_detection.preprocessing import (
    ROOT, PROCESSED_DIR, DATASETS, read_trajectory, make_windows, file_hash,
)

MODEL_DIR = ROOT / 'models' / 'best_models_autoencoder'
RESULT_DIR = ROOT / 'results' / 'anomaly_detection' / 'detection'
THRESHOLD_PERCENTILE = 99
CONSECUTIVE_WINDOWS = 3
UNITS_TO_PLOT = [1, 2, 3, 4, 5]


def reconstruction_errors(X, X_hat):
    X, X_hat = np.asarray(X), np.asarray(X_hat)
    if X.ndim != 3 or X.shape != X_hat.shape or not len(X):
        raise ValueError('Reconstruções devem ter o mesmo shape 3D das entradas, sem lote vazio.')
    if not np.isfinite(X).all() or not np.isfinite(X_hat).all():
        raise ValueError('Entrada ou reconstrução contém NaN/Inf.')
    squared = (X.astype(np.float64) - X_hat.astype(np.float64)) ** 2
    return squared.mean(axis=(1, 2)), squared.mean(axis=1)


def score_windows(model, X, batch_size=128):
    # Reconstrói em lotes para não guardar todas as saídas simultaneamente.
    scores, sensors = [], []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        reconstructed = model.predict(batch, verbose=0)
        global_error, sensor_error = reconstruction_errors(batch, reconstructed)
        scores.append(global_error)
        sensors.append(sensor_error)
    if not scores:
        return np.empty(0), np.empty((0, X.shape[2]))
    return np.concatenate(scores), np.concatenate(sensors)


def calibrate_threshold(validation_nominal_errors, percentile=THRESHOLD_PERCENTILE):
    values = np.asarray(validation_nominal_errors, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Erros nominais de validação inválidos.')
    if not 0 < percentile < 100:
        raise ValueError('Percentil deve estar entre 0 e 100, exclusive.')
    return float(np.percentile(values, percentile))


def persistent_alerts(table, threshold, consecutive_windows=CONSECUTIVE_WINDOWS):
    if consecutive_windows < 1 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Threshold ou número de janelas inválido.')
    result = table.sort_values(['unit_nr', 'end_cycle']).reset_index(drop=True).copy()
    result['above_threshold'] = result['reconstruction_error'] > threshold
    result['persistent_anomaly'] = False
    result['first_persistent_anomaly_cycle'] = np.nan
    summaries = []
    for unit, group in result.groupby('unit_nr', sort=True):
        streak, previous, detection_cycle, detection_error = 0, None, np.nan, np.nan
        for index, row in group.iterrows():
            cycle = int(row['end_cycle'])
            if previous is not None and cycle != previous + 1:
                streak = 0
            streak = streak + 1 if row['above_threshold'] else 0
            if streak >= consecutive_windows:
                result.loc[index, 'persistent_anomaly'] = True
                if np.isnan(detection_cycle):
                    # O alerta só é conhecido na terceira janela, não na primeira.
                    detection_cycle, detection_error = cycle, row['reconstruction_error']
            previous = cycle
        result.loc[group.index, 'first_persistent_anomaly_cycle'] = detection_cycle
        summaries.append({'unit_nr': int(unit), 'detection_cycle': detection_cycle,
                          'reconstruction_error': detection_error, 'threshold': threshold,
                          'windows_evaluated': len(group)})
    columns = ['unit_nr', 'detection_cycle', 'reconstruction_error', 'threshold', 'windows_evaluated']
    return result, pd.DataFrame(summaries, columns=columns)


def plot_trajectories(table, threshold, dataset, split, units, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for unit in units:
        data = table[table.unit_nr == unit].sort_values('end_cycle')
        if data.empty:
            print(f'{dataset}/{split}: unit_nr={unit} sem janelas para gráfico.')
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        try:
            ax.plot(data.end_cycle, data.reconstruction_error, label='Reconstruction MSE')
            ax.axhline(threshold, color='red', linestyle='--', label='Nominal validation threshold')
            first = data.first_persistent_anomaly_cycle.iloc[0]
            if pd.notna(first):
                ax.axvline(first, color='orange', linestyle=':', label='First persistent anomaly')
            ax.set(title=f'{dataset} - {split} - unit {unit}', xlabel='End cycle', ylabel='Reconstruction error')
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / f'{split}_unit_{unit:03d}.png', dpi=200)
        finally:
            plt.close(fig)


def detect_dataset(dataset, splits=('test',), units=UNITS_TO_PLOT,
                   consecutive_windows=CONSECUTIVE_WINDOWS):
    folder = RESULT_DIR / dataset
    output_names = ['reconstruction_errors.csv', 'detection_summary.csv', 'sensor_reconstruction_errors.csv']
    # Evita sobrescrever análises anteriores sem uma decisão explícita do usuário.
    if any((folder / name).exists() for name in output_names) or (folder / 'plots').exists():
        raise FileExistsError(f'Já existem resultados em {folder}; arquive-os antes de executar novamente.')
    calibration = json.loads((MODEL_DIR / f'threshold_{dataset}.json').read_text(encoding='utf-8'))
    processed = PROCESSED_DIR / dataset
    manifest = json.loads((processed / 'manifest.json').read_text(encoding='utf-8'))
    if calibration['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Threshold e pré-processamento pertencem a experimentos diferentes.')
    if file_hash(processed / 'preprocessor.joblib') != manifest['sha256']['preprocessor.joblib']:
        raise ValueError('Pré-processamento salvo foi alterado.')
    model_path = MODEL_DIR / f'best_autoencoder_{dataset}.keras'
    if file_hash(model_path) != calibration['model_sha256']:
        raise ValueError('O modelo foi alterado após a calibração do threshold.')
    processor = joblib.load(processed / 'preprocessor.joblib')
    model = None
    all_windows, all_summary, all_sensors = [], [], []
    try:
        model = keras.models.load_model(model_path)
        for split in splits:
            raw = read_trajectory(dataset, split)
            transformed = processor.transform(raw)
            X, metadata = make_windows(transformed, processor.features, manifest['sequence_length'])
            errors, sensors = score_windows(model, X)
            metadata['reconstruction_error'] = errors
            windows, summary = persistent_alerts(metadata, calibration['threshold'], consecutive_windows)
            # Motores curtos ficam no resumo, mas não recebem janelas artificiais.
            summary = pd.DataFrame({'unit_nr': sorted(raw.unit_nr.unique())}).merge(summary, how='left', on='unit_nr')
            summary['threshold'] = calibration['threshold']
            summary['windows_evaluated'] = summary['windows_evaluated'].fillna(0).astype(int)
            sensor_table = pd.concat([metadata.drop(columns='reconstruction_error'),
                                      pd.DataFrame(sensors, columns=processor.features)], axis=1)
            for frame in (windows, summary, sensor_table):
                frame.insert(0, 'Split', split)
                frame.insert(0, 'Dataset', dataset)
            all_windows.append(windows)
            all_summary.append(summary)
            all_sensors.append(sensor_table)
            plot_trajectories(windows, calibration['threshold'], dataset, split, units, folder / 'plots')
            del X, raw, transformed
        folder.mkdir(parents=True, exist_ok=True)
        for frames, name in zip((all_windows, all_summary, all_sensors), output_names):
            pd.concat(frames, ignore_index=True).to_csv(folder / name, index=False)
        (folder / 'detection_config.json').write_text(json.dumps({
            'Dataset': dataset, 'splits': list(splits), 'consecutive_windows': consecutive_windows,
            'units_to_plot': list(units), 'calibration': calibration,
        }, indent=2), encoding='utf-8')
        print(f'{dataset}: resultados salvos em {folder}')
    finally:
        del model
        keras.backend.clear_session()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['test'])
    parser.add_argument('--units', nargs='*', type=int, default=UNITS_TO_PLOT)
    parser.add_argument('--consecutive-windows', type=int, default=CONSECUTIVE_WINDOWS)
    args = parser.parse_args()
    if args.consecutive_windows < 1 or len(set(args.splits)) != len(args.splits):
        parser.error('Número de janelas inválido ou splits repetidos.')
    print('GPUs detectadas:', tf.config.list_physical_devices('GPU'))
    for dataset in args.datasets:
        detect_dataset(dataset, args.splits, args.units, args.consecutive_windows)


if __name__ == '__main__':
    main()
