"""Pré-processamento independente do AE; RUL é somente metadado."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.common.preprocessing.preproc_data_nasa import (
    load_nasa_data, load_rul_data, add_rul_train, add_rul_test,
)

from src.common.paths import ROOT
RAW_DIR = ROOT / 'data' / 'raw_data'
PROCESSED_DIR = ROOT / 'data' / 'processed_anomaly'
DATASETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEED = 42
NOMINAL_FRACTION = 0.30
SEQUENCE_LENGTH = 30
SETTINGS = ['setting_1', 'setting_2', 'setting_3']
INPUT_COLUMNS = SETTINGS + [f's_{i}' for i in range(1, 22)]
META_COLUMNS = ['unit_nr', 'start_cycle', 'end_cycle', 'RUL']


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_trajectory(dataset, split, raw_dir=RAW_DIR):
    if dataset not in DATASETS or split not in ('train', 'test'):
        raise ValueError('Dataset ou split inválido.')
    path = Path(raw_dir) / f'{split}_{dataset}.txt'
    data = load_nasa_data(path)
    if data is None or data.empty:
        raise ValueError(f'Trajetória ausente ou vazia: {path}')
    if not np.isfinite(data[['unit_nr', 'time_cycles'] + INPUT_COLUMNS]).all().all():
        raise ValueError(f'Dados não finitos: {path}')
    data = data.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)
    for unit, group in data.groupby('unit_nr'):
        cycles = group['time_cycles'].to_numpy()
        if not np.array_equal(cycles, np.arange(1, len(group) + 1)):
            raise ValueError(f'{path}: ciclos não consecutivos para unit_nr={unit}.')
    # Não aplica cap de RUL; essa coluna nunca participa do fit.
    if split == 'train':
        return add_rul_train(data)
    rul = load_rul_data(Path(raw_dir) / f'RUL_{dataset}.txt')
    if rul is None or len(rul) != data['unit_nr'].nunique():
        raise ValueError(f'{dataset}: arquivo de metadados RUL inválido.')
    return add_rul_test(data, rul)


def nominal_rows(data, fraction=NOMINAL_FRACTION, sequence_length=SEQUENCE_LENGTH):
    if not 0 < fraction <= 1 or sequence_length < 1:
        raise ValueError('Fração nominal ou janela inválida.')
    selected, report = [], []
    for unit, group in data.groupby('unit_nr', sort=True):
        group = group.sort_values('time_cycles')
        count = int(np.floor(len(group) * fraction))
        eligible = count >= sequence_length
        report.append({'unit_nr': int(unit), 'total_cycles': len(group),
                       'nominal_cycles': count, 'eligible': eligible})
        if eligible:
            selected.append(group.iloc[:count])
    nominal = pd.concat(selected, ignore_index=True) if selected else data.iloc[:0].copy()
    return nominal, pd.DataFrame(report)


class NominalPreprocessor:
    def __init__(self, multi_regime=False, seed=SEED):
        self.multi_regime = multi_regime
        self.seed = seed

    def fit(self, nominal_train):
        if nominal_train.empty:
            raise ValueError('Nenhum motor de treino possui região nominal suficiente.')
        # Remove apenas features constantes no treino nominal.
        std = nominal_train[INPUT_COLUMNS].std()
        self.features = [c for c in INPUT_COLUMNS if std[c] > 0]
        if not self.features:
            raise ValueError('Nenhuma feature variável no treino nominal.')
        self.kmeans = None
        if self.multi_regime:
            self.kmeans = KMeans(n_clusters=6, random_state=self.seed, n_init=10)
            regimes = self.kmeans.fit_predict(nominal_train[SETTINGS])
            if len(np.unique(regimes)) != 6:
                raise ValueError('Treino nominal não representa os seis regimes.')
        else:
            regimes = np.zeros(len(nominal_train), dtype=int)
        self.scalers = {}
        for regime in np.unique(regimes):
            self.scalers[int(regime)] = MinMaxScaler().fit(
                nominal_train.loc[regimes == regime, self.features]
            )
        return self

    def transform(self, data):
        result = data.copy()
        result[self.features] = result[self.features].astype(float)
        regimes = (self.kmeans.predict(data[SETTINGS]) if self.multi_regime
                   else np.zeros(len(data), dtype=int))
        for regime in np.unique(regimes):
            if int(regime) not in self.scalers:
                raise ValueError(f'Regime sem scaler nominal: {regime}')
            mask = regimes == regime
            # Sem clipping: valores fora do intervalo nominal são preservados.
            result.loc[mask, self.features] = self.scalers[int(regime)].transform(
                data.loc[mask, self.features]
            )
        return result


def make_windows(data, features, sequence_length=SEQUENCE_LENGTH):
    windows, metadata = [], []
    for unit, group in data.groupby('unit_nr', sort=True):
        group = group.sort_values('time_cycles')
        values = group[features].to_numpy(dtype=np.float32)
        cycles = group['time_cycles'].to_numpy()
        for start in range(len(group) - sequence_length + 1):
            end = start + sequence_length - 1
            if cycles[end] - cycles[start] != sequence_length - 1:
                raise ValueError(f'Janela com lacunas em unit_nr={unit}.')
            windows.append(values[start:end + 1])
            metadata.append([int(unit), int(cycles[start]), int(cycles[end]), group['RUL'].iloc[end]])
    X = (np.asarray(windows, dtype=np.float32) if windows else
         np.empty((0, sequence_length, len(features)), dtype=np.float32))
    return X, pd.DataFrame(metadata, columns=META_COLUMNS)


def prepare_nominal(dataset, nominal_fraction=NOMINAL_FRACTION,
                    sequence_length=SEQUENCE_LENGTH, seed=SEED, raw_dir=RAW_DIR):
    data = read_trajectory(dataset, 'train', raw_dir)
    units_train, units_val = train_test_split(
        np.sort(data['unit_nr'].unique()), test_size=0.2, random_state=seed,
    )
    train, report_train = nominal_rows(data[data.unit_nr.isin(units_train)], nominal_fraction, sequence_length)
    val, report_val = nominal_rows(data[data.unit_nr.isin(units_val)], nominal_fraction, sequence_length)
    for name, report in [('train', report_train), ('validation', report_val)]:
        ignored = report.loc[~report.eligible, 'unit_nr'].tolist()
        print(f'{dataset} | {name}: motores ignorados por região nominal curta: {ignored}')
    if val.empty:
        raise ValueError('Nenhuma janela nominal de validação disponível.')
    processor = NominalPreprocessor(dataset in ('FD002', 'FD004'), seed).fit(train)
    X_train, meta_train = make_windows(processor.transform(train), processor.features, sequence_length)
    X_val, meta_val = make_windows(processor.transform(val), processor.features, sequence_length)
    if not len(X_train) or not len(X_val):
        raise ValueError('Treino e validação precisam conter janelas nominais.')
    assert set(meta_train.unit_nr).isdisjoint(meta_val.unit_nr)
    report = pd.concat([report_train.assign(split='train'), report_val.assign(split='validation')])
    return processor, X_train, X_val, meta_train, meta_val, report


def load_or_prepare(dataset, nominal_fraction=NOMINAL_FRACTION,
                    sequence_length=SEQUENCE_LENGTH, seed=SEED,
                    raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    if dataset not in DATASETS:
        raise ValueError(f'Dataset inválido: {dataset}')
    config = {'version': 1, 'Dataset': dataset, 'nominal_fraction': nominal_fraction,
              'sequence_length': sequence_length, 'seed': seed, 'validation_fraction': 0.2,
              'raw_train_sha256': file_hash(Path(raw_dir) / f'train_{dataset}.txt')}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    folder = Path(processed_dir) / dataset
    required = ['manifest.json', 'preprocessor.joblib', 'nominal_windows.npz',
                'train_metadata.csv', 'validation_metadata.csv', 'nominal_units.csv']
    if folder.exists() and any(folder.iterdir()):
        if not all((folder / f).is_file() for f in required):
            raise ValueError(f'Preparação incompleta em {folder}; revise antes de continuar.')
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['fingerprint'] != fingerprint:
            raise ValueError(f'Configuração/dados mudaram. Use outra pasta de processamento: {folder}')
        for name, digest in manifest['sha256'].items():
            if file_hash(folder / name) != digest:
                raise ValueError(f'Artefato de pré-processamento alterado: {folder / name}')
        processor = joblib.load(folder / 'preprocessor.joblib')
        with np.load(folder / 'nominal_windows.npz') as arrays:
            X_train, X_val = arrays['X_train'], arrays['X_val']
        return processor, X_train, X_val, manifest

    processor, X_train, X_val, mt, mv, report = prepare_nominal(
        dataset, nominal_fraction, sequence_length, seed, raw_dir,
    )
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(processor, folder / 'preprocessor.joblib')
    np.savez_compressed(folder / 'nominal_windows.npz', X_train=X_train, X_val=X_val)
    mt.to_csv(folder / 'train_metadata.csv', index=False)
    mv.to_csv(folder / 'validation_metadata.csv', index=False)
    report.to_csv(folder / 'nominal_units.csv', index=False)
    manifest = {**config, 'fingerprint': fingerprint, 'features': processor.features,
                'sha256': {name: file_hash(folder / name) for name in required[1:]}}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return processor, X_train, X_val, manifest
