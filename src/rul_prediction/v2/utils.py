"""Configuração, partição e transformador serializável do RUL v2."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.common.paths import ROOT

DATASETS = ('FD001', 'FD002', 'FD003', 'FD004')
MULTI_REGIME = {'FD001': False, 'FD002': True, 'FD003': False, 'FD004': True}
SETTINGS = ['setting_1', 'setting_2', 'setting_3']
INPUT_COLUMNS = SETTINGS + [f's_{i}' for i in range(1, 22)]
RAW_COLUMNS = ['unit_nr', 'time_cycles'] + INPUT_COLUMNS
DEVELOPMENT_SPLITS = ('train', 'validation', 'calibration')
ARRAY_SPLITS = DEVELOPMENT_SPLITS + ('test_terminal', 'test_windows')
WINDOW_SIZE = 30


@dataclass(frozen=True)
class Config:
    dataset: str
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.15
    rul_cap: int = 125
    constant_tolerance: float = 1e-12

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f'Dataset inválido: {self.dataset}.')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('Seed deve ser um inteiro entre 0 e 2**32 - 1.')
        fractions = np.array(self.fractions, dtype=float)
        if (not np.isfinite(fractions).all() or (fractions <= 0).any()
                or not np.isclose(fractions.sum(), 1., atol=1e-9, rtol=0)):
            raise ValueError('As três frações devem ser positivas e somar 1.')
        if not isinstance(self.rul_cap, int) or self.rul_cap <= 0:
            raise ValueError('O cap de RUL deve ser inteiro positivo.')
        if not np.isfinite(self.constant_tolerance) or self.constant_tolerance < 0:
            raise ValueError('Tolerância de constância deve ser finita e não negativa.')

    @property
    def fractions(self):
        return (self.train_fraction, self.validation_fraction, self.calibration_fraction)

    def to_dict(self):
        return {**asdict(self), 'pipeline': 'rul_v2', 'schema_version': 1,
                'window_size': WINDOW_SIZE, 'dtype': 'float32',
                'multi_regime': MULTI_REGIME[self.dataset], 'n_clusters': 6 if MULTI_REGIME[self.dataset] else 0,
                'padding': 'repeat_first_cycle_on_left', 'clip_normalized_values': False,
                'partition_method': 'largest_remainder_then_default_rng_permutation',
                'calibration_role': 'reserved_only_no_fitting_or_selection'}


def split_engines(units, config):
    """Sorteia IDs, nunca janelas; arredonda frações pelo maior resto."""
    values = np.asarray(list(units), dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or (values <= 0).any() or (values % 1 != 0).any()):
        raise ValueError('IDs de motores devem ser inteiros positivos.')
    if len(np.unique(values)) != len(values):
        raise ValueError('Informe cada motor apenas uma vez.')
    counts_float = len(values) * np.asarray(config.fractions)
    counts = np.floor(counts_float).astype(int)
    remainder_order = np.argsort(-(counts_float - counts), kind='stable')
    counts[remainder_order[:len(values) - counts.sum()]] += 1
    if (counts < 1).any():
        raise ValueError('Motores insuficientes para produzir três divisões não vazias nessas frações.')
    shuffled = np.random.default_rng(config.seed).permutation(np.sort(values.astype(np.int64)))
    groups = np.split(shuffled, np.cumsum(counts)[:-1])
    return {name: sorted(map(int, group)) for name, group in zip(DEVELOPMENT_SPLITS, groups)}


def validate_trajectories(data):
    """Recusa dados incompletos; não remove linhas nem comprime lacunas."""
    missing = set(RAW_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f'Colunas ausentes: {sorted(missing)}.')
    if data.empty:
        raise ValueError('Trajetórias vazias.')
    result = data.copy()
    if not result.columns.is_unique:
        raise ValueError('Nomes de colunas duplicados.')
    result[RAW_COLUMNS] = result[RAW_COLUMNS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(result[RAW_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError('Trajetórias contêm NaN ou Inf.')
    for name in ['unit_nr', 'time_cycles']:
        if (result[name] < 1).any() or (result[name] % 1 != 0).any():
            raise ValueError(f'{name} deve conter inteiros positivos.')
        result[name] = result[name].astype(np.int64)
    if result.duplicated(['unit_nr', 'time_cycles']).any():
        raise ValueError('Ciclos duplicados no mesmo motor.')
    result = result.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)
    for unit, group in result.groupby('unit_nr', sort=True):
        if not np.array_equal(group.time_cycles.to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError(f'Motor {unit}: esperado início no ciclo 1 e ciclos consecutivos.')
    return result


class RULPreprocessor:
    """Ajuste único no treino; definido aqui para serialização estável com python -m."""

    def __init__(self, config):
        self.config = config
        self.fitted_ = False

    def fit(self, train):
        if self.fitted_:
            raise RuntimeError('Transformador já ajustado: não reajuste em outra divisão.')
        train = validate_trajectories(train)
        ranges = train[INPUT_COLUMNS].max() - train[INPUT_COLUMNS].min()
        self.features_ = [c for c in INPUT_COLUMNS if ranges[c] > self.config.constant_tolerance]
        if not self.features_:
            raise ValueError('Nenhuma feature variável nos motores de treinamento.')
        self.dropped_features_ = [c for c in INPUT_COLUMNS if c not in self.features_]
        self.train_units_ = sorted(map(int, train.unit_nr.unique()))
        self.kmeans_ = None
        if MULTI_REGIME[self.config.dataset]:
            if len(train[SETTINGS].drop_duplicates()) < 6:
                raise ValueError('Treino não contém seis condições distintas para o KMeans.')
            self.kmeans_ = KMeans(n_clusters=6, random_state=self.config.seed, n_init=10)
            self.kmeans_.fit(train[SETTINGS])
            regimes = self.kmeans_.labels_
            if len(np.unique(regimes)) != 6:
                raise ValueError('KMeans não produziu seis regimes no treino.')
        else:
            regimes = np.zeros(len(train), dtype=int)
        self.scalers_ = {}
        for regime in np.unique(regimes):
            scaler = MinMaxScaler(clip=False)
            scaler.fit(train.loc[regimes == regime, self.features_])
            self.scalers_[int(regime)] = scaler
        self.fitted_ = True
        return self

    def transform(self, data):
        if not self.fitted_:
            raise RuntimeError('Transformador não ajustado.')
        result = validate_trajectories(data)
        regimes = (self.kmeans_.predict(result[SETTINGS]) if self.kmeans_ is not None
                   else np.zeros(len(result), dtype=int))
        result[self.features_] = result[self.features_].astype(float)
        for regime in np.unique(regimes):
            if int(regime) not in self.scalers_:
                raise ValueError(f'Regime sem scaler: {regime}.')
            mask = regimes == regime
            result.loc[mask, self.features_] = self.scalers_[int(regime)].transform(
                result.loc[mask, self.features_])
        if not np.isfinite(result[self.features_].to_numpy()).all():
            raise ValueError('Transformação produziu valores não finitos.')
        result['regime'] = regimes.astype(int)
        return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def software_versions():
    names = ['numpy', 'pandas', 'scikit-learn', 'joblib']
    return {'python': platform.python_version(), 'platform': platform.platform(),
            **{name: importlib.metadata.version(name) for name in names}}


def output_target(output_dir, raw_dir, dataset):
    """Protege v1/AE e recusa um dataset já existente, mesmo incompleto."""
    if dataset not in DATASETS:
        raise ValueError('Dataset inválido.')
    target = (Path(output_dir).resolve() / dataset)
    protected = [Path(raw_dir).resolve(), ROOT / 'data/raw_data', ROOT / 'data/processed_data',
                 ROOT / 'data/processed_anomaly', ROOT / 'models', ROOT / 'results',
                 ROOT / 'src', ROOT / 'tests', ROOT / 'notebooks']
    for folder in protected:
        folder = folder.resolve()
        if target.is_relative_to(folder) or folder.is_relative_to(target):
            raise ValueError(f'Destino invade uma pasta protegida: {folder}.')
    if target.exists():
        raise FileExistsError(f'Destino já existe e será preservado: {target}. Use outra pasta de saída.')
    return target
