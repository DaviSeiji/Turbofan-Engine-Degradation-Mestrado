"""Configuração, splits herdados e transformador nominal serializável do AE v2."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from src.common.paths import ROOT
from src.rul_prediction.v2.utils import (
    Config as RULConfig, DATASETS, DEVELOPMENT_SPLITS, RULPreprocessor,
    file_hash, validate_trajectories,
)

PIPELINE = 'ae_coordinated_rul_v2'
ROLES = ('ae_fit', 'ae_inner_validation')
RAW_DIR = ROOT / 'data/raw_data'
RUL_DIR = ROOT / 'data/processed_rul_v2'
OUTPUT_DIR = ROOT / 'data/processed_anomaly_v2'


@dataclass(frozen=True)
class Config:
    dataset: str
    seed: int = 42
    nominal_fraction: float = .30
    inner_validation_fraction: float = .20
    constant_tolerance: float = 1e-12

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError('Dataset inválido.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Seed deve ser uint32.')
        if not np.isfinite(self.nominal_fraction) or not 0 < self.nominal_fraction <= 1:
            raise ValueError('Fração nominal deve pertencer a (0, 1].')
        if not np.isfinite(self.inner_validation_fraction) or not 0 < self.inner_validation_fraction < 1:
            raise ValueError('Fração de validação interna deve pertencer a (0, 1).')
        if not np.isfinite(self.constant_tolerance) or self.constant_tolerance < 0:
            raise ValueError('Tolerância de constância inválida.')

    def to_dict(self):
        return {**asdict(self), 'pipeline': PIPELINE, 'schema_version': 1, 'window_size': 30,
                'nominal_rule': 'floor(fraction * complete_development_trajectory_length)',
                'inner_split_method': 'sorted_IDs_default_rng_first_ceil_fraction_for_validation',
                'feature_selection': 'AE_fit_nominal_range_gt_tolerance',
                'multi_regime': self.dataset in ('FD002', 'FD004'),
                'padding': 'repeat_first_cycle_on_left_for_scoring_only',
                'dtype': 'float32', 'clip_normalized_values': False,
                'test_prepared': False, 'calibration_used': False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_rul_splits(ids):
    if set(ids) != set(DEVELOPMENT_SPLITS) | {'test'}:
        raise ValueError('Papéis RUL inválidos.')
    for values in ids.values():
        if (not isinstance(values, list) or not values or len(values) != len(set(values))
                or any(type(v) is not int or v < 1 for v in values)):
            raise ValueError('IDs RUL devem ser listas de inteiros positivos únicos.')
    for i, a in enumerate(DEVELOPMENT_SPLITS):
        for b in DEVELOPMENT_SPLITS[i + 1:]:
            if set(ids[a]) & set(ids[b]):
                raise ValueError('Motor compartilhado entre divisões RUL de desenvolvimento.')
    # Teste tem outro namespace: seus IDs podem repetir os números de train.


def load_rul_reference(dataset, rul_dir=RUL_DIR):
    """Lê apenas três JSONs; não abre NPZ, metadados de teste ou transformadores RUL."""
    folder = Path(rul_dir) / dataset
    manifest = read_json(folder / 'manifest.json')
    if (manifest['pipeline'] != 'rul_v2' or manifest['schema_version'] != 1
            or manifest['Dataset'] != dataset):
        raise ValueError('Manifesto RUL incompatível.')
    for name in ('config.json', 'splits.json'):
        if file_hash(folder / name) != manifest['sha256'][name]:
            raise ValueError(f'Hash RUL divergente: {name}.')
    config = read_json(folder / 'config.json')
    rul_config = RULConfig(**{k: config[k] for k in RULConfig.__dataclass_fields__})
    if config != manifest['config'] or config != rul_config.to_dict() or rul_config.dataset != dataset:
        raise ValueError('Identidade/configuração RUL incompatível.')
    ids = read_json(folder / 'splits.json')
    validate_rul_splits(ids)
    reference = {'Dataset': dataset, 'manifest_sha256': file_hash(folder / 'manifest.json'),
                 'config_sha256': file_hash(folder / 'config.json'),
                 'splits_sha256': file_hash(folder / 'splits.json'),
                 'raw_train_sha256': manifest['input_sha256'][f'train_{dataset}.txt']}
    return ids, reference


def split_inside_rul_train(ids, config):
    validate_rul_splits(ids)
    units = np.sort(ids['train'])
    count = int(np.ceil(len(units) * config.inner_validation_fraction))
    if count < 1 or count >= len(units):
        raise ValueError('Motores RUL train insuficientes para dois grupos AE não vazios.')
    shuffled = np.random.default_rng(config.seed).permutation(units)
    return {'rul_splits': deepcopy(ids), 'ae_fit': sorted(map(int, shuffled[count:])),
            'ae_inner_validation': sorted(map(int, shuffled[:count]))}


class CoordinatedNominalPreprocessor(RULPreprocessor):
    """Reutiliza código de MinMax/KMeans, nunca os objetos ajustados do RUL."""

    def __init__(self, config, splits, reference, fit_limits):
        super().__init__(config)
        if splits != split_inside_rul_train(splits['rul_splits'], config):
            raise ValueError('Divisão interna incompatível com RUL train/configuração.')
        if not fit_limits or not set(fit_limits) <= set(splits['ae_fit']):
            raise ValueError('Limites de ajuste fora de AE fit.')
        if any(type(u) is not int or type(n) is not int or n < 30 for u, n in fit_limits.items()):
            raise ValueError('Limites nominais inválidos.')
        if reference['Dataset'] != config.dataset:
            raise ValueError('Referência RUL de outro dataset.')
        self.splits = deepcopy(splits)
        self.reference = deepcopy(reference)
        self.fit_limits = dict(fit_limits)

    def fit(self, nominal):
        if (('Dataset' in nominal and not nominal.Dataset.eq(self.config.dataset).all())
                or ('SourceSplit' in nominal and not nominal.SourceSplit.eq('train').all())):
            raise ValueError('Origem física inválida para ajuste nominal.')
        nominal = validate_trajectories(nominal)
        actual = {int(u): len(g) for u, g in nominal.groupby('unit_nr')}
        if actual != self.fit_limits:
            raise ValueError('Ajuste permitido somente nas linhas nominais dos motores AE fit elegíveis.')
        super().fit(nominal)
        self.features_at_fit_ = tuple(self.features_)
        return self

    def transform(self, data):
        if not self.fitted_ or tuple(self.features_) != self.features_at_fit_:
            raise ValueError('Pré-processador não congelado ou ordem das features alterada.')
        return super().transform(data)


def output_target(output_dir, raw_dir, rul_dir, dataset):
    if dataset not in DATASETS:
        raise ValueError('Dataset inválido.')
    target = Path(output_dir).resolve() / dataset
    if target.is_relative_to(ROOT.resolve()) and not target.is_relative_to(OUTPUT_DIR.resolve()):
        raise ValueError('Dentro do projeto, use exclusivamente data/processed_anomaly_v2/.')
    protected = [Path(raw_dir), Path(rul_dir), ROOT / 'data/raw_data', ROOT / 'data/processed_data',
                 ROOT / 'data/processed_anomaly', ROOT / 'data/processed_rul_v2', ROOT / 'models',
                 ROOT / 'results', ROOT / 'src', ROOT / 'tests', ROOT / 'docs', ROOT / 'notebooks', ROOT / '.git']
    for source in protected:
        source = source.resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError('Destino invade uma pasta de origem protegida.')
    if target.exists():
        raise FileExistsError(f'Destino existente preservado, inclusive se parcial: {target}.')
    return target
