"""Validação por escopo e identidade dos experimentos RUL v2; sem treinamento."""

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import ROOT
from src.rul_prediction.v2.utils import Config, DEVELOPMENT_SPLITS, file_hash

DATA_ROOT = ROOT / 'data/processed_rul_v2'
MODEL_ROOT = ROOT / 'models/rul_v2'
RESULT_ROOT = ROOT / 'results/rul_prediction/v2'
MODELS = ('cnn_lstm', 'transformer')
JOIN_KEYS = ['Dataset', 'SourceSplit', 'Split', 'unit_nr', 'start_cycle', 'end_cycle']
PROTOCOL = 'rul_v2_training_1'


def check_output_root(root, expected):
    # APIs internas aceitam pastas temporárias; dentro do projeto, só o destino v2.
    resolved = Path(root).resolve()
    if resolved.is_relative_to(ROOT.resolve()) and resolved != Path(expected).resolve():
        raise ValueError('Saída dentro do projeto deve usar exclusivamente a pasta RUL v2.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    # Criação exclusiva: um artefato final nunca é sobrescrito.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def experiment_path(root, dataset, model, experiment):
    if dataset not in ('FD001', 'FD002', 'FD003', 'FD004') or model not in MODELS:
        raise ValueError('Dataset/modelo inválido.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', experiment):
        raise ValueError('Experimento deve usar letras, números, hífen ou underscore.')
    return Path(root) / dataset / model / experiment


@contextmanager
def exclusive_lock(folder):
    """Um processo por estudo; após crash, conferir e remover o lock manualmente."""
    lock = Path(folder) / '.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write('Não executar dois processos neste experimento.\n')
    try:
        yield
    finally:
        lock.unlink()


def load_data(folder, splits):
    """Confere hashes somente dos artefatos usados; nunca abre calibration.npz."""
    folder = Path(folder)
    allowed = {'train', 'validation', 'test_terminal', 'test_windows'}
    if not splits or not set(splits) <= allowed or len(set(splits)) != len(splits):
        raise ValueError('Escopo inválido; calibração não é utilizada nesta etapa.')
    manifest = read_json(folder / 'manifest.json')
    if manifest['pipeline'] != 'rul_v2' or manifest['schema_version'] != 1:
        raise ValueError('Manifesto não é RUL v2.')
    names = ['config.json', 'splits.json', 'features.json', 'preprocessor.joblib']
    names += [f'{s}{suffix}' for s in splits for suffix in ('.npz', '_metadata.csv')]
    for name in names:
        if file_hash(folder / name) != manifest['sha256'][name]:
            raise ValueError(f'Hash divergente: {name}.')
    config = read_json(folder / 'config.json')
    cfg = Config(**{key: config[key] for key in Config.__dataclass_fields__})
    if (cfg.to_dict() != config or config != manifest['config']
            or cfg.dataset != manifest['Dataset'] or cfg.dataset != folder.name or cfg.rul_cap != 125):
        raise ValueError('Configuração/target incompatível: treinamento requer cap 125.')
    ids = read_json(folder / 'splits.json')
    if set(ids) != set(DEVELOPMENT_SPLITS) | {'test'}:
        raise ValueError('Divisões inválidas.')
    for values in ids.values():
        if not values or len(values) != len(set(values)) or any(type(v) is not int or v < 1 for v in values):
            raise ValueError('IDs inválidos nas divisões.')
    for i, a in enumerate(DEVELOPMENT_SPLITS):
        for b in DEVELOPMENT_SPLITS[i + 1:]:
            if set(ids[a]) & set(ids[b]):
                raise ValueError('Motor compartilhado entre divisões de desenvolvimento.')
    features = read_json(folder / 'features.json')['selected']
    if not features or len(features) != len(set(features)):
        raise ValueError('Features inválidas.')
    data = {}
    for split in splits:
        meta = pd.read_csv(folder / f'{split}_metadata.csv')
        source = 'train' if split in DEVELOPMENT_SPLITS else 'test'
        expected = ids[split] if source == 'train' else ids['test']
        if (meta.empty or not meta.Dataset.eq(cfg.dataset).all()
                or not meta.Split.eq(split).all() or not meta.SourceSplit.eq(source).all()
                or set(meta.unit_nr) != set(expected)
                or meta.duplicated(['unit_nr', 'end_cycle']).any()):
            raise ValueError(f'{split}: identidade/metadados inválidos.')
        if split == 'test_terminal' and meta.unit_nr.duplicated().any():
            raise ValueError('Teste terminal requer exatamente uma janela por motor.')
        with np.load(folder / f'{split}.npz', allow_pickle=False) as archive:
            if set(archive.files) != {'X', 'y', 'y_original', 'features'}:
                raise ValueError('Chaves NPZ incompatíveis.')
            arrays = {key: archive[key] for key in archive.files}
        if (arrays['X'].shape != (len(meta), 30, len(features))
                or arrays['X'].dtype != np.float32 or not np.isfinite(arrays['X']).all()
                or arrays['features'].tolist() != features
                or arrays['y'].shape != (len(meta),) or arrays['y_original'].shape != (len(meta),)
                or not np.array_equal(arrays['y'], meta.RUL_capped)
                or not np.array_equal(arrays['y_original'], meta.RUL_original)
                or not np.isfinite(arrays['y_original']).all() or (arrays['y_original'] < 0).any()
                or not np.array_equal(arrays['y'], np.minimum(arrays['y_original'], 125))):
            raise ValueError(f'{split}: arrays, features ou targets incompatíveis.')
        if ((meta.end_cycle - meta.start_cycle + 1 != meta.observed_cycles).any()
                or (meta.start_cycle < 1).any() or (meta.observed_cycles < 1).any()
                or (meta.padding_cycles < 0).any()
                or (meta.observed_cycles + meta.padding_cycles != 30).any()
                or not meta.is_padded.eq(meta.padding_cycles.gt(0)).all()):
            raise ValueError(f'{split}: ciclos/padding inválidos.')
        data[split] = {**arrays, 'metadata': meta}
    provenance = {'Dataset': cfg.dataset, 'manifest_sha256': file_hash(folder / 'manifest.json'),
                  'features': features, 'input_shape': [30, len(features)],
                  'preprocessing_config': config}
    return data, provenance


def assert_model_shape(model, input_shape):
    if tuple(model.input_shape) != (None, *input_shape) or tuple(model.output_shape) != (None, 1):
        raise ValueError('Shape do modelo incompatível com features/saída RUL.')
