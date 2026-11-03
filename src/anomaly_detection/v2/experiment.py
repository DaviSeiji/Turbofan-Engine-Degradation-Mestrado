"""Integridade e identidade do treinamento AE v2; nenhuma inferência automática."""

import importlib.metadata
import platform
import re
from pathlib import Path

import numpy as np

from src.common.paths import ROOT
from src.anomaly_detection.v2.preprocessing import validate_saved_dataset
from src.anomaly_detection.v2.utils import OUTPUT_DIR, RUL_DIR, read_json, write_json
from src.rul_prediction.v2.experiment import check_output_root, exclusive_lock, identity
from src.rul_prediction.v2.utils import DATASETS, file_hash

DATA_ROOT = OUTPUT_DIR
MODEL_ROOT = ROOT / 'models/anomaly_v2'
RESULT_ROOT = ROOT / 'results/anomaly_detection/v2'
PROTOCOL = 'ae_coordinated_rul_v2_training_1'
SEARCH_SPACE = {
    'latent_dim': [8, 16, 32, 64], 'encoder_units': [32, 64, 128],
    'num_layers': [1, 2], 'dropout_rate': [0.0, 0.1, 0.2, 0.3],
    'learning_rate': {'low': 1e-4, 'high': 1e-2, 'log': True},
    'batch_size': [32, 64, 128],
}


def experiment_path(root, dataset, experiment):
    if dataset not in DATASETS or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', experiment):
        raise ValueError('Dataset/identificador de experimento inválido.')
    target = Path(root) / dataset / experiment
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Destino do experimento escapa da raiz permitida.')
    return target


def check_roots(data_root, rul_dir, model_root, result_root):
    check_output_root(model_root, MODEL_ROOT)
    check_output_root(result_root, RESULT_ROOT)
    models, results = Path(model_root).resolve(), Path(result_root).resolve()
    if models.is_relative_to(results) or results.is_relative_to(models):
        raise ValueError('Destinos de modelos e resultados devem ser separados.')
    for output in (models, results):
        for source in (data_root, rul_dir, ROOT / 'data'):
            source = Path(source).resolve()
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError('Saída invade dados protegidos.')


def load_data(folder, rul_dir=RUL_DIR):
    """Reusa a validação completa da Etapa C; não ajusta scalers ou KMeans."""
    folder = Path(folder)
    before = file_hash(folder / 'manifest.json')
    processor, summary = validate_saved_dataset(folder, rul_dir)
    manifest = read_json(folder / 'manifest.json')
    data = {}
    for role, suffix in (('ae_fit', 'fit'), ('ae_inner_validation', 'inner_validation')):
        name = f'nominal_{suffix}.npz'
        with np.load(folder / name, allow_pickle=False) as archive:
            X = archive['X']
            features = archive['features'].tolist()
        if (features != processor.features_ or X.dtype != np.float32
                or X.shape != (summary[role]['windows'], 30, len(features))
                or not len(X) or not np.isfinite(X).all()
                or file_hash(folder / name) != manifest['sha256'][name]):
            raise ValueError('Janelas nominais incompatíveis com o manifesto.')
        data[role] = X
    if file_hash(folder / 'manifest.json') != before:
        raise ValueError('Manifesto mudou durante a leitura.')
    provenance = {'Dataset': processor.config.dataset, 'manifest_sha256': before,
                  'artifact_sha256': manifest['sha256'], 'rul_reference': manifest['rul_reference'],
                  'preprocessing_config': manifest['config'], 'splits': processor.splits,
                  'features': processor.features_, 'input_shape': [30, len(processor.features_)],
                  'summary': summary}
    return data, provenance


def assert_model_shape(model, input_shape):
    expected = (None, *input_shape)
    if tuple(model.input_shape) != expected or tuple(model.output_shape) != expected:
        raise ValueError('Autoencoder deve reconstruir exatamente (30, n_features_ae).')


def validate_params(params):
    if set(params) != set(SEARCH_SPACE):
        raise ValueError('Hiperparâmetros incompletos/desconhecidos.')
    for name, choices in SEARCH_SPACE.items():
        value = params[name]
        if name == 'learning_rate':
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not choices['low'] <= value <= choices['high']):
                raise ValueError('Learning rate fora do espaço de busca.')
        elif isinstance(value, bool) or value not in choices:
            raise ValueError(f'{name} fora do espaço de busca.')


def software():
    names = ('tensorflow', 'keras', 'optuna', 'optuna-integration', 'numpy',
             'scikit-learn', 'joblib', 'pandas', 'matplotlib', 'sqlalchemy')
    return {'python': platform.python_version(),
            **{name: importlib.metadata.version(name) for name in names}}


def study_config(provenance, experiment, seed=42, trials=30, epochs=50, patience=5):
    experiment_path(MODEL_ROOT, provenance['Dataset'], experiment)
    if (any(type(v) is not int for v in (seed, trials, epochs, patience))
            or not 0 <= seed < 2**32 or trials < 1 or epochs < 1 or patience < 0):
        raise ValueError('Seed uint32, trials/épocas positivos e patience não negativa são necessários.')
    files = [Path(__file__), Path(__file__).with_name('train.py'),
             Path(__file__).with_name('preprocessing.py'), Path(__file__).with_name('utils.py'),
             ROOT / 'src/anomaly_detection/models/autoencoder_models.py',
             ROOT / 'src/rul_prediction/v2/utils.py', ROOT / 'src/rul_prediction/v2/experiment.py',
             ROOT / 'src/common/paths.py']
    return {'protocol': PROTOCOL, 'data': provenance, 'experiment': experiment,
            'architecture': 'build_lstm_autoencoder_v1', 'search_space': SEARCH_SPACE,
            'sampler': 'TPESampler', 'sampler_seed': seed, 'total_trials': trials,
            'epochs': epochs, 'patience': patience, 'objective': 'min_nominal_inner_validation_mse',
            'target': 'X', 'optimizer': 'Adam', 'loss': 'mse', 'shuffle': True,
            'pruner': {'name': 'MedianPruner', 'n_warmup_steps': 5},
            'trial_seed': '(sampler_seed + trial.number) % 2**32',
            'software': software(),
            'code_sha256': {str(f.relative_to(ROOT)): file_hash(f) for f in files}}


def ensure_config(folder, config):
    path = Path(folder) / 'study_config.json'
    if path.exists():
        if read_json(path) != config:
            raise ValueError('Estudo incompatível; use outro identificador de experimento.')
    else:
        if any(p.name != '.lock' for p in Path(folder).iterdir()):
            raise ValueError('Experimento existente sem configuração; não será reutilizado.')
        write_json(path, config)


def trial_counts(trials):
    counts = {name: 0 for name in ('COMPLETE', 'PRUNED', 'FAIL', 'RUNNING', 'WAITING')}
    for trial in trials:
        if trial.state.name not in counts:
            raise ValueError('Estado de trial desconhecido.')
        counts[trial.state.name] += 1
    return counts


def remaining_trials(trials, total):
    counts = trial_counts(trials)
    if counts['RUNNING'] or counts['WAITING']:
        raise ValueError('RUNNING/WAITING bloqueia retomada: confira a interrupção.')
    if len(trials) > total:
        raise ValueError('Orçamento total excedido.')
    return total - len(trials)


def read_best(folder, config, number):
    if type(number) is not int or not 0 <= number < config['total_trials']:
        raise ValueError('Número de trial inválido.')
    best = read_json(Path(folder) / f'best_trial_{number}.json')
    if (best['config_sha256'] != identity(config) or best['trial'] != number
            or best['completed_budget'] != config['total_trials']
            or not np.isfinite(best['validation_mse']) or best['validation_mse'] < 0):
        raise ValueError('Snapshot incompatível ou busca incompleta.')
    validate_params(best['params'])
    return best


def load_frozen_model(data_root, model_root, result_root, dataset, experiment, seed,
                      rul_dir=RUL_DIR):
    """Verifica hashes e carrega a rede, sem predict/fit ou salvar novamente."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed final deve ser uint32.')
    check_roots(data_root, rul_dir, model_root, result_root)
    _, provenance = load_data(Path(data_root) / dataset, rul_dir)
    folder = experiment_path(model_root, dataset, experiment)
    results = experiment_path(result_root, dataset, experiment) / f'seed_{seed}'
    target = folder / f'seed_{seed}'
    config = read_json(target / 'config.json')
    study = config['study']
    expected = study_config(provenance, experiment, study['sampler_seed'], study['total_trials'],
                            study['epochs'], study['patience'])
    if (study != expected or read_json(folder / 'study_config.json') != expected
            or config['seed'] != seed or config['best'] != read_best(folder, study, config['best']['trial'])):
        raise ValueError('Modelo não corresponde ao experimento/pré-processamento atual.')
    manifest = read_json(target / 'manifest.json')
    expected_keys = {'model': {'model.keras', 'config.json'},
                     'results': {'history.json', 'history.png', 'summary.json'}}
    if (manifest['protocol'] != PROTOCOL or manifest['config_sha256'] != identity(config)
            or manifest['data'] != provenance or manifest['seed'] != seed
            or set(manifest['sha256']) != set(expected_keys)):
        raise ValueError('Manifesto final incompatível.')
    for scope, base in (('model', target), ('results', results)):
        if set(manifest['sha256'][scope]) != expected_keys[scope]:
            raise ValueError('Artefatos finais incompletos.')
        for name, digest in manifest['sha256'][scope].items():
            if file_hash(base / name) != digest:
                raise ValueError(f'Hash final divergente: {name}.')
    from tensorflow import keras
    model = keras.models.load_model(target / 'model.keras', compile=False)
    assert_model_shape(model, provenance['input_shape'])
    return model
