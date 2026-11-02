"""Optuna e treino final RUL v2. Nenhum teste oficial é aberto neste módulo."""

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np

from src.rul_prediction.v2.experiment import (
    DATA_ROOT, MODEL_ROOT, MODELS, PROTOCOL, assert_model_shape, check_output_root, exclusive_lock,
    experiment_path, identity, load_data, read_json, write_json,
)
from src.rul_prediction.v2.utils import DATASETS, file_hash


def suggest_params(trial, model):
    # Espaços históricos; todas as combinações de cabeças são divisíveis.
    if model == 'cnn_lstm':
        params = {'filters': trial.suggest_categorical('filters', [32, 64, 128]),
                  'kernel_size': trial.suggest_int('kernel_size', 3, 5),
                  'lstm_units': trial.suggest_categorical('lstm_units', [32, 64, 128])}
    else:
        params = {'d_model': trial.suggest_categorical('d_model', [32, 64, 128]),
                  'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
                  'ff_dim': trial.suggest_categorical('ff_dim', [64, 128, 256]),
                  'num_layers': trial.suggest_int('num_layers', 1, 3)}
    params.update(dropout_rate=trial.suggest_float('dropout_rate', .1, .4, step=.1),
                  learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                  batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]))
    return params


def build_model(model, shape, params):
    from src.rul_prediction.models.model_builders import build_cnn_lstm, build_transformer
    return (build_cnn_lstm if model == 'cnn_lstm' else build_transformer)(shape, params)


def prepare_runtime(seed):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    # Reprodutibilidade quando suportada pelo dispositivo/versão.
    tf.config.experimental.enable_op_determinism()


def clear_runtime():
    import tensorflow as tf
    tf.keras.backend.clear_session()


def callbacks(patience, trial=None):
    from tensorflow import keras
    result = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                           restore_best_weights=True)]
    if trial is not None:
        from optuna_integration.tfkeras import TFKerasPruningCallback
        result.append(TFKerasPruningCallback(trial, 'val_loss'))
    return result


def fit_once(data, model_name, params, seed, epochs, patience, trial=None):
    prepare_runtime(seed)
    try:
        train, val = data['train'], data['validation']
        model = build_model(model_name, train['X'].shape[1:], params)
        assert_model_shape(model, train['X'].shape[1:])
        history = model.fit(train['X'], train['y'],
                            validation_data=(val['X'], val['y']), epochs=epochs,
                            batch_size=params['batch_size'], callbacks=callbacks(patience, trial),
                            shuffle=True, verbose=2).history
        history = {key: [float(v) for v in values] for key, values in history.items()}
        if not history.get('val_loss') or not np.isfinite(history['val_loss']).all():
            raise ValueError('Histórico de validação vazio/não finito.')
        return model, history
    finally:
        clear_runtime()


def software():
    return {name: importlib.metadata.version(name)
            for name in ('tensorflow', 'keras', 'optuna', 'optuna-integration', 'numpy')}


def study_config(provenance, model, experiment, seed, trials, epochs, patience):
    if trials < 1 or epochs < 1 or patience < 0 or not 0 <= seed < 2**32:
        raise ValueError('Trials/épocas positivos, patience não negativa e seed uint32 são necessários.')
    return {'protocol': PROTOCOL, 'data': provenance, 'model': model, 'experiment': experiment,
            'sampler_seed': seed, 'total_trials': trials, 'epochs': epochs, 'patience': patience,
            'objective': 'min_validation_mse', 'pruner': 'MedianPruner(n_warmup_steps=5)',
            'training_target': 'y_capped_125', 'software': software(),
            'code_sha256': {name: file_hash(path) for name, path in {
                'train.py': Path(__file__), 'experiment.py': Path(__file__).with_name('experiment.py'),
                'model_builders.py': Path(__file__).parents[1] / 'models/model_builders.py',
                'transformer_layers.py': Path(__file__).parents[1] / 'models/transformer_layers.py',
            }.items()}}


def ensure_config(folder, config):
    path = Path(folder) / 'study_config.json'
    if path.exists():
        if read_json(path) != config:
            raise ValueError('Estudo incompatível: dados/código/configuração mudaram. Use outro experimento.')
    else:
        if (Path(folder) / 'study.db').exists():
            raise ValueError('Banco existente sem configuração: retomada recusada.')
        write_json(path, config)


def remaining_trials(trials, total):
    # COMPLETE, PRUNED e FAIL consomem orçamento; não há workers paralelos.
    if any(t.state.name in ('RUNNING', 'WAITING') for t in trials):
        raise ValueError('Trial RUNNING/WAITING: confira interrupção antes de retomar.')
    if len(trials) > total:
        raise ValueError('Estudo já excede o orçamento solicitado.')
    return total - len(trials)


def open_study(folder, config):
    import optuna
    return optuna.create_study(
        study_name='rul_v2', storage='sqlite:///' + (Path(folder) / 'study.db').resolve().as_posix(),
        direction='minimize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config['sampler_seed']),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))


def optimize(data_root, model_root, dataset, model, experiment, seed=42,
             trials=30, epochs=50, patience=5, fail_interrupted=False):
    check_output_root(model_root, MODEL_ROOT)
    data, provenance = load_data(Path(data_root) / dataset, ('train', 'validation'))
    folder = experiment_path(model_root, dataset, model, experiment)
    folder.mkdir(parents=True, exist_ok=True)
    config = study_config(provenance, model, experiment, seed, trials, epochs, patience)
    with exclusive_lock(folder):
        ensure_config(folder, config)  # Antes de abrir/mutar o banco.
        study = open_study(folder, config)
        if fail_interrupted:
            from optuna.trial import TrialState
            for trial in study.trials:
                if trial.state.name == 'RUNNING':
                    study.tell(trial.number, state=TrialState.FAIL)
        remaining = remaining_trials(study.trials, trials)

        def objective(trial):
            params = suggest_params(trial, model)
            fitted, history = fit_once(data, model, params, (seed + trial.number) % 2**32,
                                       epochs, patience, trial)
            del fitted
            return min(history['val_loss'])

        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        if not any(t.state.name == 'COMPLETE' for t in study.trials):
            raise ValueError('Nenhum trial completo; não há campeão para treino final.')
        best = {'config_sha256': identity(config), 'params': study.best_params,
                'trial': study.best_trial.number, 'validation_mse': float(study.best_value)}
        # Snapshots imutáveis; a retomada pode produzir outro campeão.
        target = folder / f'best_trial_{study.best_trial.number}.json'
        if target.exists():
            if read_json(target) != best:
                raise ValueError('Snapshot do campeão incompatível.')
        else:
            write_json(target, best)
        print(f'{dataset}/{model}: {len(study.trials)}/{trials} trials; campeão: {target.name}')
        return target


def plot_history(history, destination):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for key in ('loss', 'val_loss'):
        ax.plot(np.arange(1, len(history[key]) + 1), history[key], label=key)
    ax.set(xlabel='Epoch', ylabel='MSE'); ax.legend()
    fig.tight_layout(); fig.savefig(destination); plt.close(fig)


def train_final(data_root, model_root, dataset, model, experiment, seeds,
                best_trial, epochs=50, patience=5):
    check_output_root(model_root, MODEL_ROOT)
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds devem ser únicas e uint32.')
    if epochs < 1 or patience < 0 or best_trial < 0:
        raise ValueError('Épocas/trial/patience inválidos.')
    data, provenance = load_data(Path(data_root) / dataset, ('train', 'validation'))
    folder = experiment_path(model_root, dataset, model, experiment)
    config = read_json(folder / 'study_config.json')
    expected = study_config(provenance, model, experiment, config['sampler_seed'],
                            config['total_trials'], config['epochs'], config['patience'])
    if expected != config:
        raise ValueError('Configuração, código ou pré-processamento mudou desde o Optuna.')
    best = read_json(folder / f'best_trial_{best_trial}.json')
    if best['config_sha256'] != identity(config) or best['trial'] != best_trial:
        raise ValueError('Hiperparâmetros pertencem a outro estudo.')
    with exclusive_lock(folder):
        for seed in seeds:
            if (folder / f'seed_{seed}').exists():
                raise FileExistsError(f'Seed {seed} já existe; nenhum modelo será sobrescrito.')
        for seed in seeds:
            target = folder / f'seed_{seed}'
            target.mkdir()  # Pasta parcial também fica protegida em caso de falha.
            final_config = {**config, 'seed': seed, 'final_epochs': epochs,
                            'final_patience': patience, 'best': best}
            write_json(target / 'config.json', final_config)
            model_obj, history = fit_once(data, model, best['params'], seed, epochs, patience)
            model_obj.save(target / 'model.keras')
            write_json(target / 'history.json', history)
            plot_history(history, target / 'history.png')
            write_json(target / 'manifest.json', {'protocol': PROTOCOL, 'data': provenance,
                       'model': model, 'experiment': experiment, 'seed': seed,
                       'best_validation_mse': min(history['val_loss']),
                       'sha256': {name: file_hash(target / name) for name in
                                  ('model.keras', 'config.json', 'history.json', 'history.png')}})
            del model_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'optimize', 'final'):
        p = sub.add_parser(command)
        p.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
        p.add_argument('--model', choices=MODELS, default='cnn_lstm')
        p.add_argument('--experiment', default='baseline')
        p.add_argument('--data-dir', type=Path, default=DATA_ROOT)
        if command != 'inspect':
            p.add_argument('--epochs', type=int, default=50)
            p.add_argument('--patience', type=int, default=5)
        if command == 'optimize':
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--trials', type=int, default=30, help='Orçamento TOTAL, incluindo FAIL/PRUNED.')
            p.add_argument('--fail-interrupted', action='store_true', help='Marca RUNNING como FAIL após interrupção confirmada.')
        if command == 'final':
            p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
            p.add_argument('--best-trial', required=True, type=int, help='Número do snapshot campeão informado pelo optimize.')
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Não repita datasets.')
    for dataset in args.datasets:
        if args.command == 'inspect':
            data, provenance = load_data(args.data_dir / dataset, ('train', 'validation'))
            print(dataset, provenance, {key: value['X'].shape for key, value in data.items()})
        elif args.command == 'optimize':
            optimize(args.data_dir, MODEL_ROOT, dataset, args.model, args.experiment,
                     args.seed, args.trials, args.epochs, args.patience, args.fail_interrupted)
        else:
            train_final(args.data_dir, MODEL_ROOT, dataset, args.model, args.experiment,
                        args.seeds, args.best_trial, args.epochs, args.patience)


if __name__ == '__main__':
    main()
