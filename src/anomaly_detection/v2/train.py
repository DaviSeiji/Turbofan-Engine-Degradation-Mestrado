"""AE coordenado: inspect é somente leitura; optimize e final são explícitos e separados."""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import gc
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.anomaly_detection.v2.experiment import (
    DATA_ROOT, MODEL_ROOT, RESULT_ROOT, PROTOCOL, SEARCH_SPACE, RUL_DIR,
    assert_model_shape, check_roots, ensure_config, exclusive_lock, experiment_path,
    identity, load_data, load_frozen_model, read_best, read_json, remaining_trials,
    study_config, trial_counts, validate_params, write_json,
)
from src.rul_prediction.v2.utils import DATASETS, file_hash


def suggest_params(trial):
    return {name: (trial.suggest_float(name, **space) if isinstance(space, dict)
                   else trial.suggest_categorical(name, space))
            for name, space in SEARCH_SPACE.items()}


def prepare_runtime(seed):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, NotImplementedError) as exc:
        warnings.warn(f'Determinismo não disponível neste ambiente: {exc}', RuntimeWarning)


def clear_runtime():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()


def build_model(shape, params):
    # Mesmo builder histórico, sem alterar arquitetura ou importar seu treinamento.
    from src.anomaly_detection.models.autoencoder_models import build_lstm_autoencoder
    return build_lstm_autoencoder(shape, params)


def callbacks(patience, trial=None):
    from tensorflow import keras
    result = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                           restore_best_weights=True)]
    if trial is not None:
        from optuna_integration.tfkeras import TFKerasPruningCallback
        result.append(TFKerasPruningCallback(trial, 'val_loss'))
    return result


def fit_once(data, params, seed, epochs, patience, trial=None):
    validate_params(params)
    if (any(type(v) is not int for v in (seed, epochs, patience))
            or not 0 <= seed < 2**32 or epochs < 1 or patience < 0):
        raise ValueError('Seed/épocas/patience inválidos.')
    X, val = data['ae_fit'], data['ae_inner_validation']
    if (X.ndim != 3 or X.shape[1] != 30 or X.shape[2] < 1 or not len(X) or not len(val)
            or val.ndim != 3 or val.shape[1:] != X.shape[1:]
            or not np.isfinite(X).all() or not np.isfinite(val).all()):
        raise ValueError('Arrays nominais incompatíveis.')
    try:
        prepare_runtime(seed)
        model = build_model(X.shape[1:], params)
        assert_model_shape(model, X.shape[1:])
        history = model.fit(X, X, validation_data=(val, val), epochs=epochs,
                            batch_size=params['batch_size'], shuffle=True, verbose=2,
                            callbacks=callbacks(patience, trial)).history
        history = {k: [float(v) for v in values] for k, values in history.items()}
        if (not history.get('loss') or not history.get('val_loss')
                or len(history['loss']) != len(history['val_loss'])
                or any(not np.isfinite(v).all() or (np.asarray(v) < 0).any()
                       for v in history.values())):
            raise ValueError('Histórico de MSE vazio/não finito/inválido.')
        return model, history
    finally:
        clear_runtime()


def open_study(folder, config):
    import optuna
    study = optuna.create_study(
        study_name='ae_coordinated_rul_v2',
        storage='sqlite:///' + (Path(folder) / 'study.db').resolve().as_posix(),
        direction='minimize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config['sampler_seed']),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    stored = study.user_attrs.get('protocol')
    if stored is None:
        if study.trials:
            raise ValueError('Banco com trials sem protocolo; retomada recusada.')
        study.set_user_attr('protocol', config)
    elif stored != config:
        raise ValueError('Metadados do banco incompatíveis com study_config.json.')
    if study.direction.name != 'MINIMIZE':
        raise ValueError('Direção de estudo incompatível.')
    return study


def optimize(data_root=DATA_ROOT, model_root=MODEL_ROOT, result_root=RESULT_ROOT,
             dataset='FD001', experiment='baseline', seed=42, trials=30, epochs=50,
             patience=5, rul_dir=RUL_DIR, fail_interrupted=False):
    check_roots(data_root, rul_dir, model_root, result_root)
    folder = experiment_path(model_root, dataset, experiment)
    data, provenance = load_data(Path(data_root) / dataset, rul_dir)
    config = study_config(provenance, experiment, seed, trials, epochs, patience)
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(folder):
        ensure_config(folder, config)  # Compatibilidade antes de abrir o SQLite.
        study = open_study(folder, config)
        counts = trial_counts(study.trials)
        if counts['WAITING'] or len(study.trials) > trials:
            raise ValueError('WAITING/orçamento excedido: intervenção manual necessária.')
        if fail_interrupted:
            # Opção explícita, somente após o usuário confirmar que o worker terminou.
            from optuna.trial import TrialState
            for trial in study.trials:
                if trial.state.name == 'RUNNING':
                    study.tell(trial.number, state=TrialState.FAIL)
        remaining = remaining_trials(study.trials, trials)
        print(f'{dataset}: {trial_counts(study.trials)}; faltam {remaining}/{trials} trials.')

        def objective(trial):
            model, history = fit_once(data, suggest_params(trial), (seed + trial.number) % 2**32,
                                       epochs, patience, trial)
            del model
            return min(history['val_loss'])

        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        if remaining_trials(study.trials, trials) != 0:
            raise ValueError('Busca incompleta; nenhum snapshot final será publicado.')
        if not any(t.state.name == 'COMPLETE' for t in study.trials):
            raise ValueError('Nenhum trial completo; não há campeão.')
        best = {'config_sha256': identity(config), 'trial': study.best_trial.number,
                'params': study.best_params, 'validation_mse': float(study.best_value),
                'completed_budget': trials}
        validate_params(best['params'])
        if not np.isfinite(best['validation_mse']) or best['validation_mse'] < 0:
            raise ValueError('Objetivo campeão inválido.')
        target = folder / f'best_trial_{best["trial"]}.json'
        if target.exists():
            if read_json(target) != best:
                raise ValueError('Snapshot existente incompatível.')
        else:
            write_json(target, best)
        print(f'{dataset}: {trial_counts(study.trials)}; campeão: {target}; MSE={study.best_value:.8g}')
        return target


def plot_history(history, destination):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for key in ('loss', 'val_loss'):
        ax.plot(np.arange(1, len(history[key]) + 1), history[key], label=key)
    ax.set(xlabel='Epoch', ylabel='Reconstruction MSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination)


def train_final(data_root=DATA_ROOT, model_root=MODEL_ROOT, result_root=RESULT_ROOT,
                dataset='FD001', experiment='baseline', seed=42, best_trial=0,
                epochs=50, patience=5, rul_dir=RUL_DIR):
    check_roots(data_root, rul_dir, model_root, result_root)
    if (any(type(v) is not int for v in (seed, best_trial, epochs, patience))
            or not 0 <= seed < 2**32
            or best_trial < 0 or epochs < 1 or patience < 0):
        raise ValueError('Seed/trial/épocas/patience inválidos.')
    folder = experiment_path(model_root, dataset, experiment)
    result_folder = experiment_path(result_root, dataset, experiment)
    target, results = folder / f'seed_{seed}', result_folder / f'seed_{seed}'
    data, provenance = load_data(Path(data_root) / dataset, rul_dir)
    with exclusive_lock(folder):
        config = read_json(folder / 'study_config.json')
        expected = study_config(provenance, experiment, config['sampler_seed'],
                                config['total_trials'], config['epochs'], config['patience'])
        if expected != config:
            raise ValueError('Dados/código/configuração mudaram desde a busca.')
        best = read_best(folder, config, best_trial)
        if target.exists() or results.exists():
            raise FileExistsError('Seed/modelo/resultados existentes preservados, inclusive parciais.')
        target.mkdir()
        results.mkdir(parents=True)
        final_config = {'study': config, 'seed': seed, 'epochs': epochs,
                        'patience': patience, 'best': best}
        write_json(target / 'config.json', final_config)
        model, history = fit_once(data, best['params'], seed, epochs, patience)
        try:
            model.save(target / 'model.keras')
            write_json(results / 'history.json', history)
            plot_history(history, results / 'history.png')
            score = min(history['val_loss'])
            write_json(results / 'summary.json', {'Dataset': dataset, 'experiment': experiment,
                       'seed': seed, 'best_trial': best_trial, 'best_validation_mse': score,
                       'best_epoch': int(np.argmin(history['val_loss'])) + 1,
                       'epochs_run': len(history['val_loss']), 'params': best['params'],
                       'preprocessing_manifest_sha256': provenance['manifest_sha256']})
            # Manifesto por último: uma pasta parcial não é um modelo concluído.
            write_json(target / 'manifest.json', {'protocol': PROTOCOL, 'data': provenance,
                       'experiment': experiment, 'seed': seed, 'best_validation_mse': score,
                       'created_utc': datetime.now(timezone.utc).isoformat(),
                       'config_sha256': identity(final_config),
                       'sha256': {'model': {n: file_hash(target / n) for n in ('model.keras', 'config.json')},
                                  'results': {n: file_hash(results / n) for n in
                                              ('history.json', 'history.png', 'summary.json')}}})
        finally:
            del model
        print(f'{dataset}: modelo final salvo em {target}; MSE nominal interno={score:.8g}')
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'optimize', 'final'):
        p = sub.add_parser(command)
        p.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
        p.add_argument('--experiment', default='baseline')
        p.add_argument('--data-dir', type=Path, default=DATA_ROOT)
        p.add_argument('--rul-dir', type=Path, default=RUL_DIR)
        if command != 'inspect':
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--epochs', type=int, default=50)
            p.add_argument('--patience', type=int, default=5)
        if command == 'optimize':
            p.add_argument('--trials', type=int, default=30, help='TOTAL incluindo COMPLETE, PRUNED e FAIL.')
            p.add_argument('--fail-interrupted', action='store_true',
                           help='Após confirmar interrupção, converter RUNNING em FAIL; WAITING é recusado.')
        if command in ('inspect', 'final'):
            p.add_argument('--best-trial', type=int, required=command == 'final',
                           help='Número N do snapshot best_trial_N.json.')
        if command == 'inspect':
            p.add_argument('--model-seed', type=int,
                           help='Opcional: verificar/carregar modelo salvo, sem executar inferência.')
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Não repita datasets.')
    for dataset in args.datasets:
        if args.command == 'inspect':
            folder = experiment_path(MODEL_ROOT, dataset, args.experiment)
            data, provenance = load_data(args.data_dir / dataset, args.rul_dir)
            print(dataset, {k: v.shape for k, v in data.items()}, provenance['summary'])
            print('Manifesto AE SHA256:', provenance['manifest_sha256'])
            if args.best_trial is not None:
                config = read_json(folder / 'study_config.json')
                expected = study_config(provenance, args.experiment, config['sampler_seed'],
                                        config['total_trials'], config['epochs'], config['patience'])
                if config != expected:
                    raise ValueError('Estudo incompatível com ambiente/dados atuais.')
                print(read_best(folder, config, args.best_trial))
            if args.model_seed is not None:
                model = load_frozen_model(args.data_dir, MODEL_ROOT, RESULT_ROOT, dataset,
                                          args.experiment, args.model_seed, args.rul_dir)
                print('Modelo carregado:', model.input_shape, model.output_shape)
                del model
                clear_runtime()
        elif args.command == 'optimize':
            optimize(args.data_dir, MODEL_ROOT, RESULT_ROOT, dataset, args.experiment,
                     args.seed, args.trials, args.epochs, args.patience, args.rul_dir,
                     args.fail_interrupted)
        else:
            train_final(args.data_dir, MODEL_ROOT, RESULT_ROOT, dataset, args.experiment,
                        args.seed, args.best_trial, args.epochs, args.patience, args.rul_dir)


if __name__ == '__main__':
    main()
