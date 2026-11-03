"""Treinamento somente simulado: TensorFlow e Optuna nunca executam fit/estudos reais."""

from pathlib import Path
from types import SimpleNamespace
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.anomaly_detection.v2 import experiment as exp, train
from src.anomaly_detection.v2.preprocessing import process_dataset
from src.anomaly_detection.v2.utils import Config
from tests.anomaly_detection.test_coordinated_preprocessing import fixture


PARAMS = {'latent_dim': 8, 'encoder_units': 32, 'num_layers': 1,
          'dropout_rate': 0.0, 'learning_rate': .001, 'batch_size': 32}
HISTORY = {'loss': [.3, .2], 'val_loss': [.4, .25]}


def frozen(number, state='COMPLETE'):
    return SimpleNamespace(number=number, state=SimpleNamespace(name=state))


class FakeStudy:
    def __init__(self, states=()):
        self.trials = [frozen(i, state) for i, state in enumerate(states)]
        self.best_trial = frozen(0)
        self.best_params = dict(PARAMS)
        self.best_value = .25
        self.requested = []
        self.user_attrs = {}
        self.direction = SimpleNamespace(name='MINIMIZE')

    def optimize(self, objective, n_trials, **kwargs):
        self.requested.append(n_trials)
        for _ in range(n_trials):
            number = len(self.trials)
            trial = SimpleNamespace(number=number,
                                    suggest_categorical=lambda name, choices: PARAMS[name],
                                    suggest_float=lambda name, **space: PARAMS[name])
            assert objective(trial) == .25
            self.trials.append(frozen(number))

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value

    def tell(self, number, state):
        self.trials[number].state = SimpleNamespace(name=state.name)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        _, self.raw, self.rul, _ = fixture(self.base)
        self.data = self.base / 'ae'
        self.models, self.results = self.base / 'models', self.base / 'results'
        process_dataset(Config('FD001'), self.raw, self.rul, self.data)
        self.folder = self.models / 'FD001/baseline'
        self.kw = dict(data_root=self.data, model_root=self.models, result_root=self.results,
                       rul_dir=self.rul, trials=3)

    def fake_model(self, features=20):
        model = Mock(input_shape=(None, 30, features), output_shape=(None, 30, features))
        model.fit.return_value = SimpleNamespace(history=HISTORY)
        model.save.side_effect = lambda path: Path(path).write_bytes(b'mocked-keras-not-a-real-model')
        return model

    def optimize(self, study=None):
        study = FakeStudy() if study is None else study
        with patch.object(train, 'open_study', return_value=study), \
                patch.object(train, 'fit_once', return_value=(Mock(), HISTORY)) as fit:
            path = train.optimize(**self.kw)
        return study, path, fit

    def final(self):
        self.optimize()
        with patch.object(train, 'fit_once', return_value=(self.fake_model(), HISTORY)) as fit:
            target = train.train_final(self.data, self.models, self.results, rul_dir=self.rul)
        return target, fit

    def test_load_only_nominal_arrays_and_never_fit_transform(self):
        paths = []
        original = Path.open

        def guarded(path, *args, **kwargs):
            paths.append(str(path))
            self.assertNotIn(path.name, ('calibration.npz', 'test_terminal.npz', 'train.npz',
                                         'validation.npz', 'test_windows.npz'))
            self.assertFalse(path.name.startswith(('test_', 'RUL_')))
            return original(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded), \
                patch('src.anomaly_detection.v2.utils.CoordinatedNominalPreprocessor.fit', side_effect=AssertionError), \
                patch('src.anomaly_detection.v2.utils.CoordinatedNominalPreprocessor.transform', side_effect=AssertionError):
            data, provenance = exp.load_data(self.data / 'FD001', self.rul)
        self.assertEqual(set(data), {'ae_fit', 'ae_inner_validation'})
        self.assertTrue(any(p.endswith('nominal_fit.npz') for p in paths))
        self.assertTrue(any(p.endswith('nominal_inner_validation.npz') for p in paths))
        self.assertEqual(data['ae_fit'].shape[1:], tuple(provenance['input_shape']))
        splits = provenance['splits']
        for role in data:
            self.assertTrue(set(splits[role]) <= set(splits['rul_splits']['train']))
            self.assertFalse(set(splits[role]) & set(splits['rul_splits']['validation']))
            self.assertFalse(set(splits[role]) & set(splits['rul_splits']['calibration']))

    def test_model_fit_reconstructs_X_with_independent_validation(self):
        data, provenance = exp.load_data(self.data / 'FD001', self.rul)
        model = self.fake_model(len(provenance['features']))
        with patch.object(train, 'prepare_runtime') as runtime, patch.object(train, 'clear_runtime') as clear, \
                patch.object(train, 'build_model', return_value=model), \
                patch.object(train, 'callbacks', return_value=['mock-early-stop']):
            _, history = train.fit_once(data, PARAMS, 42, 50, 5)
        args, kwargs = model.fit.call_args
        self.assertIs(args[0], data['ae_fit'])
        self.assertIs(args[1], args[0])
        self.assertIs(kwargs['validation_data'][0], data['ae_inner_validation'])
        self.assertIs(kwargs['validation_data'][1], data['ae_inner_validation'])
        self.assertNotIn('validation_split', kwargs)
        self.assertEqual(kwargs['epochs'], 50)
        self.assertEqual(history, HISTORY)
        runtime.assert_called_once_with(42)
        clear.assert_called_once()

    def test_wrong_model_shape_rejected_before_fit_and_resources_cleared(self):
        data, _ = exp.load_data(self.data / 'FD001', self.rul)
        model = self.fake_model()
        model.output_shape = (None, 1)
        with patch.object(train, 'prepare_runtime'), patch.object(train, 'clear_runtime') as clear, \
                patch.object(train, 'build_model', return_value=model):
            with self.assertRaises(ValueError):
                train.fit_once(data, PARAMS, 42, 50, 5)
        model.fit.assert_not_called()
        clear.assert_called_once()

    def test_bad_arrays_rejected_before_runtime(self):
        data, _ = exp.load_data(self.data / 'FD001', self.rul)
        data['ae_fit'][0, 0, 0] = np.nan
        with patch.object(train, 'prepare_runtime') as runtime, self.assertRaises(ValueError):
            train.fit_once(data, PARAMS, 42, 50, 5)
        runtime.assert_not_called()

    def test_nonfinite_history_rejected_and_cleared(self):
        data, provenance = exp.load_data(self.data / 'FD001', self.rul)
        model = self.fake_model(len(provenance['features']))
        model.fit.return_value.history = {'loss': [.2], 'val_loss': [np.nan]}
        with patch.object(train, 'prepare_runtime'), patch.object(train, 'clear_runtime') as clear, \
                patch.object(train, 'callbacks'), patch.object(train, 'build_model', return_value=model):
            with self.assertRaises(ValueError):
                train.fit_once(data, PARAMS, 42, 50, 5)
        clear.assert_called_once()

    def test_exact_historical_search_space(self):
        trial = Mock()
        trial.suggest_categorical.side_effect = lambda name, choices: PARAMS[name]
        trial.suggest_float.side_effect = lambda name, **kwargs: PARAMS[name]
        self.assertEqual(train.suggest_params(trial), PARAMS)
        trial.suggest_float.assert_called_once_with('learning_rate', low=1e-4, high=1e-2, log=True)
        self.assertEqual(trial.suggest_categorical.call_count, 5)
        exp.validate_params(PARAMS)
        with self.assertRaises(ValueError):
            exp.validate_params({**PARAMS, 'num_layers': 3})

    def test_runtime_seeds_python_numpy_and_keras_and_requests_determinism(self):
        keras = SimpleNamespace(backend=SimpleNamespace(clear_session=Mock()),
                                utils=SimpleNamespace(set_random_seed=Mock()))
        determinism = Mock()
        tf = SimpleNamespace(keras=keras, config=SimpleNamespace(experimental=SimpleNamespace(
                             enable_op_determinism=determinism)))
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with patch.dict(sys.modules, {'tensorflow': tf}):
                train.prepare_runtime(18)
                first = (random.random(), np.random.rand())
                train.prepare_runtime(18)
                self.assertEqual(first, (random.random(), np.random.rand()))
            self.assertEqual(keras.utils.set_random_seed.call_args.args, (18,))
            self.assertEqual(determinism.call_count, 2)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def test_cleanup_on_simulated_fit_failure(self):
        data, provenance = exp.load_data(self.data / 'FD001', self.rul)
        model = self.fake_model(len(provenance['features']))
        model.fit.side_effect = RuntimeError('mock interruption')
        with patch.object(train, 'prepare_runtime'), patch.object(train, 'clear_runtime') as clear, \
                patch.object(train, 'callbacks'), patch.object(train, 'build_model', return_value=model):
            with self.assertRaises(RuntimeError):
                train.fit_once(data, PARAMS, 42, 50, 5)
        clear.assert_called_once()

    def test_objective_uses_minimum_not_last_validation_mse(self):
        study = FakeStudy()
        history = {'loss': [.5, .3, .4], 'val_loss': [.4, .25, .6]}
        with patch.object(train, 'open_study', return_value=study), \
                patch.object(train, 'fit_once', return_value=(Mock(), history)):
            path = train.optimize(**self.kw)
        self.assertEqual(exp.read_json(path)['validation_mse'], .25)

    def test_all_failed_trials_produce_no_champion(self):
        study = FakeStudy(['FAIL'] * 3)
        with patch.object(train, 'open_study', return_value=study), self.assertRaises(ValueError):
            train.optimize(**self.kw)
        self.assertFalse(list(self.folder.glob('best_trial_*.json')))

    def test_existing_database_without_config_refused(self):
        self.folder.mkdir(parents=True)
        (self.folder / 'study.db').write_bytes(b'existing')
        with patch.object(train, 'open_study') as opened, self.assertRaises(ValueError):
            train.optimize(**self.kw)
        opened.assert_not_called()
        self.assertFalse((self.folder / 'study_config.json').exists())

    def test_final_incompatible_snapshot_rejected_before_fit(self):
        self.optimize()
        snapshot = self.folder / 'best_trial_0.json'
        best = exp.read_json(snapshot)
        best['config_sha256'] = 'wrong'
        # Alteração apenas de uma fixture temporária para simular corrupção.
        with patch.object(exp, 'read_json', side_effect=lambda path: best if Path(path) == snapshot
                          else __import__('json').loads(Path(path).read_text(encoding='utf-8'))), \
                patch.object(train, 'fit_once') as fit, self.assertRaises(ValueError):
            train.train_final(self.data, self.models, self.results, rul_dir=self.rul)
        fit.assert_not_called()
        self.assertFalse((self.folder / 'seed_42').exists())

    def test_multiregime_feature_dimension_is_not_hardcoded(self):
        multi = self.base / 'multi'
        multi.mkdir()
        _, raw, rul, _ = fixture(multi, dataset='FD002')
        process_dataset(Config('FD002'), raw, rul, multi / 'ae')
        data, provenance = exp.load_data(multi / 'ae/FD002', rul)
        model = self.fake_model(len(provenance['features']))
        with patch.object(train, 'prepare_runtime'), patch.object(train, 'clear_runtime'), \
                patch.object(train, 'callbacks'), patch.object(train, 'build_model', return_value=model) as build:
            train.fit_once(data, PARAMS, 42, 50, 5)
        self.assertEqual(build.call_args.args[0], (30, len(provenance['features'])))

    def test_early_stopping_and_keras_pruning_are_configured(self):
        early = Mock()
        pruning = Mock()
        tf = SimpleNamespace(keras=SimpleNamespace(callbacks=SimpleNamespace(EarlyStopping=early)))
        with patch.dict(sys.modules, {'tensorflow': tf,
                        'optuna_integration.tfkeras': SimpleNamespace(TFKerasPruningCallback=pruning)}):
            trial = Mock()
            train.callbacks(5, trial)
        early.assert_called_once_with(monitor='val_loss', patience=5, restore_best_weights=True)
        pruning.assert_called_once_with(trial, 'val_loss')

    def test_budget_counts_complete_pruned_fail_and_no_repeat(self):
        study = FakeStudy(['COMPLETE', 'PRUNED'])
        _, path, fit = self.optimize(study)
        self.assertEqual(study.requested, [1])
        self.assertEqual(fit.call_args.args[2], 44)
        self.assertEqual(exp.read_json(path)['validation_mse'], .25)
        self.assertEqual(exp.trial_counts(study.trials)['PRUNED'], 1)
        self.optimize(study)
        self.assertEqual(study.requested, [1])
        self.assertEqual(exp.remaining_trials([frozen(0, 'FAIL')], 3), 2)

    def test_running_waiting_and_exceeded_budget_rejected(self):
        for states in (['RUNNING'], ['WAITING'], ['COMPLETE'] * 4):
            with self.subTest(states=states), patch.object(train, 'open_study', return_value=FakeStudy(states)), \
                    patch.object(train, 'fit_once') as fit, self.assertRaises(ValueError):
                train.optimize(**self.kw)
            fit.assert_not_called()

    def test_explicit_interrupted_trial_is_failed_and_consumes_budget(self):
        study = FakeStudy(['RUNNING', 'COMPLETE'])
        state = SimpleNamespace(name='FAIL')
        with patch.object(train, 'open_study', return_value=study), \
                patch.object(train, 'fit_once', return_value=(Mock(), HISTORY)), \
                patch.dict(sys.modules, {'optuna.trial': SimpleNamespace(TrialState=SimpleNamespace(FAIL=state))}):
            train.optimize(**self.kw, fail_interrupted=True)
        self.assertEqual(study.trials[0].state.name, 'FAIL')
        self.assertEqual(study.requested, [1])

    def test_incompatible_config_refused_before_opening_study(self):
        self.optimize()
        for changed in ({'epochs': 51}, {'seed': 18}, {'trials': 4}, {'patience': 7}):
            with patch.object(train, 'open_study') as opened, self.assertRaises(ValueError):
                train.optimize(**{**self.kw, **changed})
            opened.assert_not_called()

    def test_open_study_uses_sqlite_seed_pruner_and_protocol(self):
        _, provenance = exp.load_data(self.data / 'FD001', self.rul)
        config = exp.study_config(provenance, 'baseline')
        fake = FakeStudy()
        optuna = SimpleNamespace(create_study=Mock(return_value=fake),
                                 samplers=SimpleNamespace(TPESampler=Mock()),
                                 pruners=SimpleNamespace(MedianPruner=Mock()))
        with patch.dict(sys.modules, {'optuna': optuna}):
            train.open_study(self.folder, config)
            self.assertEqual(fake.user_attrs['protocol'], config)
            optuna.samplers.TPESampler.assert_called_once_with(seed=42)
            optuna.pruners.MedianPruner.assert_called_once_with(n_warmup_steps=5)
            self.assertTrue(optuna.create_study.call_args.kwargs['storage'].startswith('sqlite:///'))
            fake.user_attrs['protocol'] = {'wrong': True}
            with self.assertRaises(ValueError):
                train.open_study(self.folder, config)

    def test_final_model_manifest_history_and_overwrite_protection(self):
        target, fit = self.final()
        config = exp.read_json(target / 'config.json')
        manifest = exp.read_json(target / 'manifest.json')
        self.assertEqual(config['best']['params'], PARAMS)
        self.assertEqual(manifest['best_validation_mse'], .25)
        self.assertEqual(manifest['sha256']['model']['model.keras'], exp.file_hash(target / 'model.keras'))
        results = self.results / 'FD001/baseline/seed_42'
        self.assertEqual(exp.read_json(results / 'history.json'), HISTORY)
        self.assertTrue((results / 'history.png').is_file())
        data = fit.call_args.args[0]
        self.assertEqual(set(data), {'ae_fit', 'ae_inner_validation'})
        with patch.object(train, 'fit_once') as fit, self.assertRaises(FileExistsError):
            train.train_final(self.data, self.models, self.results, rul_dir=self.rul)
        fit.assert_not_called()
        self.assertFalse(list(self.base.rglob('*threshold*')))

    def test_partial_final_output_refused(self):
        self.optimize()
        (self.results / 'FD001/baseline/seed_42').mkdir(parents=True)
        with patch.object(train, 'fit_once') as fit, self.assertRaises(FileExistsError):
            train.train_final(self.data, self.models, self.results, rul_dir=self.rul)
        fit.assert_not_called()

    def test_interrupted_final_has_no_completion_manifest(self):
        self.optimize()
        with patch.object(train, 'fit_once', side_effect=RuntimeError('simulated stop')):
            with self.assertRaises(RuntimeError):
                train.train_final(self.data, self.models, self.results, rul_dir=self.rul)
        self.assertFalse((self.folder / 'seed_42/manifest.json').exists())
        self.assertTrue((self.folder / 'seed_42/config.json').exists())

    def test_saved_model_loader_only_loads_and_checks_shape_no_inference(self):
        target, _ = self.final()
        _, provenance = exp.load_data(self.data / 'FD001', self.rul)
        model = self.fake_model(len(provenance['features']))
        keras = SimpleNamespace(models=SimpleNamespace(load_model=Mock(return_value=model)))
        with patch.dict(sys.modules, {'tensorflow': SimpleNamespace(keras=keras)}):
            loaded = exp.load_frozen_model(self.data, self.models, self.results, 'FD001', 'baseline', 42, self.rul)
            self.assertIs(loaded, model)
            keras.models.load_model.assert_called_once_with(target / 'model.keras', compile=False)
            model.predict.assert_not_called()
            model.fit.assert_not_called()
            model.save.assert_not_called()
            (target / 'model.keras').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                exp.load_frozen_model(self.data, self.models, self.results, 'FD001', 'baseline', 42, self.rul)
            self.assertEqual(keras.models.load_model.call_count, 1)

    def test_corrupted_npz_blocks_optimize_before_study_or_fit(self):
        (self.data / 'FD001/nominal_fit.npz').write_bytes(b'bad')
        with patch.object(train, 'open_study') as opened, self.assertRaises(ValueError):
            train.optimize(**self.kw)
        opened.assert_not_called()
        self.assertFalse(self.folder.exists())

    def test_lock_compatibility_and_immutable_snapshot(self):
        self.optimize()
        with exp.exclusive_lock(self.folder), patch.object(train, 'open_study') as opened:
            with self.assertRaises(FileExistsError):
                train.optimize(**self.kw)
        opened.assert_not_called()
        best = exp.read_json(self.folder / 'best_trial_0.json')
        best['params']['latent_dim'] = 64
        with patch.object(train, 'read_json', side_effect=lambda path: best if Path(path).name == 'best_trial_0.json' else exp.read_json(path)), \
                patch.object(train, 'open_study', return_value=FakeStudy(['COMPLETE'] * 3)), self.assertRaises(ValueError):
            train.optimize(**self.kw)

    def test_identity_is_reproducible_and_changes_with_seed_or_protocol(self):
        _, provenance = exp.load_data(self.data / 'FD001', self.rul)
        config = exp.study_config(provenance, 'baseline')
        self.assertEqual(config, exp.study_config(provenance, 'baseline'))
        self.assertNotEqual(exp.identity(config), exp.identity(exp.study_config(provenance, 'baseline', seed=18)))
        for name in ('study_config.json', 'study.db'):
            self.assertFalse((self.folder / name).exists())

    def test_protected_historical_roots_and_invalid_experiments(self):
        for root in ('models/best_models_autoencoder', 'models/optuna_logs_autoencoder',
                     'models/rul_v2', 'data/processed_anomaly', 'data/processed_anomaly_v2'):
            with self.assertRaises(ValueError):
                exp.check_roots(self.data, self.rul, exp.ROOT / root, self.results)
        with self.assertRaises(ValueError):
            exp.check_roots(self.data, self.rul, self.data / 'bad', self.results)
        for name in ('../old', '', 'a/b'):
            with self.assertRaises(ValueError):
                exp.experiment_path(self.models, 'FD001', name)

    def test_cli_help_and_inspect_never_import_neural_training_libraries(self):
        for args in (['--help'], ['optimize', '--help'], ['final', '--help'],
                     ['inspect', '--data-dir', str(self.data), '--rul-dir', str(self.rul)]):
            script = ("import sys; sys.modules['tensorflow']=None; sys.modules['keras']=None; "
                      "sys.modules['optuna']=None; "
                      f"sys.argv={['ae-v2', *args]!r}; "
                      "from src.anomaly_detection.v2.train import main; main()")
            result = subprocess.run([sys.executable, '-B', '-c', script], text=True,
                                    capture_output=True, encoding='utf-8', errors='replace')
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.models.exists())
        self.assertFalse(self.results.exists())


if __name__ == '__main__':
    unittest.main()
