"""Etapa 2: somente fixtures temporárias e mocks, sem redes ou estudos reais."""

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.common.paths import ROOT
from src.rul_prediction.v2 import evaluation as ev, experiment as ex, train as tr
from src.rul_prediction.v2.preprocessing import process_dataset
from src.rul_prediction.v2.utils import Config, file_hash
from tests.rul_prediction.test_v2_preprocessing import synthetic_sources


def trial(number, state='COMPLETE'):
    return SimpleNamespace(number=number, state=SimpleNamespace(name=state))


class V2TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        raw = cls.base / 'raw'
        synthetic_sources(raw)
        cls.data_root = cls.base / 'data'
        process_dataset(Config('FD001'), raw, cls.data_root)
        cls.folder = cls.data_root / 'FD001'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.models = Path(self.work.name) / 'models'
        self.results = Path(self.work.name) / 'results'

    def scoped(self, splits=('train', 'validation')):
        return ex.load_data(self.folder, splits)

    def fake_model(self, shape):
        fake = Mock(input_shape=(None, *shape), output_shape=(None, 1))
        fake.fit.return_value.history = {'loss': [5., 2.], 'val_loss': [6., 3.]}
        fake.save.side_effect = lambda path: Path(path).write_bytes(b'synthetic model; not Keras')
        fake.predict.side_effect = lambda x, **kw: np.zeros((len(x), 1))
        return fake

    def config_and_best(self, model='cnn_lstm'):
        data, provenance = self.scoped()
        config = tr.study_config(provenance, model, 'demo', 42, 30, 50, 5)
        folder = ex.experiment_path(self.models, 'FD001', model, 'demo')
        folder.mkdir(parents=True)
        tr.ensure_config(folder, config)
        params = {'filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dropout_rate': .1,
                  'learning_rate': .001, 'batch_size': 32}
        ex.write_json(folder / 'best_trial_0.json', {'config_sha256': ex.identity(config),
                      'params': params, 'trial': 0, 'validation_mse': 3.})
        return data, provenance, folder

    def final_fake(self, model='cnn_lstm', seeds=(1,)):
        data, provenance, folder = self.config_and_best(model)
        fake = self.fake_model(provenance['input_shape'])
        with patch.object(tr, 'prepare_runtime'), patch.object(tr, 'clear_runtime'), \
                patch.object(tr, 'callbacks', return_value=[]), \
                patch.object(tr, 'build_model', return_value=fake), \
                patch.object(tr, 'plot_history', side_effect=lambda h, p: Path(p).write_bytes(b'plot')):
            tr.train_final(self.data_root, self.models, 'FD001', model, 'demo', list(seeds), 0)
        return fake, folder

    def test_scoped_hashes_and_npz_no_calibration_or_test_access(self):
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.name.startswith(('calibration', 'test_')):
                raise AssertionError(f'Reserva/teste acessado: {path}')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded):
            data, provenance = self.scoped()
        self.assertEqual(set(data), {'train', 'validation'})
        self.assertEqual(data['train']['X'].shape[1:], tuple(provenance['input_shape']))
        self.assertFalse(set(data['train']['metadata'].unit_nr) & set(data['validation']['metadata'].unit_nr))
        with self.assertRaises(ValueError):
            self.scoped(('calibration',))

    def test_hash_tampering_rejected(self):
        with patch.object(ex, 'file_hash', return_value='tampered'):
            with self.assertRaisesRegex(ValueError, 'Hash'):
                self.scoped()

    def test_metadata_misalignment_and_feature_order_rejected(self):
        original = pd.read_csv

        def changed(path, *args, **kwargs):
            frame = original(path, *args, **kwargs)
            frame.loc[0, 'RUL_capped'] += 1
            return frame

        with patch.object(ex.pd, 'read_csv', changed):
            with self.assertRaisesRegex(ValueError, 'targets'):
                self.scoped()
        original_json = ex.read_json

        def wrong_features(path):
            value = original_json(path)
            if Path(path).name == 'features.json':
                value['selected'].reverse()
            return value

        with patch.object(ex, 'read_json', wrong_features):
            with self.assertRaisesRegex(ValueError, 'features'):
                self.scoped()

    def test_training_explicit_validation_capped_targets_and_mock_fit(self):
        data, provenance = self.scoped()
        fake = self.fake_model(provenance['input_shape'])
        with patch.object(tr, 'prepare_runtime') as runtime, patch.object(tr, 'clear_runtime'), \
                patch.object(tr, 'callbacks', return_value=[]), patch.object(tr, 'build_model', return_value=fake):
            _, history = tr.fit_once(data, 'cnn_lstm', {'batch_size': 32}, 7, 50, 5)
        runtime.assert_called_once_with(7)
        args, kwargs = fake.fit.call_args
        self.assertIs(args[0], data['train']['X'])
        self.assertIs(args[1], data['train']['y'])
        self.assertIs(kwargs['validation_data'][0], data['validation']['X'])
        self.assertNotIn('validation_split', kwargs)
        self.assertLessEqual(args[1].max(), 125)
        self.assertEqual(min(history['val_loss']), 3.)

    def test_model_shape_compatibility(self):
        ex.assert_model_shape(self.fake_model((30, 5)), [30, 5])
        with self.assertRaises(ValueError):
            ex.assert_model_shape(self.fake_model((30, 5)), [30, 6])

    def test_total_budget_complete_pruned_failed_and_interrupted(self):
        counted = [trial(i, ('COMPLETE', 'PRUNED', 'FAIL')[i % 3]) for i in range(20)]
        self.assertEqual(tr.remaining_trials(counted, 30), 10)
        self.assertEqual(tr.remaining_trials(counted, 20), 0)
        for state in ('RUNNING', 'WAITING'):
            with self.assertRaises(ValueError):
                tr.remaining_trials([trial(0, state)], 30)
        with self.assertRaises(ValueError):
            tr.remaining_trials(counted, 10)

    def test_config_persistence_and_incompatible_resume(self):
        folder = Path(self.work.name)
        tr.ensure_config(folder, {'identity': 'a'})
        tr.ensure_config(folder, {'identity': 'a'})
        with self.assertRaises(ValueError):
            tr.ensure_config(folder, {'identity': 'b'})
        with self.assertRaises(FileExistsError):
            ex.write_json(folder / 'study_config.json', {})

    def test_mock_optimize_only_remaining_and_no_reserved_arrays(self):
        _, provenance = self.scoped()
        config = tr.study_config(provenance, 'cnn_lstm', 'demo', 42, 30, 50, 5)
        study = Mock()
        study.trials = [trial(i) for i in range(20)]
        study.best_trial, study.best_params, study.best_value = trial(0), {'batch_size': 32}, 3.
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.name.startswith(('test_', 'calibration')):
                raise AssertionError('Vazamento')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded), patch.object(tr, 'open_study', return_value=study), \
                patch.object(tr, 'fit_once') as fit:
            tr.optimize(self.data_root, self.models, 'FD001', 'cnn_lstm', 'demo')
        study.optimize.assert_called_once()
        self.assertEqual(study.optimize.call_args.kwargs['n_trials'], 10)
        fit.assert_not_called()  # optimize é mock; nenhum estudo é executado.
        folder = ex.experiment_path(self.models, 'FD001', 'cnn_lstm', 'demo')
        self.assertEqual(ex.read_json(folder / 'study_config.json'), config)

    def test_objective_uses_min_validation_mse_and_no_real_optimizer(self):
        study = Mock(trials=[trial(0)], best_params={'batch_size': 32}, best_trial=trial(0), best_value=3.)
        fitted = Mock()
        with patch.object(tr, 'open_study', return_value=study), \
                patch.object(tr, 'suggest_params', return_value={'batch_size': 32}), \
                patch.object(tr, 'fit_once', return_value=(fitted, {'val_loss': [8., 3., 4.]})) as fit:
            tr.optimize(self.data_root, self.models, 'FD001', 'cnn_lstm', 'demo')
            objective = study.optimize.call_args.args[0]
            self.assertEqual(objective(trial(21)), 3.)
        self.assertEqual(set(fit.call_args.args[0]), {'train', 'validation'})
        self.assertEqual(fit.call_args.args[3], 42 + 21)

    def test_resume_rejects_configuration_before_opening_database(self):
        self.config_and_best()
        with patch.object(tr, 'open_study') as opener, self.assertRaises(ValueError):
            tr.optimize(self.data_root, self.models, 'FD001', 'cnn_lstm', 'demo', epochs=51)
        opener.assert_not_called()

    def test_search_spaces_and_residual_dimensions(self):
        fake = Mock()
        fake.suggest_categorical.side_effect = lambda name, values: values[-1]
        fake.suggest_int.side_effect = lambda name, low, high: high
        fake.suggest_float.side_effect = lambda name, low, high, **kwargs: high
        cnn = tr.suggest_params(fake, 'cnn_lstm')
        transformer = tr.suggest_params(fake, 'transformer')
        self.assertEqual(cnn['filters'], 128)
        self.assertEqual(cnn['kernel_size'], 5)
        self.assertEqual(transformer['num_layers'], 3)
        for width in (32, 64, 128):
            for heads in (2, 4, 8):
                self.assertEqual(width % heads, 0)

    def test_runtime_cleans_up_when_mock_fit_fails(self):
        data, provenance = self.scoped()
        fake = self.fake_model(provenance['input_shape'])
        fake.fit.side_effect = RuntimeError('synthetic failure')
        with patch.object(tr, 'prepare_runtime'), patch.object(tr, 'clear_runtime') as clear, \
                patch.object(tr, 'callbacks', return_value=[]), \
                patch.object(tr, 'build_model', return_value=fake), self.assertRaises(RuntimeError):
            tr.fit_once(data, 'cnn_lstm', {'batch_size': 32}, 1, 50, 5)
        clear.assert_called_once()

    def test_model_builder_bridge_uses_existing_architectures_without_instantiating(self):
        import types
        module = types.ModuleType('src.rul_prediction.models.model_builders')
        module.build_cnn_lstm, module.build_transformer = Mock(), Mock()
        with patch.dict(sys.modules, {'src.rul_prediction.models.model_builders': module}):
            tr.build_model('cnn_lstm', (30, 19), {'batch_size': 32})
            tr.build_model('transformer', (30, 19), {'batch_size': 64})
        module.build_cnn_lstm.assert_called_once_with((30, 19), {'batch_size': 32})
        module.build_transformer.assert_called_once_with((30, 19), {'batch_size': 64})

    def test_final_saves_identity_history_all_seeds_and_refuses_overwrite(self):
        fake, folder = self.final_fake(seeds=(1, 2))
        self.assertEqual(fake.fit.call_count, 2)
        for seed in (1, 2):
            manifest = ex.read_json(folder / f'seed_{seed}' / 'manifest.json')
            self.assertEqual(manifest['seed'], seed)
            self.assertEqual(manifest['best_validation_mse'], 3.)
        with patch.object(tr, 'fit_once') as fit:
            with self.assertRaises(FileExistsError):
                tr.train_final(self.data_root, self.models, 'FD001', 'cnn_lstm', 'demo', [1], 0)
            fit.assert_not_called()

    def test_uncapped_metrics_known_values_no_prediction_clipping(self):
        scores = ev.metrics([140, 7], [125, -3])
        self.assertAlmostEqual(scores['RMSE'], np.sqrt(162.5))
        self.assertEqual(scores['MAE'], 12.5)
        self.assertEqual(scores['MedianAE'], 12.5)
        self.assertEqual(scores['P95AE'], 14.75)
        self.assertEqual(scores['MaxAE'], 15.)
        for bad in ([], [np.nan], [[1.]]):
            with self.assertRaises(ValueError):
                ev.metrics(bad, bad)

    def test_terminal_one_engine_join_keys_and_uncapped_predictions(self):
        data, _ = self.scoped(('test_terminal',))
        part = data['test_terminal']
        table = ev.prediction_table(part, np.array([[125.], [-3.]]), 'cnn_lstm', 1, 'demo')
        self.assertEqual(len(table), 2)
        self.assertTrue(table.unit_nr.is_unique)
        self.assertEqual(table.RUL_true.tolist(), [140, 7])
        self.assertEqual(table.RUL_pred.tolist(), [125, -3])
        self.assertTrue(set(ex.JOIN_KEYS) <= set(table))
        self.assertEqual(table.AbsoluteError.tolist(), [15, 10])
        original = pd.read_csv

        def duplicate(path, *args, **kwargs):
            meta = original(path, *args, **kwargs)
            meta.loc[1, 'unit_nr'] = meta.loc[0, 'unit_nr']
            return meta

        with patch.object(ex.pd, 'read_csv', duplicate), self.assertRaises(ValueError):
            self.scoped(('test_terminal',))

    def test_mock_evaluation_and_comparison_preserve_seeds_and_protocols(self):
        for model in ex.MODELS:
            fake, _ = self.final_fake(model, seeds=(1, 2))
            with patch.object(ev, 'load_model', return_value=fake), patch.object(ev, 'clear_runtime'), \
                    patch.object(ev, 'plot_predictions', side_effect=lambda t, p: Path(p).write_bytes(b'plot')):
                for protocol in ('terminal', 'windows'):
                    ev.evaluate(self.data_root, self.models, self.results, 'FD001', model,
                                'demo', [1, 2], protocol)
        individuals, summary = ev.compare(self.results, ['FD001'], 'demo', [1, 2], 'terminal', 'both')
        self.assertEqual(len(individuals), 4)
        self.assertEqual(len(summary), 2)
        self.assertEqual(set(individuals.Protocol), {'terminal'})
        self.assertEqual(set(individuals.Seed), {1, 2})
        self.assertTrue((summary['MAE_std'] == 0).all())
        with self.assertRaises(FileExistsError):
            ev.compare(self.results, ['FD001'], 'demo', [1, 2], 'terminal', 'both')
        with self.assertRaises(FileNotFoundError):
            ev.compare(self.results, ['FD001'], 'demo', [1, 2, 3], 'terminal', 'missing')
        with patch.object(ev, 'load_model') as loader, self.assertRaises(FileExistsError):
            ev.evaluate(self.data_root, self.models, self.results, 'FD001', 'cnn_lstm', 'demo', [1])
        loader.assert_not_called()

    def test_historical_destinations_and_path_traversal_rejected(self):
        for root in (ROOT / 'models', ROOT / 'results', ROOT / 'data/processed_data', ROOT / 'src'):
            with self.assertRaises(ValueError):
                ex.check_output_root(root, ex.MODEL_ROOT)
        ex.check_output_root(ex.MODEL_ROOT, ex.MODEL_ROOT)
        with self.assertRaises(ValueError):
            ex.experiment_path(self.models, 'FD001', 'cnn_lstm', '../old')

    def test_all_cli_help_do_not_import_tensorflow_or_execute(self):
        for module, commands in (('train', ('', 'inspect', 'optimize', 'final')),
                                 ('evaluation', ('', 'terminal', 'windows', 'compare'))):
            for command in commands:
                code = ("import runpy, sys; sys.modules['tensorflow'] = None; sys.modules['optuna'] = None; "
                        f"sys.argv = ['cli', *{[command] if command else []!r}, '--help']; "
                        f"runpy.run_module('src.rul_prediction.v2.{module}', run_name='__main__')")
                result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
