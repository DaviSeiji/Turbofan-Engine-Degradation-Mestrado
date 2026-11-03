"""Orquestração com mocks: nenhum treino, estudo ou avaliação real."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.rul_prediction.v2 import run_pipeline as runner
from src.rul_prediction.v2.experiment import write_json


class V2RunnerTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        folder = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(runner, 'MODEL_ROOT', folder / 'models'))
        stack.enter_context(patch.object(runner, 'RESULT_ROOT', folder / 'results'))
        stack.enter_context(patch.object(runner, 'DATASETS', ['FD001']))
        stack.enter_context(patch.object(runner, 'read_json', return_value={'trial': 7}))
        self.configure_logging = runner.configure_tensorflow_logging
        stack.enter_context(patch.object(runner, 'configure_tensorflow_logging'))
        self.calls = Mock()
        for module, name in ((runner.train, 'optimize'), (runner.train, 'train_final'),
                             (runner.evaluation, 'evaluate'), (runner.evaluation, 'compare')):
            method = stack.enter_context(patch.object(module, name))
            self.calls.attach_mock(method, name)

    def test_fd001_all_training_before_evaluation_and_automatic_champion(self):
        runner.run_pipeline()
        names = [call[0] for call in self.calls.mock_calls]
        self.assertEqual(names, ['optimize', 'train_final'] * 2 + ['evaluate'] * 4 + ['compare'] * 2)
        for call in self.calls.optimize.call_args_list:
            self.assertEqual(call.args[2], 'FD001')
            self.assertEqual(call.kwargs['trials'], 30)
        for call in self.calls.train_final.call_args_list:
            self.assertEqual(call.kwargs['best_trial'], 7)
            self.assertEqual(call.kwargs['seeds'], [1, 2, 3, 4, 5])

    def test_dry_run_never_executes_steps(self):
        runner.run_pipeline(dry_run=True)
        self.assertEqual(self.calls.mock_calls, [])
        runner.read_json.assert_not_called()

    def test_existing_seed_stops_before_optimization(self):
        path = runner.experiment_path(runner.MODEL_ROOT, 'FD001', 'cnn_lstm', runner.EXPERIMENTO) / 'seed_1'
        path.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            runner.run_pipeline()
        self.assertEqual(self.calls.mock_calls, [])

    def test_failed_training_prevents_test_access(self):
        self.calls.train_final.side_effect = RuntimeError('synthetic failure')
        with self.assertRaises(RuntimeError):
            runner.run_pipeline()
        self.calls.evaluate.assert_not_called()
        self.calls.compare.assert_not_called()

    def test_completed_dataset_skipped_but_included_in_comparison(self):
        with patch.object(runner, 'DATASETS', ['FD001', 'FD002']), \
                patch.object(runner, 'dataset_complete', side_effect=lambda d: d == 'FD001'):
            runner.run_pipeline()
        for call in self.calls.optimize.call_args_list:
            self.assertEqual(call.args[2], 'FD002')
        for call in self.calls.evaluate.call_args_list:
            self.assertEqual(call.args[3], 'FD002')
        for call in self.calls.compare.call_args_list:
            self.assertEqual(call.args[1], ['FD001', 'FD002'])

    def test_all_completed_and_existing_comparisons_execute_nothing(self):
        with patch.object(runner, 'dataset_complete', return_value=True), \
                patch.object(runner, 'comparison_complete', return_value=True):
            runner.run_pipeline()
        self.assertEqual(self.calls.mock_calls, [])

    def test_changed_artifact_is_not_accepted_as_completed(self):
        folder = runner.MODEL_ROOT
        folder.mkdir()
        (folder / 'model.keras').write_bytes(b'fixture')
        write_json(folder / 'manifest.json', {'protocol': runner.PROTOCOL,
                   'sha256': {'model.keras': runner.file_hash(folder / 'model.keras')}})
        from src.rul_prediction.v2.experiment import read_json
        with patch.object(runner, 'read_json', read_json):
            runner.checked_manifest(folder, ['model.keras'])
            (folder / 'model.keras').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                runner.checked_manifest(folder, ['model.keras'])

    def test_logging_preserves_error_output_and_filters_tensorflow_noise(self):
        import logging
        import os
        import warnings
        levels = {name: logging.getLogger(name).level for name in ('tensorflow', 'absl')}
        try:
            with patch.dict(os.environ), warnings.catch_warnings():
                self.configure_logging()
                self.assertEqual(os.environ['TF_CPP_MIN_LOG_LEVEL'], '2')
                self.assertFalse(logging.getLogger('tensorflow').isEnabledFor(logging.WARNING))
                self.assertTrue(logging.getLogger('tensorflow').isEnabledFor(logging.ERROR))
        finally:
            for name, level in levels.items():
                logging.getLogger(name).setLevel(level)

    def test_invalid_datasets_rejected_before_any_step(self):
        for datasets in ([], ['FD005'], ['FD001', 'FD001']):
            with patch.object(runner, 'DATASETS', datasets), self.assertRaises(ValueError):
                runner.run_pipeline()
        self.assertEqual(self.calls.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
