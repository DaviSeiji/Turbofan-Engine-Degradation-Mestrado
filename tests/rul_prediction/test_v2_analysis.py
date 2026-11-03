"""Fixtures de previsões já salvas; nunca executa redes, Optuna ou pré-processamento."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.rul_prediction.v2 import analyze_results as analysis
from src.rul_prediction.v2.utils import file_hash


def json_file(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream)


def manifest(folder, names, **extra):
    json_file(folder / 'manifest.json', {'protocol': analysis.PROTOCOL, **extra,
              'sha256': {name: file_hash(folder / name) for name in names}})


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results, self.models, self.data = [self.root / label for label in ('results', 'models', 'data')]
        folder = self.data / 'FD001'
        folder.mkdir(parents=True)
        self.preconfig = {'dataset': 'FD001', 'window_size': 30, 'rul_cap': 125}
        json_file(folder / 'config.json', self.preconfig)
        json_file(folder / 'splits.json', {'train': [1], 'validation': [2], 'calibration': [3], 'test': [1, 2]})
        json_file(folder / 'features.json', {'selected': ['s_1'], 'dropped': []})
        self.windows = pd.DataFrame({'Dataset': ['FD001'] * 4, 'SourceSplit': ['test'] * 4,
                 'Split': ['test_windows'] * 4, 'unit_nr': [1, 1, 2, 2], 'start_cycle': [1, 16, 1, 21],
                 'end_cycle': [30, 45, 30, 50], 'RUL_original': [140, 125, 30, 10],
                 'RUL_capped': [125, 125, 30, 10]})
        self.terminal = self.windows.iloc[[1, 3]].copy()
        self.terminal['Split'] = 'test_terminal'
        for protocol, frame in (('windows', self.windows), ('terminal', self.terminal)):
            frame.to_csv(folder / f'test_{protocol}_metadata.csv', index=False)
        names = ['config.json', 'splits.json', 'features.json', 'test_windows_metadata.csv', 'test_terminal_metadata.csv']
        json_file(folder / 'manifest.json', {'pipeline': 'rul_v2', 'schema_version': 1, 'Dataset': 'FD001',
                  'config': self.preconfig, 'sha256': {name: file_hash(folder / name) for name in names}})
        self.data_hash = file_hash(folder / 'manifest.json')
        self.training = {'protocol': analysis.PROTOCOL, 'sampler_seed': 42, 'total_trials': 30,
                         'epochs': 50, 'patience': 5, 'final_epochs': 50, 'final_patience': 5}
        for model in analysis.MODELS:
            for seed in (1, 2):
                self.save_fixture(model, seed)

    def save_fixture(self, model, seed):
        source = self.models / 'FD001' / model / 'baseline' / f'seed_{seed}'
        source.mkdir(parents=True)
        config = {**self.training, 'model': model, 'seed': seed, 'experiment': 'baseline',
                  'training_target': 'y_capped_125',
                  'best': {'trial': 0, 'params': {}, 'validation_mse': 2.},
                  'data': {'Dataset': 'FD001', 'manifest_sha256': self.data_hash,
                           'features': ['s_1'], 'input_shape': [30, 1], 'preprocessing_config': self.preconfig}}
        json_file(source / 'config.json', config)
        json_file(source / 'history.json', {'loss': [2.], 'val_loss': [2.]})
        (source / 'model.keras').write_bytes(b'fixture: NOT a Keras model')
        (source / 'history.png').write_bytes(b'fixture')
        manifest(source, ['model.keras', 'config.json', 'history.json', 'history.png'],
                 data=config['data'], model=model, seed=seed, experiment='baseline')
        for protocol, meta in (('windows', self.windows), ('terminal', self.terminal)):
            folder = self.results / 'FD001' / model / 'baseline' / f'seed_{seed}' / protocol
            folder.mkdir(parents=True)
            frame = meta.copy()
            frame['Model'], frame['Seed'], frame['Experiment'] = model, seed, 'baseline'
            frame['RUL_true'] = frame.RUL_original
            predictions = np.array([120., 120., 35., 5.]) + (seed - 1)
            frame['RUL_pred'] = predictions if protocol == 'windows' else predictions[[1, 3]]
            frame['AbsoluteError'] = np.abs(frame.RUL_pred - frame.RUL_true)
            frame.to_csv(folder / 'predictions.csv', index=False)
            (folder / 'predictions.png').write_bytes(b'fixture')
            record = {'Dataset': 'FD001', 'Model': model, 'Seed': seed, 'Experiment': 'baseline',
                      'Protocol': protocol, 'Target': 'y_original_uncapped', 'N': len(frame),
                      'PreprocessingManifest': self.data_hash, 'ModelSHA256': file_hash(source / 'model.keras'),
                      'TrainingProtocol': self.training, **{k: v for k, v in
                       analysis.scores(frame.RUL_true, frame.RUL_pred).items() if k != 'Bias'}}
            json_file(folder / 'metrics.json', record)
            manifest(folder, ['metrics.json', 'predictions.csv', 'predictions.png'], record=record)

    def reader(self):
        return analysis.SavedResults(self.results, self.models, self.data, 'baseline')

    def test_saved_results_identity_metrics_and_float_tolerance(self):
        frame, config, scores = self.reader().load('FD001', 'cnn_lstm', 1, 'windows')
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame.RUL_true.tolist(), [140, 125, 30, 10])
        self.assertEqual(config['seed'], 1)
        self.assertEqual(scores['Bias'], -6.25)
        self.assertAlmostEqual(scores['RMSE'], np.sqrt(475 / 4))

    def test_band_boundaries_counts_empty_band_and_signed_bias(self):
        self.assertEqual(analysis.rul_band([0, 30, 31, 125, 126]).tolist(),
                         [analysis.BANDS[i] for i in (0, 0, 1, 1, 2)])
        frame, _, _ = self.reader().load('FD001', 'cnn_lstm', 1, 'windows')
        rows = analysis.band_statistics(frame)
        self.assertEqual([r['N'] for r in rows], [2, 1, 1])
        self.assertEqual(sum(r['N'] for r in rows), len(frame))
        self.assertEqual(rows[2]['Bias'], -20.)
        empty = analysis.band_statistics(frame.loc[frame.RUL_true.le(30)])[2]
        self.assertEqual(empty['N'], 0)
        self.assertTrue(np.isnan(empty['MAE']))
        for bad in ([-1], [30.5], [np.nan]):
            with self.assertRaises(ValueError):
                analysis.rul_band(bad)

    def test_cap_share_and_complement_do_not_clip_predictions(self):
        frame, _, _ = self.reader().load('FD001', 'cnn_lstm', 1, 'windows')
        contribution = analysis.cap_contribution(frame)
        self.assertEqual(contribution['NHigh'], 1)
        self.assertAlmostEqual(contribution['SSEHighShare'], 400 / 475)
        self.assertEqual(analysis.scores(np.minimum(frame.RUL_true, 125), frame.RUL_pred)['RMSE'], 5.)
        self.assertEqual(analysis.scores([140], [150])['Bias'], 10.)
        self.assertEqual(analysis.scores([125], [150])['RMSE'], 25.)

    def test_sample_standard_deviation_not_pooled_rmse(self):
        frame = pd.DataFrame({'Dataset': ['FD001'] * 2, 'Seed': [1, 2], 'RMSE': [2., 4.]})
        result = analysis.aggregate(frame, ['Dataset'], ['RMSE']).iloc[0]
        self.assertEqual(result.RMSE_mean, 3.)
        self.assertAlmostEqual(result.RMSE_std, np.sqrt(2))
        self.assertEqual(result.NSeeds, 2)

    def test_duplicate_alignment_and_prediction_dispersion(self):
        first, _, _ = self.reader().load('FD001', 'cnn_lstm', 1, 'windows')
        second, _, _ = self.reader().load('FD001', 'cnn_lstm', 2, 'windows')
        frame = pd.concat([first, second]).sample(frac=1, random_state=42)
        frame['Protocol'] = 'windows'
        result = analysis.ensemble(frame)
        self.assertEqual(result.end_cycle.tolist(), [30, 45, 30, 50])
        np.testing.assert_allclose(result.PredictionSD, np.sqrt(.5))
        with self.assertRaises(ValueError):
            analysis.ensemble(pd.concat([frame, frame.iloc[[0]]]))

    def test_engine_selection_independent_of_errors(self):
        self.assertEqual(analysis.select_engines([100, 2, 50, 1]), [1, 2, 100])
        self.assertEqual(analysis.select_engines(range(1, 101)), [1, 50, 100])

    def test_hash_corruption_rejected(self):
        path = self.results / 'FD001/cnn_lstm/baseline/seed_1/windows/predictions.csv'
        with path.open('a') as stream:
            stream.write('\ncorrupt')
        with self.assertRaisesRegex(ValueError, 'Hash'):
            self.reader().load('FD001', 'cnn_lstm', 1, 'windows')

    def test_rehashed_wrong_seed_and_metrics_rejected(self):
        folder = self.results / 'FD001/cnn_lstm/baseline/seed_1/windows'
        saved = json.loads((folder / 'metrics.json').read_text())
        for key, value in (('Seed', 9), ('MAE', 999.)):
            record = {**saved, key: value}
            (folder / 'metrics.json').unlink()
            json_file(folder / 'metrics.json', record)
            (folder / 'manifest.json').unlink()
            manifest(folder, ['metrics.json', 'predictions.csv', 'predictions.png'], record=record)
            with self.assertRaises(ValueError):
                self.reader().load('FD001', 'cnn_lstm', 1, 'windows')

    def test_complete_analysis_read_only_and_overwrite_protection(self):
        originals = {str(p): file_hash(p) for root in (self.models, self.data, self.results)
                     for p in root.rglob('*') if p.is_file()}
        with patch.dict('sys.modules', {'tensorflow': None, 'keras': None, 'optuna': None}), \
                patch.object(analysis, 'make_figures'):
            summary = analysis.analyze(self.results, self.models, self.data,
                                       ['FD001'], [1, 2], name='synthetic')
        self.assertEqual(summary['validated_evaluations'], 8)
        self.assertEqual(summary['issues'], [])
        for path, digest in originals.items():
            self.assertEqual(file_hash(path), digest)
        with self.assertRaises(FileExistsError):
            analysis.analyze(self.results, self.models, self.data, ['FD001'], [1, 2], name='synthetic')
        bands = pd.read_csv(self.results / 'analysis/synthetic/tables/bands_individual.csv')
        self.assertEqual(bands.N.sum(), (4 + 2) * 2 * 2)
        for bad in ('../baseline', '/tmp/bad'):
            with self.assertRaises(ValueError):
                analysis.analyze(self.results, self.models, self.data, ['FD001'], [1], name=bad)

    def test_missing_results_recorded_not_invented(self):
        with patch.object(analysis, 'make_figures'):
            summary = analysis.analyze(self.results, self.models, self.data,
                                       ['FD001'], [1, 2, 3], name='missing')
        self.assertEqual(summary['validated_evaluations'], 8)
        self.assertEqual(len(summary['issues']), 4)
        self.assertEqual(summary['status'], 'partial')
        table = pd.read_csv(self.results / 'analysis/missing/tables/terminal_summary.csv')
        self.assertTrue(table.NSeeds.eq(2).all())


if __name__ == '__main__':
    unittest.main()
