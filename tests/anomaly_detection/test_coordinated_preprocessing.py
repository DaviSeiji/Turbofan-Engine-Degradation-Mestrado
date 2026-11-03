"""Somente fixtures temporárias; sem TensorFlow, Optuna ou dados C-MAPSS reais."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.common.paths import ROOT
from src.anomaly_detection.v2 import preprocessing as prep
from src.anomaly_detection.v2.utils import (
    Config, CoordinatedNominalPreprocessor, OUTPUT_DIR, load_rul_reference, output_target,
    read_json, split_inside_rul_train, validate_rul_splits, write_json,
)
from src.rul_prediction.v2.preprocessing import add_rul, make_windows
from src.rul_prediction.v2.utils import (
    Config as RULConfig, INPUT_COLUMNS, RAW_COLUMNS, RULPreprocessor, SETTINGS, file_hash, split_engines,
)
from tests.rul_prediction.test_v2_preprocessing import synthetic


def fixture(base, dataset='FD001', cycles=120):
    raw_dir, rul_dir = base / 'raw', base / 'rul'
    raw_dir.mkdir()
    folder = rul_dir / dataset
    folder.mkdir(parents=True)
    raw = synthetic(cycles=cycles, multi=dataset in ('FD002', 'FD004'))
    path = raw_dir / f'train_{dataset}.txt'
    raw.to_csv(path, sep=' ', index=False, header=False)
    cfg = RULConfig(dataset)
    ids = split_engines(range(1, 21), cfg)
    ids['test'] = [1, 2]  # Mesmo número, outra origem física.
    write_json(folder / 'config.json', cfg.to_dict())
    write_json(folder / 'splits.json', ids)
    write_json(folder / 'manifest.json', {'pipeline': 'rul_v2', 'schema_version': 1, 'Dataset': dataset,
               'config': cfg.to_dict(), 'input_sha256': {path.name: file_hash(path)},
               'sha256': {name: file_hash(folder / name) for name in ('config.json', 'splits.json')}})
    return raw, raw_dir, rul_dir, ids


def rul_metadata(raw, dataset, split):
    if split.startswith('test_'):
        data = add_rul(raw, 'test', final_rul={int(u): 7 for u in raw.unit_nr.unique()})
    else:
        data = add_rul(raw, 'train')
    data['regime'] = 0
    return make_windows(data, ['s_1'], dataset, split, terminal=split == 'test_terminal')[1]


class CoordinatedPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.raw, self.raw_dir, self.rul_dir, self.ids = fixture(self.base)
        self.config = Config('FD001')
        self.reference = load_rul_reference('FD001', self.rul_dir)[1]

    def prepared(self, raw=None, config=None, ids=None, reference=None):
        return prep.prepare_frames(self.raw if raw is None else raw, config or self.config,
                                   ids or self.ids, reference or self.reference)

    def test_inherits_saved_splits_and_inner_80_20_only_rul_train(self):
        ids, reference = load_rul_reference('FD001', self.rul_dir)
        self.assertEqual(ids, self.ids)
        self.assertEqual(reference['splits_sha256'], file_hash(self.rul_dir / 'FD001/splits.json'))
        splits = split_inside_rul_train(ids, self.config)
        self.assertEqual(set(splits['ae_fit']) | set(splits['ae_inner_validation']), set(ids['train']))
        self.assertTrue(set(splits['ae_fit']).isdisjoint(splits['ae_inner_validation']))
        for role in ('validation', 'calibration'):
            self.assertFalse(set(splits['ae_fit']) & set(ids[role]))
            self.assertFalse(set(splits['ae_inner_validation']) & set(ids[role]))
        big = {'train': list(range(1, 71)), 'validation': list(range(71, 86)),
               'calibration': list(range(86, 101)), 'test': [1, 2]}
        big_split = split_inside_rul_train(big, self.config)
        self.assertEqual([len(big_split[r]) for r in prep.ROLES], [56, 14])

    def test_scaler_and_feature_selection_use_ae_fit_nominal_only(self):
        proc, splits, units, _, _ = self.prepared()
        fit = self.raw[self.raw.unit_nr.isin(splits['ae_fit']) & self.raw.time_cycles.le(36)]
        scaler = proc.scalers_[0]
        np.testing.assert_allclose(scaler.data_min_, fit[proc.features_].min())
        np.testing.assert_allclose(scaler.data_max_, fit[proc.features_].max())
        self.assertNotIn('s_2', proc.features_)
        self.assertFalse(scaler.clip)
        modified = self.raw.copy()
        external = ~modified.unit_nr.isin(splits['ae_fit']) | modified.time_cycles.gt(36)
        modified.loc[external, INPUT_COLUMNS] += 10000
        changed, _, _, changed_arrays, _ = self.prepared(raw=modified)
        self.assertEqual(changed.features_, proc.features_)
        np.testing.assert_array_equal(changed.scalers_[0].data_min_, scaler.data_min_)
        np.testing.assert_array_equal(changed.scalers_[0].data_max_, scaler.data_max_)
        self.assertTrue((changed_arrays['ae_inner_validation']['X'] > 1).any())
        self.assertEqual(proc.train_units_, splits['ae_fit'])
        self.assertEqual(set(units.unit_nr), set(self.ids['train']))

    def test_kmeans_six_regimes_fit_only_nominal_ae_fit(self):
        for dataset in ('FD002', 'FD004'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                raw, _, rul, ids = fixture(Path(temp), dataset)
                config = Config(dataset)
                reference = load_rul_reference(dataset, rul)[1]
                actual = []
                original = KMeans.fit

                def capture(model, X, *args, **kwargs):
                    actual.append(X.copy())
                    return original(model, X, *args, **kwargs)

                with patch.object(KMeans, 'fit', capture):
                    proc, splits, _, _, _ = prep.prepare_frames(raw, config, ids, reference)
                expected = raw.loc[raw.unit_nr.isin(splits['ae_fit']) & raw.time_cycles.le(36), SETTINGS]
                self.assertEqual(len(actual), 1)
                np.testing.assert_array_equal(actual[0].to_numpy(), expected.to_numpy())
                self.assertEqual(len(proc.scalers_), 6)
                self.assertEqual(set(proc.kmeans_.labels_), set(range(6)))

    def test_single_regime_never_fits_kmeans(self):
        for dataset in ('FD001', 'FD003'):
            with self.subTest(dataset=dataset), patch.object(KMeans, 'fit', side_effect=AssertionError('KMeans indevido')):
                proc = self.prepared(config=Config(dataset), reference={**self.reference, 'Dataset': dataset})[0]
                self.assertIsNone(proc.kmeans_)

    def test_default_io_never_accesses_test_reserve_or_rul_arrays(self):
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.name.startswith(('test_', 'calibration', 'validation_', 'RUL_FD')) or path.suffix == '.npz':
                if args and 'x' in args[0]:
                    return original(path, *args, **kwargs)
                # Hashes de saída próprios .npz são permitidos, nunca os da referência.
                if path.is_relative_to(self.base / 'out'):
                    return original(path, *args, **kwargs)
                raise AssertionError(f'Acesso indevido: {path}')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded):
            folder = prep.process_dataset(self.config, self.raw_dir, self.rul_dir, self.base / 'out')
        self.assertTrue((folder / 'manifest.json').is_file())
        self.assertFalse((self.raw_dir / 'test_FD001.txt').exists())

    def test_fit_rejects_inner_validation_external_engines_and_late_cycles(self):
        proc, splits, _, _, _ = self.prepared()
        nominal = self.raw.loc[self.raw.unit_nr.isin(splits['ae_fit']) & self.raw.time_cycles.le(36)]
        for bad in (self.raw[self.raw.unit_nr.isin(splits['ae_inner_validation'])],
                    self.raw[self.raw.unit_nr.isin(splits['ae_fit'])],
                    nominal.assign(SourceSplit='test')):
            fresh = CoordinatedNominalPreprocessor(self.config, splits, self.reference, proc.fit_limits)
            with self.assertRaises(ValueError):
                fresh.fit(bad)
        with self.assertRaises(RuntimeError):
            proc.fit(nominal)

    def test_no_rul_targets_or_ids_in_input_npz(self):
        raw = self.raw.assign(RUL=np.inf, RUL_original=-123, RUL_capped=999)
        proc, _, _, arrays, metadata = self.prepared(raw=raw)
        plain = self.prepared()[3]
        for role in prep.ROLES:
            self.assertEqual(set(arrays[role]), {'X', 'features'})
            self.assertEqual(arrays[role]['X'].shape[1:], (30, len(proc.features_)))
            self.assertFalse(set(proc.features_) - set(INPUT_COLUMNS))
            self.assertTrue(metadata[role].Split.eq('train').all())
            self.assertTrue(metadata[role].AERole.eq(role).all())
            self.assertFalse(metadata[role].is_padded.any())
            np.testing.assert_array_equal(arrays[role]['X'], plain[role]['X'])

    def test_insufficient_nominal_regions_are_reported_without_replacement(self):
        splits = split_inside_rul_train(self.ids, self.config)
        short = splits['ae_fit'][0]
        raw = self.raw.loc[~self.raw.unit_nr.eq(short) | self.raw.time_cycles.le(60)].copy()
        proc, saved, units, _, meta = self.prepared(raw=raw)
        row = units.loc[units.unit_nr.eq(short)].iloc[0]
        self.assertFalse(row.eligible)
        self.assertEqual(row.nominal_cycles, 18)
        self.assertEqual(row.nominal_windows, 0)
        self.assertIn(short, saved['ae_fit'])
        self.assertNotIn(short, proc.train_units_)
        self.assertNotIn(short, meta['ae_fit'].unit_nr.tolist())
        with self.assertRaisesRegex(ValueError, 'nenhum motor'):
            self.prepared(raw=self.raw.loc[self.raw.time_cycles.le(60)])

    def test_roundtrip_reload_without_any_fit_and_configuration_checks(self):
        original_splits = file_hash(self.rul_dir / 'FD001/splits.json')
        folder = prep.process_dataset(self.config, self.raw_dir, self.rul_dir, self.base / 'out')
        with patch.object(RULPreprocessor, 'fit', side_effect=AssertionError('Novo fit proibido')), \
                patch.object(MinMaxScaler, 'fit', side_effect=AssertionError('Novo fit proibido')):
            processor, summary = prep.validate_saved_dataset(folder, self.rul_dir)
            same = prep.load_preprocessor(folder, self.rul_dir)
        self.assertEqual(processor.features_, same.features_)
        self.assertEqual(summary['ae_fit']['assigned_engines'], 11)
        self.assertEqual(file_hash(self.rul_dir / 'FD001/splits.json'), original_splits)
        with self.assertRaises(ValueError):
            prep.validate_saved_dataset(folder, self.rul_dir, {'seed': 17})

    def test_fd002_fd004_short_motor_left_padding_matches_rul_keys(self):
        for dataset in ('FD002', 'FD004'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                raw, _, rul, ids = fixture(Path(temp), dataset)
                proc = prep.prepare_frames(raw, Config(dataset), ids, load_rul_reference(dataset, rul)[1])[0]
                test = pd.concat([synthetic([1], 12, multi=True), synthetic([2], 19, multi=True)])
                meta = rul_metadata(test, dataset, 'test_terminal')
                test = test.assign(Dataset=dataset, SourceSplit='test')
                arrays, aligned = prep.prepare_aligned_windows(test, meta, proc, allow_test=True)
                self.assertEqual(arrays['X'].shape, (2, 30, len(proc.features_)))
                self.assertEqual(aligned.padding_cycles.tolist(), [18, 11])
                self.assertTrue(aligned[prep.PHYSICAL_KEYS].equals(meta[prep.PHYSICAL_KEYS]))
                self.assertEqual(aligned.start_cycle.tolist(), [1, 1])
                self.assertEqual(aligned.observed_cycles.tolist(), [12, 19])
                first = proc.transform(test.loc[test.unit_nr.eq(1), RAW_COLUMNS])[proc.features_].to_numpy(np.float32)
                np.testing.assert_array_equal(arrays['X'][0, :18], np.repeat(first[:1], 18, axis=0))
                np.testing.assert_array_equal(arrays['X'][0, 18:], first)

    def test_test_and_calibration_require_explicit_permission(self):
        proc = self.prepared()[0]
        for split, unit in (('test_terminal', 1), ('calibration', self.ids['calibration'][0])):
            raw = synthetic([unit], 35)
            meta = rul_metadata(raw, 'FD001', split)
            tagged = raw.assign(Dataset='FD001', SourceSplit='test' if split.startswith('test') else 'train')
            with patch.object(proc, 'transform') as transform, self.assertRaises(ValueError):
                prep.prepare_aligned_windows(tagged, meta, proc)
            transform.assert_not_called()

    def test_temporal_prefix_invariance_no_future_cycles_used(self):
        proc = self.prepared()[0]
        raw = synthetic([1], 100).assign(Dataset='FD001', SourceSplit='test')
        meta = rul_metadata(raw, 'FD001', 'test_windows').iloc[:2]
        first, aligned = prep.prepare_aligned_windows(raw, meta, proc, allow_test=True)
        changed = raw.copy()
        changed.loc[changed.time_cycles.gt(31), INPUT_COLUMNS] = np.nan
        second, _ = prep.prepare_aligned_windows(changed, meta, proc, allow_test=True)
        prefix, _ = prep.prepare_aligned_windows(raw.loc[raw.time_cycles.le(31)], meta, proc, allow_test=True)
        np.testing.assert_array_equal(first['X'], second['X'])
        np.testing.assert_array_equal(first['X'], prefix['X'])
        self.assertTrue(aligned.end_cycle.equals(meta.end_cycle.reset_index(drop=True)))

    def test_source_namespaces_and_duplicate_or_invalid_metadata_rejected(self):
        proc = self.prepared()[0]
        raw = synthetic([1], 35).assign(Dataset='FD001', SourceSplit='test')
        meta = rul_metadata(raw, 'FD001', 'test_windows')
        with self.assertRaisesRegex(ValueError, 'origem'):
            prep.prepare_aligned_windows(raw.assign(SourceSplit='train'), meta, proc, allow_test=True)
        with self.assertRaises(ValueError):
            prep.prepare_aligned_windows(raw, pd.concat([meta, meta.iloc[:1]]), proc, allow_test=True)
        changed = meta.copy()
        changed.loc[0, 'observed_cycles'] = 29
        with self.assertRaises(ValueError):
            prep.prepare_aligned_windows(raw, changed, proc, allow_test=True)
        with self.assertRaises(ValueError):
            self.prepared(raw=self.raw.assign(SourceSplit='test'))

    def test_feature_order_and_metadata_row_order_preserved(self):
        proc = self.prepared()[0]
        raw = synthetic([1], 35).assign(Dataset='FD001', SourceSplit='test')
        meta = rul_metadata(raw, 'FD001', 'test_windows')
        expected, _ = prep.prepare_aligned_windows(raw, meta, proc, allow_test=True)
        actual, aligned = prep.prepare_aligned_windows(raw[raw.columns[::-1]], meta.iloc[::-1], proc, allow_test=True)
        np.testing.assert_array_equal(expected['X'][::-1], actual['X'])
        self.assertEqual(aligned.end_cycle.tolist(), meta.iloc[::-1].end_cycle.tolist())
        proc.features_.reverse()
        with self.assertRaises(ValueError):
            prep.prepare_aligned_windows(raw, meta, proc, allow_test=True)

    def test_reference_hash_and_dataset_mismatches_refused(self):
        with patch('src.anomaly_detection.v2.utils.file_hash', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Hash'):
                load_rul_reference('FD001', self.rul_dir)
        with self.assertRaises(ValueError):
            self.prepared(config=Config('FD003'))
        ids = {**self.ids, 'validation': self.ids['validation'] + self.ids['train'][:1]}
        with self.assertRaises(ValueError):
            validate_rul_splits(ids)

    def test_saved_hash_corruption_and_partial_folders_refused(self):
        out = self.base / 'out'
        folder = prep.process_dataset(self.config, self.raw_dir, self.rul_dir, out)
        with (folder / 'features.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'Hash'):
            prep.validate_saved_dataset(folder, self.rul_dir)
        with self.assertRaises(FileExistsError):
            prep.process_dataset(self.config, self.raw_dir, self.rul_dir, out)
        partial = self.base / 'partial/FD001'
        partial.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            prep.validate_saved_dataset(partial, self.rul_dir)
        with self.assertRaises(FileExistsError):
            prep.process_dataset(self.config, self.raw_dir, self.rul_dir, partial.parent)

    def test_raw_identity_mismatch_stops_before_fit(self):
        with (self.raw_dir / 'train_FD001.txt').open('a') as stream:
            stream.write('\n')
        with patch.object(CoordinatedNominalPreprocessor, 'fit') as fit, self.assertRaises(ValueError):
            prep.process_dataset(self.config, self.raw_dir, self.rul_dir, self.base / 'out')
        fit.assert_not_called()
        self.assertFalse((self.base / 'out').exists())

    def test_multiregime_reload_and_validate_never_read_raw_or_refit(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            _, raw_dir, rul_dir, _ = fixture(base, 'FD002')
            folder = prep.process_dataset(Config('FD002'), raw_dir, rul_dir, base / 'out')
            original = pd.read_csv

            def guarded(path, *args, **kwargs):
                if Path(path).is_relative_to(raw_dir):
                    raise AssertionError('Validação não pode ler bruto.')
                return original(path, *args, **kwargs)

            with patch.object(pd, 'read_csv', guarded), patch.object(KMeans, 'fit', side_effect=AssertionError('Novo fit')), \
                    patch.object(MinMaxScaler, 'fit', side_effect=AssertionError('Novo fit')):
                processor, _ = prep.validate_saved_dataset(folder, rul_dir)
            self.assertEqual(len(processor.scalers_), 6)

    def test_mid_write_failure_never_creates_completion_manifest_or_overwrites(self):
        with patch.object(np, 'savez_compressed', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError):
                prep.process_dataset(self.config, self.raw_dir, self.rul_dir, self.base / 'out')
        folder = self.base / 'out/FD001'
        self.assertTrue(folder.exists())
        self.assertFalse((folder / 'manifest.json').exists())
        with self.assertRaises(FileNotFoundError):
            prep.validate_saved_dataset(folder, self.rul_dir)
        with self.assertRaises(FileExistsError):
            prep.process_dataset(self.config, self.raw_dir, self.rul_dir, self.base / 'out')

    def test_historical_paths_protected_before_reading_or_fitting(self):
        for path in (ROOT, ROOT / 'data/processed_anomaly', ROOT / 'data/processed_rul_v2',
                     ROOT / 'models', ROOT / 'results', self.raw_dir, self.rul_dir):
            with self.subTest(path=path), self.assertRaises(ValueError):
                output_target(path, self.raw_dir, self.rul_dir, 'FD001')
        # Independe de o usuário já ter preparado FD001 na pasta real.
        with patch.object(Path, 'exists', return_value=False):
            self.assertEqual(output_target(OUTPUT_DIR, self.raw_dir, self.rul_dir, 'FD001'), OUTPUT_DIR / 'FD001')

    def test_same_seed_same_arrays_and_different_seed_different_inner_ids(self):
        first = self.prepared()
        second = self.prepared(raw=self.raw.sample(frac=1, random_state=9))
        self.assertEqual(first[1], second[1])
        for role in prep.ROLES:
            np.testing.assert_array_equal(first[3][role]['X'], second[3][role]['X'])
            self.assertTrue(first[4][role].equals(second[4][role]))
        other = split_inside_rul_train(self.ids, Config('FD001', seed=17))
        self.assertNotEqual(first[1]['ae_fit'], other['ae_fit'])

    def test_invalid_configuration(self):
        for options in ({'seed': -1}, {'seed': True}, {'nominal_fraction': 0},
                        {'nominal_fraction': np.nan}, {'inner_validation_fraction': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config('FD001', **options)

    def test_cli_help_and_synthetic_roundtrip_without_tf_or_optuna(self):
        code = ("import sys,runpy; sys.modules['tensorflow']=None; sys.modules['keras']=None; "
                "sys.modules['optuna']=None; sys.argv=['prep', *ARGS]; "
                "runpy.run_module('src.anomaly_detection.v2.preprocessing',run_name='__main__')")
        common = ['--datasets', 'FD001', '--raw-dir', str(self.raw_dir),
                  '--rul-dir', str(self.rul_dir), '--output-dir', str(self.base / 'cli')]
        cases = [(['--help'], True), (common, True),
                 (common + ['--validate-only', '--raw-dir', str(self.base / 'absent_raw')], True),
                 (common + ['--validate-only', '--seed', '17'], False), (common, False)]
        for args, success in cases:
            result = subprocess.run([sys.executable, '-B', '-c', code.replace('ARGS', repr(args))],
                                    cwd=ROOT, capture_output=True, text=True, timeout=40)
            if success:
                self.assertEqual(result.returncode, 0, result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.base / 'cli/FD001/manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
