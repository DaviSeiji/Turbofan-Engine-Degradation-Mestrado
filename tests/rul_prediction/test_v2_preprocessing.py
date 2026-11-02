"""RUL v2 com fixtures pequenas; nenhum dataset ou modelo real é processado."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.common.paths import ROOT
from src.rul_prediction.v2.preprocessing import (
    META_COLUMNS, add_rul, load_preprocessor, make_windows, prepare_frames,
    process_dataset, read_sources, validate_saved_dataset,
)
from src.rul_prediction.v2.utils import (
    ARRAY_SPLITS, Config, DEVELOPMENT_SPLITS, INPUT_COLUMNS, RAW_COLUMNS,
    RULPreprocessor, SETTINGS, file_hash, output_target, split_engines,
    validate_trajectories,
)


def synthetic(units=range(1, 21), cycles=34, multi=False):
    rows = []
    for unit in units:
        for cycle in range(1, cycles + 1):
            regime = cycle % 6 if multi else 0
            row = {'unit_nr': unit, 'time_cycles': cycle,
                   'setting_1': regime * 20. + unit * .001 + cycle * .0001,
                   'setting_2': regime * .1 + cycle * .00001,
                   'setting_3': regime * 10.}
            row.update({f's_{i}': i + unit * .1 + cycle * .01 + regime for i in range(1, 22)})
            row['s_2'] = 7.  # Constante de treino que pode variar nos holdouts.
            rows.append(row)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def synthetic_sources(raw_dir, dataset='FD001'):
    raw_dir.mkdir()
    multi = dataset in ('FD002', 'FD004')
    train = synthetic(multi=multi)
    test = pd.concat([synthetic([1], 35, multi), synthetic([2], 19, multi)], ignore_index=True)
    train.to_csv(raw_dir / f'train_{dataset}.txt', sep=' ', header=False, index=False)
    test.to_csv(raw_dir / f'test_{dataset}.txt', sep=' ', header=False, index=False)
    pd.DataFrame([140, 7]).to_csv(raw_dir / f'RUL_{dataset}.txt', header=False, index=False)
    return train, test


class V2PreprocessingTests(unittest.TestCase):
    def test_engine_split_disjoint_complete_reproducible_and_configurable(self):
        config = Config('FD001')
        first = split_engines(range(1, 101), config)
        self.assertEqual([len(first[s]) for s in DEVELOPMENT_SPLITS], [70, 15, 15])
        self.assertEqual(first, split_engines(reversed(range(1, 101)), config))
        sets = [set(first[s]) for s in DEVELOPMENT_SPLITS]
        self.assertEqual(set.union(*sets), set(range(1, 101)))
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                self.assertFalse(a & b)
        other = split_engines(range(1, 21), Config('FD001', train_fraction=.6,
                              validation_fraction=.2, calibration_fraction=.2))
        self.assertEqual([len(other[s]) for s in DEVELOPMENT_SPLITS], [12, 4, 4])
        self.assertNotEqual(first, split_engines(range(1, 101), Config('FD001', seed=7)))

    def test_invalid_config_and_too_few_engines(self):
        for kwargs in [{'dataset': 'FD005'}, {'dataset': 'FD001', 'seed': -1},
                       {'dataset': 'FD001', 'train_fraction': .8},
                       {'dataset': 'FD001', 'calibration_fraction': 0.},
                       {'dataset': 'FD001', 'rul_cap': 0},
                       {'dataset': 'FD001', 'constant_tolerance': float('nan')}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with self.assertRaises(ValueError):
            split_engines([1, 2], Config('FD001'))
        with self.assertRaises(ValueError):
            split_engines([1, 1, 2], Config('FD001'))

    def test_feature_selection_and_scaler_use_train_only_without_clipping(self):
        train = synthetic([1, 2], 34)
        processor = RULPreprocessor(Config('FD001')).fit(train)
        self.assertNotIn('s_2', processor.features_)
        self.assertEqual(processor.features_, [c for c in INPUT_COLUMNS if c not in ['s_2', 'setting_3']])
        scaler = processor.scalers_[0]
        np.testing.assert_allclose(scaler.data_min_, train[processor.features_].min())
        np.testing.assert_allclose(scaler.data_max_, train[processor.features_].max())
        self.assertEqual(scaler.n_samples_seen_, len(train))
        before = scaler.data_max_.copy()
        for unit in [3, 4, 5]:  # Validação, reserva e teste com valores novos.
            heldout = synthetic([unit], 34)
            heldout['s_1'] = 1000.
            heldout['s_2'] = 2000.
            normalized = processor.transform(heldout)
            self.assertTrue((normalized.s_1 > 1).all())
        np.testing.assert_array_equal(before, scaler.data_max_)
        self.assertEqual(scaler.n_samples_seen_, len(train))
        with self.assertRaises(RuntimeError):
            processor.fit(heldout)

    def test_single_condition_datasets_never_construct_kmeans(self):
        for dataset in ['FD001', 'FD003']:
            with self.subTest(dataset=dataset), patch(
                    'src.rul_prediction.v2.utils.KMeans', side_effect=AssertionError('KMeans indevido')):
                processor = RULPreprocessor(Config(dataset)).fit(synthetic())
                self.assertIsNone(processor.kmeans_)
                self.assertEqual(set(processor.transform(synthetic([31])).regime), {0})

    def test_multi_condition_kmeans_and_scalers_fit_only_training_rows(self):
        original_fit = KMeans.fit
        for dataset in ['FD002', 'FD004']:
            with self.subTest(dataset=dataset):
                train = synthetic(multi=True)
                seen = []

                def record_fit(estimator, X, *args, **kwargs):
                    seen.append(X.copy())
                    return original_fit(estimator, X, *args, **kwargs)

                with patch.object(KMeans, 'fit', record_fit):
                    processor = RULPreprocessor(Config(dataset)).fit(train)
                    centers = processor.kmeans_.cluster_centers_.copy()
                    for unit in [30, 31, 32]:
                        processor.transform(synthetic([unit], multi=True))
                self.assertEqual(len(seen), 1)
                pd.testing.assert_frame_equal(seen[0], train[SETTINGS])
                self.assertEqual(len(processor.scalers_), 6)
                self.assertEqual(sum(int(s.n_samples_seen_) for s in processor.scalers_.values()), len(train))
                np.testing.assert_array_equal(centers, processor.kmeans_.cluster_centers_)
                self.assertEqual(len(np.unique(processor.kmeans_.labels_)), 6)
        with self.assertRaises(ValueError):
            RULPreprocessor(Config('FD002')).fit(synthetic([1], cycles=3, multi=True))

    def test_feature_order_is_independent_of_input_column_order(self):
        data = synthetic()
        processor = RULPreprocessor(Config('FD001')).fit(data)
        expected = processor.transform(data)
        actual = processor.transform(data[data.columns[::-1]])
        np.testing.assert_array_equal(expected[processor.features_], actual[processor.features_])
        with self.assertRaises(ValueError):
            processor.transform(data.drop(columns='s_1'))

    def test_original_rul_and_cap_are_correct(self):
        train = add_rul(synthetic([1], 170), 'train')
        self.assertEqual(train.RUL_original.iloc[0], 169)
        self.assertEqual(train.RUL_capped.iloc[0], 125)
        self.assertEqual(train.RUL_original.iloc[-1], 0)
        test = add_rul(synthetic([1], 35), 'test', final_rul={1: 140})
        self.assertEqual(test.RUL_original.iloc[0], 174)
        self.assertEqual(test.RUL_original.iloc[-1], 140)
        self.assertTrue(test.RUL_capped.eq(125).all())
        with self.assertRaises(ValueError):
            add_rul(synthetic([1]), 'test', final_rul={2: 7})
        with self.assertRaises(ValueError):
            add_rul(synthetic([1]), 'test', final_rul={1: -2})

    def test_window_alignment_motor_boundaries_and_padding(self):
        raw = pd.concat([synthetic([1], 35), synthetic([2], 19)], ignore_index=True)
        data = add_rul(raw, 'test', final_rul={1: 140, 2: 7})
        processor = RULPreprocessor(Config('FD001')).fit(synthetic())
        transformed = processor.transform(data)
        arrays, meta = make_windows(transformed, processor.features_, 'FD001', 'test_windows')
        self.assertEqual(arrays['X'].shape, (7, 30, len(processor.features_)))
        self.assertEqual(list(meta.columns), META_COLUMNS)
        self.assertEqual(meta[meta.unit_nr == 1].end_cycle.tolist(), list(range(30, 36)))
        for i, row in meta.iterrows():
            observed = transformed[(transformed.unit_nr == row.unit_nr) &
                                   transformed.time_cycles.between(row.start_cycle, row.end_cycle)]
            expected = observed[processor.features_].to_numpy(dtype=np.float32)
            np.testing.assert_array_equal(arrays['X'][i, -len(expected):], expected)
            self.assertEqual(arrays['y_original'][i], observed.RUL_original.iloc[-1])
            self.assertEqual(arrays['y'][i], min(arrays['y_original'][i], 125))
        short = meta[meta.unit_nr == 2].iloc[0]
        self.assertTrue(short.is_padded)
        self.assertEqual((short.start_cycle, short.end_cycle, short.padding_cycles), (1, 19, 11))
        index = meta.index[meta.unit_nr == 2][0]
        np.testing.assert_array_equal(arrays['X'][index, :11], np.repeat(arrays['X'][index, 11:12], 11, axis=0))

    def test_terminal_has_one_last_window_per_official_test_motor(self):
        test = add_rul(pd.concat([synthetic([1], 35), synthetic([2], 19)]),
                       'test', final_rul={1: 140, 2: 7})
        processor = RULPreprocessor(Config('FD001')).fit(synthetic())
        transformed = processor.transform(test)
        all_arrays, all_meta = make_windows(transformed, processor.features_, 'FD001', 'test_windows')
        terminal_arrays, terminal_meta = make_windows(transformed, processor.features_, 'FD001', 'test_terminal', terminal=True)
        self.assertEqual(terminal_meta.unit_nr.tolist(), [1, 2])
        self.assertEqual(terminal_meta.end_cycle.tolist(), [35, 19])
        self.assertEqual(terminal_arrays['y_original'].tolist(), [140, 7])
        last_indices = all_meta.groupby('unit_nr').tail(1).index.to_numpy()
        np.testing.assert_array_equal(terminal_arrays['X'], all_arrays['X'][last_indices])

    def test_prepare_frames_is_reproducible_and_test_ids_are_separate_namespace(self):
        train = add_rul(synthetic(), 'train')
        test = add_rul(synthetic([1, 2]), 'test', final_rul={1: 8, 2: 9})
        config = Config('FD001')
        p1, ids1, arrays1 = prepare_frames(train, test, config)
        p2, ids2, arrays2 = prepare_frames(train.sample(frac=1, random_state=5), test, config)
        self.assertEqual(ids1, ids2)
        self.assertEqual(p1.train_units_, ids1['train'])
        for split in ARRAY_SPLITS:
            for key in ['X', 'y', 'y_original', 'features']:
                np.testing.assert_array_equal(arrays1[split][0][key], arrays2[split][0][key])
            pd.testing.assert_frame_equal(arrays1[split][1], arrays2[split][1])
        for split in DEVELOPMENT_SPLITS:
            self.assertEqual(set(arrays1[split][1].unit_nr), set(ids1[split]))
            self.assertTrue(arrays1[split][1].SourceSplit.eq('train').all())
        self.assertTrue(arrays1['test_terminal'][1].SourceSplit.eq('test').all())
        self.assertEqual(ids1['test'], [1, 2])  # Não é vazamento ter IDs numéricos coincidentes.

    def test_invalid_trajectories_are_rejected_without_dropping_rows(self):
        good = synthetic([1])
        broken = [good.drop(index=4), pd.concat([good, good.iloc[:1]]),
                  good.assign(s_1=np.nan), good.assign(time_cycles=good.time_cycles + 1)]
        for data in broken:
            with self.assertRaises(ValueError):
                validate_trajectories(data)
        ordered = validate_trajectories(good.iloc[::-1])
        self.assertEqual(ordered.time_cycles.tolist(), list(range(1, 35)))

    def test_synthetic_io_roundtrip_no_refit_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='rul_v2_test_') as directory:
            root = Path(directory)
            synthetic_sources(root / 'raw')
            historical = root / 'processed_data'
            historical.mkdir()
            sentinel = historical / 'historical.npz'
            sentinel.write_bytes(b'arquivo historico intocado')
            legacy_hash = file_hash(ROOT / 'src/common/preprocessing/preproc_data_nasa.py')
            target = process_dataset(Config('FD001'), root / 'raw', root / 'v2')
            with patch.object(RULPreprocessor, 'fit', side_effect=AssertionError('Reajuste indevido')):
                processor, summary = validate_saved_dataset(target)
                loaded = load_preprocessor(target)
            self.assertEqual(processor.features_, loaded.features_)
            self.assertEqual(summary['test_terminal'], {'engines': 2, 'windows': 2})
            before = {p.name: file_hash(p) for p in target.iterdir()}
            with patch('src.rul_prediction.v2.preprocessing.read_sources', side_effect=AssertionError('Leitura indevida')):
                with self.assertRaises(FileExistsError):
                    process_dataset(Config('FD001'), root / 'raw', root / 'v2')
            self.assertEqual(before, {p.name: file_hash(p) for p in target.iterdir()})
            self.assertEqual(sentinel.read_bytes(), b'arquivo historico intocado')
            self.assertEqual(legacy_hash, file_hash(ROOT / 'src/common/preprocessing/preproc_data_nasa.py'))
            manifest = json.loads((target / 'manifest.json').read_text())
            self.assertEqual(manifest['config']['seed'], 42)
            self.assertEqual(len(manifest['sha256']), 14)
            self.assertTrue(manifest['input_sha256'])

    def test_integrity_check_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory(prefix='rul_v2_test_') as directory:
            root = Path(directory)
            synthetic_sources(root / 'raw')
            target = process_dataset(Config('FD001'), root / 'raw', root / 'v2')
            (target / 'features.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Hash divergente'):
                validate_saved_dataset(target)

    def test_multi_regime_artifacts_reload_and_reproduce_saved_windows(self):
        for dataset in ['FD002', 'FD004']:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory(prefix='rul_v2_multi_') as directory:
                root = Path(directory)
                synthetic_sources(root / 'raw', dataset)
                target = process_dataset(Config(dataset), root / 'raw', root / 'v2')
                with patch.object(RULPreprocessor, 'fit', side_effect=AssertionError('Reajuste indevido')):
                    loaded = load_preprocessor(target)
                    # Releitura dos mesmos bytes: CSV pode arredondar floats da fixture em memória.
                    _, test = read_sources(Config(dataset), root / 'raw')
                    arrays, metadata = make_windows(loaded.transform(test), loaded.features_,
                                                    dataset, 'test_terminal', terminal=True)
                self.assertEqual(len(loaded.scalers_), 6)
                self.assertIsNotNone(loaded.kmeans_)
                with np.load(target / 'test_terminal.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(arrays['X'], saved['X'])
                self.assertEqual(metadata.end_cycle.tolist(), [35, 19])

    def test_historical_output_paths_are_protected(self):
        for output in [ROOT / 'data/processed_data', ROOT / 'data/processed_anomaly',
                       ROOT / 'models', ROOT / 'results', ROOT / 'src']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                output_target(output, ROOT / 'data/raw_data', 'FD001')
        with tempfile.TemporaryDirectory(prefix='rul_v2_test_') as directory:
            empty = Path(directory) / 'FD001'
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                output_target(Path(directory), Path(directory) / 'raw', 'FD001')

    def test_cli_serialization_with_synthetic_files_only(self):
        with tempfile.TemporaryDirectory(prefix='rul_v2_cli_') as directory:
            root = Path(directory)
            synthetic_sources(root / 'raw')
            command = [sys.executable, '-B', '-m', 'src.rul_prediction.v2.preprocessing',
                       '--datasets', 'FD001', '--raw-dir', str(root / 'raw'),
                       '--output-dir', str(root / 'v2')]
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    encoding='utf-8', env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            loaded = load_preprocessor(root / 'v2/FD001')
            self.assertIsInstance(loaded, RULPreprocessor)
            result = subprocess.run(command + ['--validate-only'], cwd=ROOT, capture_output=True,
                                    text=True, encoding='utf-8', env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
