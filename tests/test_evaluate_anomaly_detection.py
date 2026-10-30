"""Testes com dados sintéticos; não carregam nem treinam redes."""

import ast
import unittest

import numpy as np
import pandas as pd

from src.ml.evaluate_anomaly_detection import (
    ROOT, annotate_alerts, motor_metrics, coverage_by_rul, nominal_analysis,
    validate_windows, compare_persistence, tag, validation_errors_from_csv,
)


def windows(cycles, errors, split='train', unit=1, failure=100):
    return pd.DataFrame({'Dataset': 'FD001', 'Split': split, 'unit_nr': unit,
                         'start_cycle': np.array(cycles) - 29, 'end_cycle': cycles,
                         'RUL': failure - np.array(cycles), 'reconstruction_error': errors})


def inventory(split='train', unit=1, last=100, last_rul=0, role='fit_train'):
    return {'Dataset': 'FD001', 'Split': split, 'unit_nr': unit,
            'first_observed_cycle': 1, 'last_observed_cycle': last,
            'total_observed_cycles': last, 'last_RUL': last_rul,
            'DevelopmentRole': role}


class EvaluationTests(unittest.TestCase):
    def test_persistence_resets_at_gap_motor_and_split(self):
        data = pd.concat([
            windows([30, 31, 33, 34, 35, 36], [2, 2, 2, 2, 2, 1]),
            windows([30, 31], [2, 2], unit=2),
            windows([32], [2], split='test'),
        ], ignore_index=True)
        result = annotate_alerts(data, 1, 3)
        active = result[result.persistent_anomaly]
        self.assertEqual(active[['Split', 'unit_nr', 'end_cycle']].values.tolist(), [['train', 1, 35]])
        self.assertFalse(result.loc[result.end_cycle == 36, 'above_threshold'].any())
        self.assertEqual(int(result.transient_window.sum()), 5)

    def test_parity_with_original_function_without_importing_tensorflow(self):
        # Executa a função original isolada; o módulo completo importaria TensorFlow.
        path = ROOT / 'src/ml/detect_anomalies.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'persistent_alerts')
        scope = {'np': np, 'pd': pd, 'CONSECUTIVE_WINDOWS': 3}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        rng = np.random.default_rng(42)
        frames = []
        for split in ['train', 'test']:
            for unit in [1, 2, 3]:
                cycles = np.delete(np.arange(30, 90), [10, 26, 27])
                frames.append(windows(cycles, rng.choice([0., 1., 2.], len(cycles)), split, unit))
        data = pd.concat(frames).sample(frac=1, random_state=42)
        for k in [1, 3, 5]:
            actual = annotate_alerts(data, 1., k)
            for split in ['train', 'test']:
                expected, _ = scope['persistent_alerts'](data[data.Split == split], 1., k)
                selected = actual[actual.Split == split].reset_index(drop=True)
                for column in ['above_threshold', 'persistent_anomaly', 'first_persistent_anomaly_cycle']:
                    pd.testing.assert_series_equal(selected[column], expected[column])

    def test_rul_in_full_train_and_truncated_test(self):
        data = pd.concat([windows([30, 31, 32], [2, 2, 2]),
                          windows([30, 31, 32], [2, 2, 2], 'test', failure=120)])
        inv = pd.DataFrame([inventory(), inventory('test', last=80, last_rul=40, role='test')])
        result = motor_metrics(annotate_alerts(data, 1, 3), inv, 3).set_index('Split')
        self.assertEqual(result.loc['train', 'alert_RUL'], 68)
        self.assertEqual(result.loc['test', 'alert_RUL'], 88)
        self.assertEqual(result.loc['test', 'detection_cycle'], 32)
        data['RUL'] = np.nan
        fallback = motor_metrics(annotate_alerts(data, 1, 3), inv, 3).set_index('Split')
        self.assertEqual(fallback.loc['test', 'alert_RUL'], 88)

    def test_no_alert_is_nan_and_insufficient_is_separate(self):
        data = pd.concat([windows([30, 31, 32], [0, 0, 0]), windows([30, 31], [2, 2], unit=2)])
        inv = pd.DataFrame([inventory(), inventory(unit=2), inventory(unit=3, last=20)])
        result = motor_metrics(annotate_alerts(data, 1, 3), inv, 3)
        self.assertEqual(result.detection_status.tolist(), ['no_detection', 'insufficient_windows', 'insufficient_windows'])
        self.assertTrue(result.detection_cycle.isna().all())
        self.assertTrue(result.alert_RUL.isna().all())
        self.assertTrue(pd.isna(result.iloc[2].above_windows_pct))

    def test_nominal_uses_complete_train_and_fully_contained_windows(self):
        data = pd.concat([windows([30, 31, 32, 33, 34], [2] * 5, failure=110),
                          windows([30, 31, 32], [2] * 3, 'test', failure=110)])
        inv = pd.DataFrame([inventory(last=110), inventory(unit=2, last=90),
                            inventory('test', last=80, last_rul=30, role='test')])
        scored = annotate_alerts(data, 1, 3)
        metrics = motor_metrics(scored, inv, 3)
        detail, summary = nominal_analysis(scored, metrics, .3, 30)
        first = detail[detail.unit_nr == 1].iloc[0]
        self.assertEqual(first.nominal_end_cycle, 33)
        self.assertEqual(first.nominal_windows, 4)
        self.assertEqual(first.nominal_persistent_windows, 2)
        total = summary[summary.DevelopmentRole == 'all_train'].iloc[0]
        self.assertEqual(total.eligible_motors, 1)
        self.assertEqual(total.insufficient_nominal_cycles, 1)
        self.assertEqual(total.nominal_motor_alert_pct, 100)
        self.assertNotIn('test', detail.DevelopmentRole.tolist())

    def test_cumulative_coverage_denominator_and_missing_rul(self):
        motors = pd.DataFrame({'Split': ['test'] * 5, 'last_RUL': [10., 20., 40., 20., np.nan],
                               'sufficient_windows': [True, True, True, False, True],
                               'detected': [True, False, True, False, False],
                               'alert_RUL': [50., np.nan, 80., np.nan, np.nan]})
        result = coverage_by_rul(motors).set_index('RULGroup')
        self.assertEqual(result.loc['RUL <= 15', 'coverage_pct'], 100)
        self.assertEqual(result.loc['RUL <= 30', 'coverage_pct'], 50)
        self.assertEqual(result.loc['RUL <= 30', 'insufficient_motors'], 1)
        self.assertEqual(result.loc['All', 'coverage_pct'], 50)
        self.assertEqual(result.loc['All', 'alert_rul_median'], 65)
        self.assertEqual(result.loc['All', 'alert_rul_iqr'], 15)

    def test_persistence_disappearing_and_delayed(self):
        data = pd.concat([windows(np.arange(30, 35), [2] * 5), windows(np.arange(30, 33), [2] * 3, unit=2)])
        inv = pd.DataFrame([inventory(), inventory(unit=2)])
        motors = pd.concat([tag(motor_metrics(annotate_alerts(data, 1, k), inv, k), 'FD001', 99, 1, k) for k in [3, 5]])
        row = compare_persistence(motors).iloc[0]
        self.assertEqual(row.alerts_disappearing, 1)
        self.assertEqual(row.alerts_later, 1)
        self.assertEqual(row.median_delay_cycles_among_later, 2)
        self.assertEqual(row.alerts_new_at_5, 0)

    def test_validation_rejects_duplicates_missing_columns_and_inconsistent_rul(self):
        data = windows([30, 31, 32], [1, 1, 1])
        for invalid in [pd.concat([data, data.iloc[:1]]), data.drop(columns='Split'),
                        data.assign(RUL=[70, 70, 70]), data.assign(reconstruction_error=[1, np.nan, 1])]:
            with self.assertRaises(ValueError):
                validate_windows(invalid, 'FD001', 30)
        ordered = validate_windows(data.iloc[::-1], 'FD001', 30)
        self.assertEqual(ordered.end_cycle.tolist(), [30, 31, 32])
        self.assertTrue(validate_windows(data.drop(columns='RUL'), 'FD001', 30).RUL.isna().all())

    def test_calibration_requires_verified_validation_origin(self):
        with self.assertRaises(ValueError):
            validation_errors_from_csv('FD001', pd.DataFrame(), None, {'source': 'test'})

    def test_nominal_diagnostic_absent_for_test_only(self):
        data = annotate_alerts(windows([30, 31, 32], [2, 2, 2], 'test'), 1, 3)
        motors = motor_metrics(data, pd.DataFrame([inventory('test', last=80, last_rul=20)]), 3)
        detail, summary = nominal_analysis(data, motors, .3, 30)
        self.assertTrue(detail.empty)
        self.assertTrue(summary.empty)
        self.assertIn('nominal_motor_alert_pct', summary)


if __name__ == '__main__':
    unittest.main()
