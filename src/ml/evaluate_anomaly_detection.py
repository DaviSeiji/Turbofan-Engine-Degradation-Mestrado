"""Avaliação descritiva dos CSVs do AE, sem treinamento ou seleção no teste."""

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ml.anomaly_preprocessing import (
    ROOT, RAW_DIR, PROCESSED_DIR, DATASETS, SETTINGS, read_trajectory, file_hash,
)

SOURCE_DIR = ROOT / 'results' / 'anomaly_detection'
OUTPUT_DIR = ROOT / 'results' / 'anomaly_evaluation'
RUL_LIMITS = [15, 30, 50, 75, 100]
PERCENTILES = [95.0, 97.0, 99.0, 99.5]
PERSISTENCES = [3, 5]
KEYS = ['Split', 'unit_nr', 'start_cycle', 'end_cycle']
CONFIG_KEYS = ['Dataset', 'Percentile', 'Threshold', 'Persistence']


def percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else np.nan


def quartiles(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return {'alert_rul_median': np.nan, 'alert_rul_q25': np.nan,
                'alert_rul_q75': np.nan, 'alert_rul_iqr': np.nan}
    q25, median, q75 = values.quantile([0.25, 0.5, 0.75])
    return {'alert_rul_median': median, 'alert_rul_q25': q25,
            'alert_rul_q75': q75, 'alert_rul_iqr': q75 - q25}


def validate_windows(table, dataset, sequence_length):
    required = set(KEYS + ['Dataset', 'reconstruction_error'])
    if required - set(table):
        raise ValueError(f'Colunas obrigatórias ausentes: {sorted(required - set(table))}')
    if table.empty:
        raise ValueError('CSV de erros vazio: não há janelas para avaliar.')
    table = table.copy()
    if table[list(required)].isna().any().any():
        raise ValueError('Chaves/erros de reconstrução ausentes.')
    if not table.Dataset.eq(dataset).all() or not table.Split.isin(['train', 'test']).all():
        raise ValueError('Dataset/Split inválido.')
    for name in ['unit_nr', 'start_cycle', 'end_cycle']:
        values = pd.to_numeric(table[name], errors='raise')
        if not np.isfinite(values).all() or (values < 1).any() or (values % 1 != 0).any():
            raise ValueError(f'{name} deve conter inteiros positivos.')
        table[name] = values.astype(int)
    if table.duplicated(['Split', 'unit_nr', 'end_cycle']).any():
        raise ValueError('Ciclos duplicados para o mesmo motor/split.')
    errors = pd.to_numeric(table.reconstruction_error, errors='raise')
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('Erros de reconstrução inválidos.')
    table['reconstruction_error'] = errors
    if not (table.end_cycle - table.start_cycle + 1).eq(sequence_length).all():
        raise ValueError('Janelas incompatíveis com sequence_length.')
    if 'RUL' not in table:
        table['RUL'] = np.nan
    table['RUL'] = pd.to_numeric(table.RUL, errors='raise')
    known = table.RUL.dropna()
    if not np.isfinite(known).all() or (known < 0).any():
        raise ValueError('Metadados RUL inválidos.')
    for _, group in table.groupby(['Split', 'unit_nr']):
        failure = (group.end_cycle + group.RUL).dropna()
        if len(failure) and not np.allclose(failure, failure.iloc[0], atol=1e-6, rtol=0):
            raise ValueError('RUL não diminui de forma consistente com os ciclos.')
    return table.sort_values(['Split', 'unit_nr', 'end_cycle']).reset_index(drop=True)


def annotate_alerts(table, threshold, persistence):
    """Versão vetorizada da regra de persistent_alerts, testada contra a original."""
    if persistence < 1 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Persistência/threshold inválidos.')
    table = table.sort_values(['Split', 'unit_nr', 'end_cycle']).reset_index(drop=True).copy()
    above = table.reconstruction_error.gt(threshold)
    boundary = (table.Split.ne(table.Split.shift()) | table.unit_nr.ne(table.unit_nr.shift())
                | table.end_cycle.diff().ne(1))
    segments = boundary.cumsum()
    runs = (boundary | above.ne(above.shift())).cumsum()
    streak = table.groupby(runs).cumcount() + 1
    run_length = table.groupby(runs).end_cycle.transform('size')
    table['above_threshold'] = above
    table['persistent_anomaly'] = above & streak.ge(persistence)
    table['first_persistent_anomaly_cycle'] = table.end_cycle.where(table.persistent_anomaly).groupby(
        [table.Split, table.unit_nr]).transform('min')
    # Classificação retrospectiva do episódio, sem antecipar o ciclo do alerta.
    table['transient_window'] = above & run_length.lt(persistence)
    table['sustained_episode_window'] = above & run_length.ge(persistence)
    table['above_episode_length'] = np.where(above, run_length, 0)
    table['contiguous_windows'] = table.groupby(segments).end_cycle.transform('size')
    return table


def motor_metrics(windows, inventory, persistence):
    records = []
    groups = {(s, int(u)): g for (s, u), g in windows.groupby(['Split', 'unit_nr'])}
    for _, item in inventory.iterrows():
        group = groups.get((item.Split, int(item.unit_nr)), windows.iloc[:0])
        n = len(group)
        sufficient = bool(n and group.contiguous_windows.max() >= persistence)
        alerted = group[group.persistent_anomaly]
        first = alerted.iloc[0] if not alerted.empty else None
        cycle = float(first.end_cycle) if first is not None else np.nan
        rul = float(first.RUL) if first is not None else np.nan
        if first is not None and pd.isna(rul) and pd.notna(item.last_RUL):
            rul = item.last_RUL + item.last_observed_cycle - cycle
        row = item.to_dict()
        row.update(windows_evaluated=n, sufficient_windows=sufficient, detected=first is not None,
                   detection_status=('detected' if first is not None else
                                     'no_detection' if sufficient else 'insufficient_windows'),
                   detection_cycle=cycle, alert_RUL=rul,
                   above_windows=int(group.above_threshold.sum()),
                   persistent_windows=int(group.persistent_anomaly.sum()),
                   above_windows_pct=percent(group.above_threshold.sum(), n),
                   persistent_windows_pct=percent(group.persistent_anomaly.sum(), n),
                   first_evaluated_end_cycle=group.end_cycle.min() if n else np.nan,
                   last_evaluated_end_cycle=group.end_cycle.max() if n else np.nan,
                   first_alert_life_fraction=(
                       (cycle - item.first_observed_cycle + 1) / item.total_observed_cycles
                       if item.Split == 'train' and first is not None else np.nan))
        records.append(row)
    return pd.DataFrame(records)


def coverage_by_rul(motors):
    records = []
    for split, group in motors.groupby('Split'):
        for limit in RUL_LIMITS + [None]:
            selected = group if limit is None else group[group.last_RUL.le(limit)]
            eligible = selected[selected.sufficient_windows]
            detected = eligible[eligible.detected]
            records.append({'Split': split, 'RULGroup': f'RUL <= {limit}' if limit is not None else 'All',
                            'RULLimit': limit, 'cumulative': True,
                            'motors_in_group': len(selected), 'eligible_motors': len(eligible),
                            'insufficient_motors': len(selected) - len(eligible),
                            'missing_final_RUL': int(group.last_RUL.isna().sum()),
                            'detected_motors': len(detected),
                            'coverage_pct': percent(len(detected), len(eligible)),
                            **quartiles(detected.alert_RUL)})
    return pd.DataFrame(records)


def nominal_analysis(windows, motors, fraction, sequence_length):
    train = motors[(motors.Split == 'train') & motors.last_RUL.eq(0)
                   & motors.first_observed_cycle.eq(1)].copy()
    train['nominal_end_cycle'] = (train.first_observed_cycle
                                  + np.floor(fraction * train.total_observed_cycles) - 1)
    rows = []
    groups = {int(u): g for u, g in windows[windows.Split == 'train'].groupby('unit_nr')}
    for _, motor in train.iterrows():
        g = groups.get(int(motor.unit_nr), windows.iloc[:0])
        nominal = g[g.start_cycle.ge(motor.first_observed_cycle) & g.end_cycle.le(motor.nominal_end_cycle)]
        rows.append({'unit_nr': motor.unit_nr, 'DevelopmentRole': motor.DevelopmentRole,
                     'nominal_end_cycle': motor.nominal_end_cycle,
                     'enough_nominal_cycles': np.floor(fraction * motor.total_observed_cycles) >= sequence_length,
                     'nominal_windows': len(nominal), 'nominal_above_windows': int(nominal.above_threshold.sum()),
                     'nominal_persistent_windows': int(nominal.persistent_anomaly.sum()),
                     'any_nominal_exceedance': bool(nominal.above_threshold.any()),
                     'any_nominal_persistent_alert': bool(nominal.persistent_anomaly.any()),
                     'first_alert_life_fraction': motor.first_alert_life_fraction})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame(columns=['DevelopmentRole', 'eligible_motors', 'insufficient_nominal_cycles',
            'motors_with_nominal_alert', 'nominal_motor_alert_pct', 'nominal_above_windows_pct',
            'nominal_persistent_windows_pct'])
    summaries = []
    selections = [('all_train', detail)] + list(detail.groupby('DevelopmentRole'))
    for role, g in selections:
        eligible = g[g.nominal_windows > 0]
        count = len(eligible)
        summaries.append({'DevelopmentRole': role, 'motors_total': len(g), 'eligible_motors': count,
                          'insufficient_nominal_cycles': int((~g.enough_nominal_cycles).sum()),
                          'motors_without_nominal_windows': int((g.nominal_windows == 0).sum()),
                          'motors_with_nominal_exceedance': int(eligible.any_nominal_exceedance.sum()),
                          'motors_with_nominal_alert': int(eligible.any_nominal_persistent_alert.sum()),
                          'nominal_motor_alert_pct': percent(eligible.any_nominal_persistent_alert.sum(), count),
                          'nominal_windows': int(g.nominal_windows.sum()),
                          'nominal_above_windows_pct': percent(g.nominal_above_windows.sum(), g.nominal_windows.sum()),
                          'nominal_persistent_windows_pct': percent(g.nominal_persistent_windows.sum(), g.nominal_windows.sum())})
    return detail, pd.DataFrame(summaries)


def compare_persistence(motors):
    a = motors[(motors.Percentile == 99) & (motors.Persistence == 3)]
    b = motors[(motors.Percentile == 99) & (motors.Persistence == 5)]
    merged = a.merge(b, on=['Dataset', 'Split', 'unit_nr'], suffixes=('_3', '_5'), validate='one_to_one')
    records = []
    for (dataset, split), g in merged.groupby(['Dataset', 'Split']):
        both = g.detected_3 & g.detected_5
        later = both & g.detection_cycle_5.gt(g.detection_cycle_3)
        records.append({'Dataset': dataset, 'Split': split, 'motors': len(g),
                        'detected_3': int(g.detected_3.sum()), 'detected_5': int(g.detected_5.sum()),
                        'alerts_disappearing': int((g.detected_3 & ~g.detected_5).sum()),
                        'alerts_later': int(later.sum()),
                        'alerts_new_at_5': int((~g.detected_3 & g.detected_5).sum()),
                        'median_delay_cycles_among_later': (g.loc[later, 'detection_cycle_5'] - g.loc[later, 'detection_cycle_3']).median()})
    return pd.DataFrame(records)


def verify_artifact(folder, manifest, name):
    path = folder / name
    expected = manifest.get('sha256', {}).get(name)
    if not path.is_file() or expected is None or file_hash(path) != expected:
        raise ValueError(f'Artefato ausente ou hash incompatível: {path}')
    return path


def load_context(dataset, notes):
    folder = SOURCE_DIR / dataset
    config = json.loads((folder / 'detection_config.json').read_text(encoding='utf-8'))
    if config['Dataset'] != dataset:
        raise ValueError('Dataset divergente na configuração.')
    if not config.get('splits') or not set(config['splits']).issubset({'train', 'test'}):
        raise ValueError('Splits da configuração inválidos.')
    calibration = config['calibration']
    if calibration['percentile'] != 99 or not np.isfinite(calibration['threshold']) or calibration['threshold'] < 0:
        raise ValueError('Esperado threshold original P99 finito e não negativo.')
    processed = PROCESSED_DIR / dataset
    manifest, roles = None, {}
    try:
        manifest = json.loads((processed / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['fingerprint'] != calibration['fingerprint']:
            raise ValueError('Fingerprint de pré-processamento divergente.')
        if file_hash(RAW_DIR / f'train_{dataset}.txt') != manifest['raw_train_sha256']:
            raise ValueError('Trajetória de treino alterada.')
    except (OSError, ValueError, KeyError) as exc:
        notes.append(f'Pré-processamento não verificado: {exc}')
        manifest = None
    length = manifest['sequence_length'] if manifest else 30
    fraction = manifest['nominal_fraction'] if manifest else 0.30
    if manifest is None:
        notes.append('Janela=30 e fração nominal=0,30 assumidas conforme protocolo solicitado; sem confirmação pelo manifesto.')
    if manifest:
        try:
            units = pd.read_csv(verify_artifact(processed, manifest, 'nominal_units.csv'))
            if units.unit_nr.duplicated().any() or not units.split.isin(['train', 'validation']).all():
                raise ValueError('Partição de desenvolvimento inválida.')
            roles = dict(zip(units.unit_nr, units.split.map({'train': 'fit_train', 'validation': 'validation'})))
        except (OSError, ValueError, KeyError) as exc:
            notes.append(f'Separação treino/validação indisponível: {exc}')
    original = pd.read_csv(folder / 'reconstruction_errors.csv')
    windows = validate_windows(original, dataset, length)
    if not set(windows.Split).issubset(config['splits']):
        raise ValueError('CSV contém split não declarado na configuração.')
    sorted_keys = original.sort_values(['Split', 'unit_nr', 'end_cycle']).index
    if not sorted_keys.equals(original.index):
        notes.append('CSV foi ordenado temporalmente para avaliação; arquivo original preservado.')
    inventory, raw_data = [], {}
    saved_summary = pd.read_csv(folder / 'detection_summary.csv')
    if not {'Split', 'unit_nr', 'windows_evaluated'}.issubset(saved_summary):
        raise ValueError('detection_summary sem chaves/contagens obrigatórias.')
    if saved_summary[['Split', 'unit_nr']].isna().any().any() or saved_summary.duplicated(['Split', 'unit_nr']).any():
        raise ValueError('Chaves duplicadas/ausentes em detection_summary.')
    for split in config['splits']:
        try:
            raw = read_trajectory(dataset, split)
            raw_data[split] = raw
        except (OSError, ValueError) as exc:
            raw = None
            notes.append(f'{split}: dados brutos indisponíveis; inventário limitado aos CSVs: {exc}')
        current = windows[windows.Split == split]
        if raw is not None and not set(current.unit_nr).issubset(set(raw.unit_nr)):
            raise ValueError(f'{split}: motores do CSV não existem na trajetória bruta.')
        units = (sorted(raw.unit_nr.unique()) if raw is not None else
                 sorted(set(saved_summary.loc[saved_summary.Split == split, 'unit_nr']) | set(current.unit_nr)))
        for unit in units:
            g = current[current.unit_nr == unit]
            if raw is not None:
                trajectory = raw[raw.unit_nr == unit]
                first, last, n = trajectory.time_cycles.min(), trajectory.time_cycles.max(), len(trajectory)
                last_rul = float(trajectory.RUL.iloc[-1])
                if not g.empty:
                    truth = trajectory.set_index('time_cycles').RUL.reindex(g.end_cycle)
                    if truth.isna().any() or not np.allclose(g.RUL.dropna(), truth.to_numpy()[g.RUL.notna()], atol=1e-6, rtol=0):
                        raise ValueError(f'{dataset}/{split}/{unit}: RUL do CSV diverge dos dados brutos.')
                    missing = g.RUL.isna()
                    if missing.any():
                        windows.loc[g.index[missing], 'RUL'] = truth.to_numpy()[missing]
                        notes.append(f'{split}/{unit}: RUL ausente recuperado dos metadados brutos.')
                expected_windows = max(0, n - length + 1)
                if len(g) != expected_windows:
                    notes.append(f'{split}/{unit}: {len(g)} janelas disponíveis, {expected_windows} esperadas.')
            elif not g.empty:
                first, last = g.start_cycle.min(), g.end_cycle.max()
                n, last_rul = last - first + 1, g.RUL.iloc[-1]
            else:
                first = last = n = last_rul = np.nan
            inventory.append({'Dataset': dataset, 'Split': split, 'unit_nr': int(unit),
                              'first_observed_cycle': first, 'last_observed_cycle': last,
                              'total_observed_cycles': n, 'last_RUL': last_rul,
                              'failure_cycle_from_RUL': last + last_rul,
                              'failure_observed': split == 'train' and last_rul == 0,
                              'trajectory_source': 'raw' if raw is not None else 'CSV_available_span',
                              'DevelopmentRole': roles.get(unit, 'unknown') if split == 'train' else 'test'})
    inventory = pd.DataFrame(inventory)
    return windows, inventory, raw_data, config, manifest, fraction, length


def validation_errors_from_csv(dataset, windows, manifest, calibration):
    if manifest is None or calibration.get('source') != 'validation_nominal':
        raise ValueError('Origem nominal de validação não verificável.')
    folder = PROCESSED_DIR / dataset
    meta = pd.read_csv(verify_artifact(folder, manifest, 'validation_metadata.csv'))
    train_meta = pd.read_csv(verify_artifact(folder, manifest, 'train_metadata.csv'))
    if set(meta.unit_nr) & set(train_meta.unit_nr):
        raise ValueError('Motores se sobrepõem entre treino e validação.')
    if len(meta) != calibration['validation_windows']:
        raise ValueError('Número de janelas de calibração divergente.')
    joined = meta.merge(windows[windows.Split == 'train'][['unit_nr', 'start_cycle', 'end_cycle', 'reconstruction_error']],
                        on=['unit_nr', 'start_cycle', 'end_cycle'], how='left', validate='one_to_one')
    if joined.reconstruction_error.isna().any():
        raise ValueError('CSV não contém todos os erros das janelas nominais de validação.')
    values = joined.reconstruction_error.to_numpy()
    if not np.isclose(np.percentile(values, 99), calibration['threshold'], rtol=1e-5, atol=1e-10):
        raise ValueError('P99 dos erros recuperados não reproduz o threshold original.')
    return joined


def recover_calibration(dataset, output):
    """Etapa opcional: só inferência nas janelas nominais salvas, nunca fit."""
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from tensorflow import keras
    from src.ml.detect_anomalies import score_windows
    folder = PROCESSED_DIR / dataset
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((SOURCE_DIR / dataset / 'detection_config.json').read_text(encoding='utf-8'))
    calibration = config['calibration']
    if manifest['fingerprint'] != calibration['fingerprint'] or calibration['source'] != 'validation_nominal':
        raise ValueError('Calibração incompatível com o manifesto.')
    model_path = ROOT / 'models' / 'best_models_autoencoder' / f'best_autoencoder_{dataset}.keras'
    if file_hash(model_path) != calibration['model_sha256']:
        raise ValueError('Hash do modelo divergente.')
    verify_artifact(folder, manifest, 'preprocessor.joblib')
    metadata = pd.read_csv(verify_artifact(folder, manifest, 'validation_metadata.csv'))
    train_meta = pd.read_csv(verify_artifact(folder, manifest, 'train_metadata.csv'))
    if set(metadata.unit_nr) & set(train_meta.unit_nr):
        raise ValueError('Motores de treino e validação sobrepostos.')
    with np.load(verify_artifact(folder, manifest, 'nominal_windows.npz')) as data:
        X_val = data['X_val']
    if len(X_val) != len(metadata) or len(X_val) != calibration['validation_windows']:
        raise ValueError('Tamanho da validação nominal incompatível.')
    model = None
    try:
        model = keras.models.load_model(model_path, compile=False)
        scores, _ = score_windows(model, X_val)
    finally:
        del model
        keras.backend.clear_session()
    if not np.isclose(np.percentile(scores, 99), calibration['threshold'], rtol=1e-5, atol=1e-10):
        raise ValueError('Inferência não reproduz o threshold original dentro da tolerância.')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'recovered_validation_errors.csv'
    if path.exists():
        raise FileExistsError(f'Erros recuperados já existem: {path}')
    metadata.assign(reconstruction_error=scores).to_csv(path, index=False)
    provenance = {'fingerprint': manifest['fingerprint'], 'model_sha256': calibration['model_sha256'],
                  'source': 'saved_validation_nominal_X_val', 'errors_sha256': file_hash(path)}
    (output / 'recovered_validation_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


def threshold_options(dataset, windows, manifest, calibration, output, notes):
    options = {99.0: float(calibration['threshold'])}
    source = 'original_detection_config'
    try:
        recovered = validation_errors_from_csv(dataset, windows, manifest, calibration)
        source = 'existing_train_CSV_joined_to_validation_nominal_metadata'
    except (OSError, ValueError, KeyError) as exc:
        notes.append(f'Recuperação de calibração pelos CSVs: {exc}')
        try:
            path = output / 'recovered_validation_errors.csv'
            provenance = json.loads((output / 'recovered_validation_provenance.json').read_text(encoding='utf-8'))
            if (manifest is None or provenance['fingerprint'] != manifest['fingerprint']
                    or provenance['model_sha256'] != calibration['model_sha256']
                    or provenance['errors_sha256'] != file_hash(path)):
                raise ValueError('Proveniência da inferência opcional inválida.')
            recovered = pd.read_csv(path)
            meta = pd.read_csv(verify_artifact(PROCESSED_DIR / dataset, manifest, 'validation_metadata.csv'))
            if not recovered[['unit_nr', 'start_cycle', 'end_cycle']].equals(meta[['unit_nr', 'start_cycle', 'end_cycle']]):
                raise ValueError('Janelas recuperadas não correspondem à validação nominal.')
            source = 'optional_saved_validation_inference'
        except (OSError, ValueError, KeyError) as recovery_exc:
            notes.append(f'Sensibilidade P95/P97/P99,5 não executada: sem erros de calibração verificados ({recovery_exc}).')
            return options, source
    scores = recovered.reconstruction_error.to_numpy(dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        notes.append('Sensibilidade não executada: erros de calibração inválidos.')
        return options, 'original_detection_config'
    if not np.isclose(np.percentile(scores, 99), options[99], rtol=1e-5, atol=1e-10):
        notes.append('Sensibilidade não executada: P99 recuperado diverge do original.')
        return options, 'original_detection_config'
    for p in PERCENTILES:
        if p != 99:
            options[p] = float(np.percentile(scores, p))
    recovered.to_csv(output / 'calibration_errors_used.csv', index=False)
    return dict(sorted(options.items())), source


def regime_analysis(dataset, baseline, raw_data, manifest, notes):
    if dataset not in ('FD002', 'FD004'):
        notes.append('Análise de regimes não aplicável: dataset de condição operacional única.')
        return None
    try:
        if manifest is None:
            raise ValueError('Manifesto de pré-processamento não verificado.')
        processor = joblib.load(verify_artifact(PROCESSED_DIR / dataset, manifest, 'preprocessor.joblib'))
        if processor.kmeans is None:
            raise ValueError('KMeans ajustado ausente.')
        frames, profiles = [], []
        for split, raw in raw_data.items():
            raw = raw.sort_values(['unit_nr', 'time_cycles']).copy()
            raw['regime'] = processor.kmeans.predict(raw[SETTINGS])
            for unit, trajectory in raw.groupby('unit_nr'):
                g = baseline[(baseline.Split == split) & baseline.unit_nr.eq(unit)].copy()
                if g.empty:
                    continue
                regimes = trajectory.regime.to_numpy()
                cycles = trajectory.time_cycles.to_numpy()
                transitions = np.r_[False, regimes[1:] != regimes[:-1]]
                transition_cycles = cycles[transitions]
                end_regime = pd.Series(regimes, index=cycles)
                g['end_regime'] = g.end_cycle.map(end_regime)
                g['transition_count_inside'] = (np.searchsorted(transition_cycles, g.end_cycle, side='right')
                                                - np.searchsorted(transition_cycles, g.start_cycle, side='right'))
                g['transition_inside'] = g.transition_count_inside.gt(0)
                # Próximo ao fim = ciclo final e dois ciclos anteriores; sem olhar o futuro.
                g['transition_near_end'] = (np.searchsorted(transition_cycles, g.end_cycle, side='right')
                                            - np.searchsorted(transition_cycles, g.end_cycle - 2, side='left')) > 0
                g['error_change_from_previous_window'] = g.reconstruction_error.diff().where(g.end_cycle.diff().eq(1))
                frames.append(g)
                lookup = g.set_index('end_cycle')
                for lag in range(-3, 4):
                    selected = lookup.reindex(transition_cycles + lag)
                    selected = selected[selected.reconstruction_error.notna()]
                    if not selected.empty:
                        profiles.append(pd.DataFrame({'Split': split, 'unit_nr': unit, 'lag': lag,
                                                      'reconstruction_error': selected.reconstruction_error.to_numpy(),
                                                      'above_threshold': selected.above_threshold.to_numpy(dtype=bool)}))
        if not frames:
            raise ValueError('Sem dados brutos/janelas para reconstruir regimes.')
        windows = pd.concat(frames, ignore_index=True)
        rows = []
        for split, g in windows.groupby('Split'):
            for dimension in ['end_regime', 'transition_inside', 'transition_near_end']:
                for value, subset in g.groupby(dimension):
                    rows.append({'Dataset': dataset, 'Split': split, 'Dimension': dimension, 'Group': str(value),
                                 'windows': len(subset), 'motors': subset.unit_nr.nunique(),
                                 'mean_error': subset.reconstruction_error.mean(),
                                 'median_error': subset.reconstruction_error.median(),
                                 'above_threshold_pct': 100 * subset.above_threshold.mean(),
                                 'persistent_window_pct': 100 * subset.persistent_anomaly.mean(),
                                 'mean_error_change': subset.error_change_from_previous_window.mean()})
                if g[dimension].nunique() < 2:
                    notes.append(f'{split}: {dimension} sem grupo de contraste; associação não comparável.')
        if profiles:
            events = pd.concat(profiles).groupby(['Split', 'lag']).agg(
                observations=('reconstruction_error', 'size'), mean_error=('reconstruction_error', 'mean'),
                median_error=('reconstruction_error', 'median'), above_fraction=('above_threshold', 'mean'),
            ).reset_index()
        else:
            events = pd.DataFrame()
        return windows, pd.DataFrame(rows), events
    except (OSError, ValueError, KeyError, AttributeError, ImportError) as exc:
        notes.append(f'Análise de regimes não executada: {exc}')
        return None


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_individual(windows, threshold, dataset, output, units=(1, 2, 3)):
    for split in windows.Split.unique():
        for unit in units:
            g = windows[(windows.Split == split) & windows.unit_nr.eq(unit)]
            if g.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(g.end_cycle, g.reconstruction_error, label='Reconstruction MSE', linewidth=1)
            ax.axhline(threshold, color='red', linestyle='--', label='Original P99 threshold')
            for column, color, label in [('transient_window', 'orange', 'Transient episode (<3 windows)'),
                                         ('persistent_anomaly', 'purple', 'Persistent alert (3 windows)')]:
                selected = g[g[column]]
                if not selected.empty:
                    ax.scatter(selected.end_cycle, selected.reconstruction_error, s=12, color=color, label=label)
            alerts = g[g.persistent_anomaly]
            if not alerts.empty:
                ax.axvline(alerts.end_cycle.iloc[0], color='black', linestyle=':', label='First persistent alert')
            ax.set(title=f'{dataset} | {split} | unit {unit}', xlabel='End cycle', ylabel='Reconstruction MSE')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            save_figure(fig, output / f'trajectory_{split}_{unit:03d}.png')
            if dataset == 'FD004' and split == 'test' and unit == 1:
                g[(g.end_cycle >= 85) & (g.end_cycle <= 120)].to_csv(output / 'fd004_test_unit1_cycles85_120.csv', index=False)


def plot_dataset(motors, baseline_windows, thresholds, coverage, regimes, output, fraction):
    dataset = motors.Dataset.iloc[0]
    baseline = motors[(motors.Percentile == 99) & (motors.Persistence == 3)]
    fig, ax = plt.subplots(figsize=(8, 4))
    all_values = baseline.loc[baseline.detected, 'alert_RUL'].dropna()
    bins = np.histogram_bin_edges(all_values, bins=20)
    for split, g in baseline.groupby('Split'):
        values = g.loc[g.detected, 'alert_RUL'].dropna()
        if len(values):
            ax.hist(values, bins=bins, alpha=0.5, label=f'{split} (n={len(values)})')
    ax.set(title=f'{dataset}: RUL at first alert (detected engines only)', xlabel='RUL (cycles)', ylabel='Engines')
    ax.legend()
    save_figure(fig, output / 'alert_rul_distribution.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    for role, g in baseline[baseline.Split == 'train'].groupby('DevelopmentRole'):
        values = g.first_alert_life_fraction.dropna()
        if len(values):
            ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.5, label=f'{role} (n={len(values)})')
    ax.axvline(fraction, color='red', linestyle='--', label=f'Assumed nominal boundary ({fraction:.0%})')
    ax.set(title=f'{dataset}: first alert in full training trajectory', xlabel='Fraction of observed full lifetime', ylabel='Engines')
    ax.legend()
    save_figure(fig, output / 'alert_relative_lifetime.png')
    if len(thresholds) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        for (split, persistence), g in coverage[coverage.RULGroup == 'All'].groupby(['Split', 'Persistence']):
            g = g.sort_values('Percentile')
            ax.plot(g.Percentile, g.coverage_pct, marker='o', label=f'{split}, {persistence} windows')
        ax.set(title=f'{dataset}: threshold sensitivity (descriptive)', xlabel='Nominal validation percentile', ylabel='Engine coverage (%)', ylim=(0, 105))
        ax.legend()
        save_figure(fig, output / 'threshold_sensitivity.png')
    plot_individual(baseline_windows, thresholds[99], dataset, output)
    if regimes is not None:
        _, summary, events = regimes
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for split, g in summary[summary.Dimension == 'end_regime'].groupby('Split'):
            axes[0].plot(g.Group, g.mean_error, marker='o', label=split)
            axes[1].plot(g.Group, g.above_threshold_pct, marker='o', label=split)
        axes[0].set(xlabel='Regime at window end', ylabel='Mean reconstruction MSE')
        axes[1].set(xlabel='Regime at window end', ylabel='Windows above P99 (%)')
        for ax in axes:
            ax.legend()
        fig.suptitle(f'{dataset}: descriptive regime association')
        save_figure(fig, output / 'regime_analysis.png')
        if not events.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for split, g in events.groupby('Split'):
                ax.plot(g.lag, g.mean_error, marker='o', label=split)
            ax.set(xlabel='Window end cycle relative to transition', ylabel='Mean reconstruction MSE',
                   title=f'{dataset}: transition-aligned errors (overlapping events)')
            ax.legend()
            save_figure(fig, output / 'transition_profile.png')


def audit_original(dataset, baseline, motors, config):
    """Confere a reprodução da detecção salva, sem editar o arquivo de origem."""
    if config['consecutive_windows'] != 3:
        raise ValueError('A auditoria espera a configuração original de 3 janelas.')
    source = pd.read_csv(SOURCE_DIR / dataset / 'reconstruction_errors.csv')
    checked = baseline.merge(source, on=KEYS, suffixes=('_new', '_saved'), validate='one_to_one')
    audit = {}
    for name in ['above_threshold', 'persistent_anomaly']:
        saved = checked[f'{name}_saved'].astype(str).str.lower().map({'true': True, 'false': False})
        if saved.isna().any():
            raise ValueError(f'Valores inválidos na coluna original {name}.')
        audit[f'{name}_mismatches'] = int(saved.ne(checked[f'{name}_new']).sum())
    first = motors[['Split', 'unit_nr', 'detection_cycle']]
    checked = source.merge(first, on=['Split', 'unit_nr'], validate='many_to_one')
    audit['first_cycle_mismatches'] = int((~np.isclose(
        checked.first_persistent_anomaly_cycle, checked.detection_cycle, equal_nan=True)).sum())
    summary = pd.read_csv(SOURCE_DIR / dataset / 'detection_summary.csv')
    compared = summary.merge(motors, on=['Split', 'unit_nr'], suffixes=('_saved', '_new'),
                             how='outer', validate='one_to_one', indicator=True)
    audit['summary_missing_engines'] = int(compared['_merge'].ne('both').sum())
    both = compared[compared['_merge'] == 'both']
    for name in ['detection_cycle', 'windows_evaluated']:
        audit[f'summary_{name}_mismatches'] = int((~np.isclose(
            both[f'{name}_saved'], both[f'{name}_new'], equal_nan=True)).sum())
    return audit


def tag(table, dataset, percentile, threshold, persistence):
    return table.assign(Dataset=dataset, Percentile=percentile, Threshold=threshold, Persistence=persistence)


def evaluate_dataset(dataset, overwrite=False):
    output = OUTPUT_DIR / dataset
    if (output / 'evaluation_config.json').exists() and not overwrite:
        raise FileExistsError(f'{dataset} já avaliado. Use --overwrite para atualizar apenas esta avaliação.')
    output.mkdir(parents=True, exist_ok=True)
    notes = []
    windows, inventory, raw, config, manifest, fraction, length = load_context(dataset, notes)
    thresholds, calibration_source = threshold_options(dataset, windows, manifest, config['calibration'], output, notes)
    tables = {name: [] for name in ['motor_metrics', 'rul_coverage', 'nominal_alerts', 'nominal_motor_metrics']}
    baseline = None
    for percentile, threshold in thresholds.items():
        for persistence in PERSISTENCES:
            annotated = annotate_alerts(windows, threshold, persistence)
            motors = motor_metrics(annotated, inventory, persistence)
            coverage = coverage_by_rul(motors)
            nominal_detail, nominal_summary = nominal_analysis(annotated, motors, fraction, length)
            for name, table in zip(tables, [motors, coverage, nominal_summary, nominal_detail]):
                tables[name].append(tag(table, dataset, percentile, threshold, persistence))
            if percentile == 99 and persistence == 3:
                baseline = annotated
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    motors, coverage, nominal = (tables[name] for name in ['motor_metrics', 'rul_coverage', 'nominal_alerts'])
    if nominal.empty:
        notes.append('Diagnóstico nominal não executado: nenhuma trajetória completa de treino verificável.')
    baseline_motors = motors[(motors.Percentile == 99) & (motors.Persistence == 3)]
    audit = audit_original(dataset, baseline, baseline_motors, config)
    for name, count in audit.items():
        if count:
            notes.append(f'Auditoria da detecção original: {name} = {count}.')
    persistence_table = compare_persistence(motors)
    if persistence_table.alerts_new_at_5.any():
        raise AssertionError('Persistência 5 não pode criar detecções ausentes em persistência 3.')
    tables['persistence_comparison'] = persistence_table
    tables['threshold_sensitivity'] = coverage[coverage.RULGroup == 'All'].merge(
        nominal[nominal.DevelopmentRole == 'all_train'][CONFIG_KEYS + ['nominal_motor_alert_pct',
            'nominal_above_windows_pct', 'nominal_persistent_windows_pct']], on=CONFIG_KEYS, how='left')
    # As taxas nominais nesta tabela referem-se sempre ao treino completo, nunca ao teste.
    tables['threshold_sensitivity']['nominal_scope'] = 'all_train_diagnostic_only'
    tables['evaluation_summary'] = tables['threshold_sensitivity'][tables['threshold_sensitivity'].Percentile == 99]
    regimes = regime_analysis(dataset, baseline, raw, manifest, notes)
    if regimes is not None:
        tables['regime_windows'], tables['regime_analysis'], tables['transition_profile'] = regimes
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    baseline.to_csv(output / 'evaluated_windows_p99_k3.csv', index=False)
    plot_dataset(motors, baseline, thresholds, coverage, regimes, output, fraction)
    source_hashes = {name: file_hash(SOURCE_DIR / dataset / name) for name in
                     ['reconstruction_errors.csv', 'detection_summary.csv', 'detection_config.json']}
    evaluation_config = {
        'Dataset': dataset, 'input_sha256': source_hashes,
        'preprocessing_fingerprint': manifest['fingerprint'] if manifest else None,
        'sequence_length': length, 'nominal_fraction': fraction,
        'thresholds': thresholds, 'persistences': PERSISTENCES,
        'calibration_error_source': calibration_source, 'independent_calibration': False,
        'regime_analysis_executed': regimes is not None,
        'baseline': {'percentile': 99, 'persistence': 3},
        'coverage_denominator': 'engines with at least k consecutive available windows',
        'nominal_denominator': 'full training engines with at least one fully nominal window',
        'transition_near_end': 'a regime change in the final 3 cycles of the window',
        'individual_plots_units': [1, 2, 3], 'original_results_audit': audit,
        'notes': notes,
    }
    (output / 'evaluation_config.json').write_text(json.dumps(evaluation_config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{dataset}: avaliação concluída; {len(thresholds)} thresholds, sem inferência/treino.', flush=True)
    return tables, evaluation_config


def plot_comparison(coverage, summary, output):
    baseline = coverage[(coverage.Percentile == 99) & (coverage.Persistence == 3)]
    labels = [f'RUL <= {limit}' for limit in RUL_LIMITS] + ['All']
    for split, group in baseline.groupby('Split'):
        fig, ax = plt.subplots(figsize=(9, 5))
        for dataset, g in group.groupby('Dataset'):
            g = g.set_index('RULGroup').reindex(labels)
            ax.plot(labels, g.coverage_pct, marker='o', label=dataset)
        ax.set(title=f'{split}: detection coverage (P99, 3 consecutive windows)',
               xlabel='Final RUL — cumulative groups (cycles)', ylabel='Eligible engines detected (%)', ylim=(0, 105))
        ax.legend()
        ax.grid(alpha=0.2)
        save_figure(fig, output / f'rul_coverage_{split}.png')
    for split, group in summary.groupby('Split'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        datasets = sorted(group.Dataset.unique())
        positions = np.arange(len(datasets))
        for k, offset in [(3, -0.18), (5, 0.18)]:
            g = group[group.Persistence == k].set_index('Dataset').reindex(datasets)
            axes[0].bar(positions + offset, g.coverage_pct, width=0.36, label=f'{k} windows')
            axes[1].bar(positions + offset, g.alert_rul_median, width=0.36, label=f'{k} windows')
        for ax in axes:
            ax.set_xticks(positions, datasets)
            ax.legend()
        axes[0].set(ylabel='Eligible engines detected (%)', ylim=(0, 105))
        axes[1].set(ylabel='Median RUL at first alert (cycles)')
        fig.suptitle(f'{split}: persistence comparison at original P99')
        save_figure(fig, output / f'persistence_comparison_{split}.png')


def markdown_table(table, columns, digits=2):
    """Não depende do pacote opcional tabulate."""
    def fmt(value):
        if pd.isna(value):
            return 'N/A'
        if isinstance(value, (float, np.floating)):
            return f'{value:.{digits}f}'
        return str(value)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |' for _, row in table.iterrows()]
    return '\n'.join(lines)


def dataset_findings(tables):
    summary = tables['evaluation_summary']
    coverage = tables['rul_coverage']
    nominal = tables['nominal_alerts']
    lines = ['\n### Leitura dos resultados por dataset\n']
    for dataset, group in summary[summary.Persistence == 3].groupby('Dataset'):
        train, test = group[group.Split == 'train'], group[group.Split == 'test']
        near = coverage[(coverage.Dataset == dataset) & (coverage.Split == 'test') &
                        (coverage.Percentile == 99) & (coverage.Persistence == 3) & (coverage.RULGroup == 'RUL <= 30')]
        text = f'- **{dataset}:** '
        if not train.empty:
            row = train.iloc[0]
            text += (f'alerta em {int(row.detected_motors)}/{int(row.eligible_motors)} motores completos de treino '
                     f'({row.coverage_pct:.2f}%), com RUL mediano de {row.alert_rul_median:.1f} ciclos. ')
        if not test.empty:
            row = test.iloc[0]
            text += (f'No teste truncado, {int(row.detected_motors)}/{int(row.eligible_motors)} motores elegíveis '
                     f'({row.coverage_pct:.2f}%), com RUL mediano de {row.alert_rul_median:.1f} ciclos; '
                     f'{int(row.insufficient_motors)} motores sem janelas suficientes. ')
        if not near.empty:
            row = near.iloc[0]
            text += (f'Entre os motores com RUL final ≤30, a cobertura é {row.coverage_pct:.2f}% '
                     f'({int(row.detected_motors)}/{int(row.eligible_motors)}). ')
        n = nominal[(nominal.Dataset == dataset) & (nominal.Percentile == 99) &
                    (nominal.Persistence == 3) & (nominal.DevelopmentRole == 'all_train')]
        if not n.empty:
            row = n.iloc[0]
            text += (f'Há {int(row.motors_with_nominal_alert)} motores com alerta na região nominal assumida '
                     f'({row.nominal_motor_alert_pct:.2f}%).')
        lines.append(text)
    lines.append('\nA cobertura maior no grupo próximo da falha, quando comparada ao teste completo, '
                 'é compatível com um score sensível à degradação, mas a comparação entre grupos não demonstra '
                 'crescimento monotônico dentro de cada motor. A cobertura global menor no teste também reflete '
                 'o truncamento das trajetórias e não deve ser interpretada isoladamente como erro do detector. '
                 'Alertas muito antecipados podem refletir heterogeneidade entre motores ou desvios transitórios, '
                 'não necessariamente maior qualidade. Os gráficos individuais e as taxas nominais precisam ser considerados em conjunto.')
    return lines


def write_report(tables, configs):
    summary, coverage = tables['evaluation_summary'], tables['rul_coverage']
    baseline = summary[summary.Persistence == 3]
    nominal = tables['nominal_alerts']
    nominal = nominal[(nominal.Percentile == 99) & (nominal.Persistence == 3)]
    sections = [
        '# Avaliação quantitativa da detecção de anomalias\n',
        'Avaliação descritiva dos erros existentes. Nenhum modelo foi treinado, nenhum estudo Optuna foi executado '
        'e nenhuma configuração foi selecionada pelos resultados de teste. O baseline continua sendo P99 e 3 janelas consecutivas.\n',
        '## Cobertura e antecedência do alerta\n',
        'Cobertura não é acurácia, recall ou sensibilidade real de falhas: não há anotação do início da anomalia. '
        'O denominador é o número de motores com pelo menos k janelas consecutivas disponíveis. Motores insuficientes '
        'são contabilizados separadamente; a ausência de alerta é NaN, nunca ciclo zero. Quartis e mediana incluem apenas motores detectados.\n',
        markdown_table(baseline, ['Dataset', 'Split', 'eligible_motors', 'insufficient_motors', 'detected_motors',
                                 'coverage_pct', 'alert_rul_median', 'alert_rul_q25', 'alert_rul_q75']),
        '\nNo treino completo, o último ciclo observado é a falha (RUL final zero). No teste truncado, '
        'RUL no alerta = RUL final fornecido + último ciclo observado − ciclo do alerta. Isso mede distância até a falha, '
        'não distância até o início verdadeiro da degradação.\n',
        '### Motores de teste próximos da falha\n',
        'Os limites de RUL são cumulativos. No treino, todas as faixas coincidem porque o RUL final é zero.\n',
        markdown_table(coverage[(coverage.Percentile == 99) & (coverage.Persistence == 3) &
                               (coverage.Split == 'test') & (coverage.RULGroup == 'RUL <= 30')],
                       ['Dataset', 'RULGroup', 'eligible_motors', 'detected_motors', 'coverage_pct', 'alert_rul_median']),
        *dataset_findings(tables),
        '\n## Alertas na região nominal assumida\n',
        'Somente janelas inteiramente dentro dos primeiros 30% da trajetória completa de treino. '
        'O teste não recebe essa regra. A elegibilidade nominal exige ao menos uma janela completa; '
        'as tabelas também informam motores sem ciclos/janelas suficientes. A divisão fit_train/validation é recuperada '
        'dos artefatos, não inferida a partir dos erros.\n',
        markdown_table(nominal, ['Dataset', 'DevelopmentRole', 'eligible_motors', 'insufficient_nominal_cycles',
                                 'motors_with_nominal_alert', 'nominal_motor_alert_pct',
                                 'nominal_above_windows_pct', 'nominal_persistent_windows_pct']),
        '\nAlertas nessa região são candidatos a alertas prematuros sob a hipótese operacional de normalidade, '
        'não falsos positivos comprovados. fit_train é diagnóstico in-sample; validation também já participou '
        'da seleção de hiperparâmetros, EarlyStopping e definição do threshold. Nenhuma dessas taxas é uma estimativa '
        'de generalização em calibração independente.\n',
        '## Persistência: 3 versus 5 janelas\n',
        markdown_table(tables['persistence_comparison'], ['Dataset', 'Split', 'detected_3', 'detected_5',
                         'alerts_disappearing', 'alerts_later', 'median_delay_cycles_among_later']),
        '\nAumentar a persistência pode suprimir episódios curtos ou adiar o primeiro alerta. '
        'As janelas de 30 ciclos se sobrepõem em 29 ciclos; não são observações independentes. '
        'A tabela evaluation_summary compara cobertura e RUL mediano, e nominal_alerts compara as taxas nominais '
        'para ambas as regras. O denominador elegível pode mudar com k. Não há escolha automática de uma regra vencedora.\n',
        '## Sensibilidade ao threshold\n',
        'Os percentis adicionais usam exclusivamente os erros das janelas nominais de validação identificadas pelos '
        'metadados salvos. O cruzamento com o CSV é validado por hashes, separação de motores e reprodução do P99 original '
        '(tolerância relativa 1e-5). O valor original de P99 é preservado exatamente. '
        'Não se calculam percentis no teste nem no conjunto completo de treino.\n',
    ]
    details = tables['nominal_motor_metrics']
    if 'any_nominal_persistent_alert' in details:
        early = details[(details.Percentile == 99) & (details.Persistence == 3) & details.any_nominal_persistent_alert]
        descriptions = []
        for dataset, group in early.groupby('Dataset'):
            units = ', '.join(map(str, group.unit_nr.astype(int)))
            descriptions.append(f'- {dataset}: motores de treino {units}.')
        if descriptions:
            position = sections.index('## Persistência: 3 versus 5 janelas\n')
            sections[position:position] = ['### Motores com alerta nominal no baseline\n', *descriptions]
    # Expõe o contraste de treino/validação sem chamar a hipótese nominal de rótulo real.
    fd003 = nominal[(nominal.Dataset == 'FD003') & nominal.DevelopmentRole.isin(['fit_train', 'validation'])]
    if len(fd003) == 2:
        rates = fd003.set_index('DevelopmentRole').nominal_above_windows_pct
        position = sections.index('## Persistência: 3 versus 5 janelas\n')
        sections.insert(position, f'\nNo FD003, {rates["fit_train"]:.2f}% das janelas nominais de ajuste '
                        f'ultrapassam o threshold, contra {rates["validation"]:.2f}% na validação usada para calibrá-lo. '
                        'Isso merece investigar representatividade dos motores e adequação da hipótese nominal. '
                        'Não é, por si só, evidência de erro no cálculo ou vazamento; o P99 foi reproduzido e a divisão por motor conferida.\n')
    for config in configs:
        sections.append(f"- {config['Dataset']}: origem `{config['calibration_error_source']}`; "
                        f"percentis disponíveis: {', '.join(map(str, config['thresholds']))}.")
    sections += ['\nA sensibilidade é descritiva. Reutilizar a validação do desenvolvimento não a torna uma calibração independente.\n',
                 markdown_table(tables['threshold_sensitivity'][tables['threshold_sensitivity'].Split == 'test'],
                                ['Dataset', 'Percentile', 'Persistence', 'detected_motors', 'eligible_motors',
                                 'coverage_pct', 'nominal_motor_alert_pct']),
                 '\nA última coluna acima sempre se refere à região nominal do treino completo, não ao teste.\n',
                 '## Regimes operacionais e FD004, motor 1 de teste\n']
    for config in configs:
        dataset = config['Dataset']
        path = OUTPUT_DIR / dataset / 'regime_analysis.csv'
        if config['regime_analysis_executed'] and path.exists():
            regime = pd.read_csv(path)
            contrast = regime[regime.Dimension == 'transition_near_end']
            sections += [f'### {dataset}\n', markdown_table(contrast, ['Split', 'Group', 'windows', 'mean_error',
                            'median_error', 'above_threshold_pct', 'persistent_window_pct', 'mean_error_change'], digits=6)]
            inside = regime[(regime.Dimension == 'transition_inside') & regime.Group.astype(str).str.lower().eq('true')]
            for _, row in inside.iterrows():
                total = regime[(regime.Split == row.Split) & (regime.Dimension == 'transition_inside')].windows.sum()
                sections.append(f'\n{row.Split}: {int(row.windows)}/{int(total)} janelas contêm transição interna '
                                f'({percent(row.windows, total):.2f}%).')
    motors = tables['motor_metrics']
    unit1 = motors[(motors.Dataset == 'FD004') & (motors.Split == 'test') & (motors.unit_nr == 1) & (motors.Percentile == 99)]
    if not unit1.empty:
        sections += ['\nFD004, motor 1 do teste, mantendo P99:\n', markdown_table(unit1, ['Persistence', 'detection_cycle',
                          'alert_RUL', 'last_observed_cycle', 'last_RUL'])]
        near_path = OUTPUT_DIR / 'FD004' / 'fd004_test_unit1_cycles85_120.csv'
        if near_path.exists():
            near = pd.read_csv(near_path)
            at101 = near[near.end_cycle == 101]
            if not at101.empty:
                row = at101.iloc[0]
                sections.append(f'\nNo ciclo 101, o erro é {row.reconstruction_error:.6f}, '
                                f'acima do threshold: {row.above_threshold}; alerta persistente: {row.persistent_anomaly}; '
                                f'comprimento total do episódio acima do threshold: {row.above_episode_length} janelas. '
                                'O recorte de ciclos 85–120 está salvo em CSV para inspeção.')
                if row.above_episode_length == 3 and bool(row.persistent_anomaly):
                    sections.append('O episódio termina no ciclo 101 e tem exatamente o mínimo exigido pela regra de 3. '
                                    'Portanto, o alerta é persistente segundo a regra implementada, mas o episódio é curto '
                                    'e não constitui evidência de degradação sustentada. Sua supressão por 5 janelas '
                                    'ilustra a sensibilidade à regra, sem provar que 5 seja a escolha ideal.')
    sections += [
        '\nO regime é estimado pelo KMeans já salvo, sem ajuste novo. Regime da janela significa regime do último ciclo. '
        'Transição interna exige mudança após o primeiro ciclo e até o último; transição próxima ao final ocorre nos '
        'três ciclos finais. mean_error_change é a diferença de MSE em relação à janela anterior consecutiva. '
        'O perfil −3 a +3 ciclos pode contar janelas em vários eventos sobrepostos. Associações são descritivas, '
        'sem controle por motor, RUL ou dependência temporal: não comprovam causalidade. Quando quase todas as janelas '
        'contêm transições, o contraste transição versus ausência é limitado ou inexistente.\n',
        '## Auditoria, limitações e próxima etapa\n',
    ]
    for config in configs:
        mismatches = sum(config['original_results_audit'].values())
        sections.append(f"- {config['Dataset']}: {mismatches} divergências na reprodução dos indicadores originais.")
        sections.extend(f"  - {note}" for note in config['notes'])
    sections += [
        '\nNão houve modificação dos resultados originais, dos modelos ou do pré-processamento. '
        'Não há retreinamento nem integração ao Transformer nesta etapa. A investigação de regimes não se aplica a FD001/FD003.\n',
        'A hipótese nominal de 30% não é verdade-terreno. O erro agrega features normalizadas, inclusive condições '
        'operacionais quando presentes, e não corresponde apenas ao erro físico de sensores. Comparar magnitude bruta '
        'do MSE entre datasets não é uma escala universal de severidade. Os motores representativos dos gráficos são '
        '1, 2 e 3 em cada split; as métricas usam todos os motores disponíveis.\n',
        '**Recomendação:** os resultados permitem investigar o score como candidato a variável adicional, '
        'mas não comprovam ainda detecção do início da degradação ou valor preditivo incremental para RUL. '
        'Antes de afirmar validação independente, fazer uma segunda rodada com divisão por motor em ajuste, '
        'validação de desenvolvimento e calibração separada. O grupo de calibração deve ficar fora tanto do ajuste '
        'quanto da escolha de hiperparâmetros/EarlyStopping. Fixar threshold e persistência antes de abrir uma avaliação '
        'final independente; como este teste já foi examinado, tratar futuras escolhas informadas por ele como exploratórias. '
        'Se os motores disponíveis já influenciaram a seleção atual, apenas renomear a validação não resolve: '
        'a separação exige novo protocolo de desenvolvimento.\n',
        'Em uma etapa futura, a utilidade para o Transformer exigirá uma comparação controlada com/sem o score, '
        'preservando a separação por motor e o mesmo protocolo experimental. Esta comparação não foi executada.\n',
        '## Reprodutibilidade\n',
        '`python -m src.ml.evaluate_anomaly_detection --datasets FD001 FD002 FD003 FD004`\n\n'
        'Para atualizar somente esta avaliação, acrescentar `--overwrite`. Todas as figuras são PNG a 300 dpi; '
        'os CSVs e evaluation_config.json documentam configurações, denominadores, hashes e omissões. '
        'O arquivo comparison/evaluation_summary.csv reúne o baseline e a persistência 5.\n',
    ]
    (OUTPUT_DIR / 'evaluation_report.md').write_text('\n\n'.join(section.strip() for section in sections) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--overwrite', action='store_true', help='Atualiza somente os arquivos desta avaliação.')
    parser.add_argument('--recover-calibration-only', action='store_true',
                        help='Etapa separada: inferência nominal no modelo salvo, sem avaliar nem treinar.')
    args = parser.parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    if args.recover_calibration_only:
        for dataset in datasets:
            recover_calibration(dataset, OUTPUT_DIR / dataset)
        return
    # Evita iniciar parcialmente uma execução que sobrescreveria avaliações sem autorização.
    existing = [d for d in datasets if (OUTPUT_DIR / d / 'evaluation_config.json').exists()]
    if existing and not args.overwrite:
        parser.error(f'Avaliações existentes: {existing}. Use --overwrite para atualizá-las.')
    protected = [p for d in datasets for base in [SOURCE_DIR / d, PROCESSED_DIR / d] for p in base.rglob('*') if p.is_file()]
    protected += list((ROOT / 'models' / 'best_models_autoencoder').glob('*'))
    hashes_before = {str(p.relative_to(ROOT)): file_hash(p) for p in protected if p.is_file()}
    collected, configs = {}, []
    for dataset in datasets:
        print(f'Avaliando {dataset}...', flush=True)
        tables, config = evaluate_dataset(dataset, args.overwrite)
        configs.append(config)
        for name, table in tables.items():
            if name not in ['regime_windows', 'regime_analysis', 'transition_profile']:
                collected.setdefault(name, []).append(table)
    combined = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    comparison = OUTPUT_DIR / 'comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    for name, table in combined.items():
        table.to_csv(comparison / f'{name}.csv', index=False)
    plot_comparison(combined['rul_coverage'], combined['evaluation_summary'], comparison)
    write_report(combined, configs)
    changed = [name for name, digest in hashes_before.items() if file_hash(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f'Entradas protegidas foram alteradas durante a execução: {changed}')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    run = {'datasets': datasets, 'git_revision': revision, 'evaluation_script_sha256': file_hash(Path(__file__)),
           'protected_files_unchanged': True, 'protected_input_sha256': hashes_before,
           'comparison_scope': 'only datasets requested in this invocation',
           'software': {'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}}
    (OUTPUT_DIR / 'evaluation_config.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(f'Relatório: {OUTPUT_DIR / "evaluation_report.md"}', flush=True)


if __name__ == '__main__':
    main()
