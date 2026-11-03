"""Consolida exclusivamente CSVs/JSONs RUL v2 salvos; não importa redes ou Optuna."""

import argparse
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd

from src.common.paths import ROOT
from src.rul_prediction.v2.experiment import (
    DATA_ROOT, JOIN_KEYS, MODEL_ROOT, MODELS, PROTOCOL, RESULT_ROOT,
)
from src.rul_prediction.v2.utils import DATASETS, file_hash

METRICS = ['RMSE', 'MAE', 'MedianAE', 'P95AE', 'MaxAE', 'Bias']
BANDS = ['A_0_30', 'B_31_125', 'C_gt125']
GROUP = ['Dataset', 'Model', 'Protocol']
TOLERANCE = 1e-4  # CSV serializa previsões float32 com menos casas que os JSONs.


def scores(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if (truth.ndim != 1 or truth.shape != prediction.shape
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError('Vetores de RUL/previsão inválidos.')
    if not len(truth):
        return dict.fromkeys(METRICS, np.nan)
    residual = prediction - truth
    error = np.abs(residual)
    return dict(zip(METRICS, map(float, [np.sqrt(np.mean(residual**2)), np.mean(error),
                    np.median(error), np.percentile(error, 95), error.max(), residual.mean()])))


def rul_band(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError('RUL original deve ser inteiro, finito e não negativo.')
    return np.select([values <= 30, values <= 125], BANDS[:2], default=BANDS[2])


def aggregate(frame, keys, columns):
    if frame.empty:
        return pd.DataFrame(columns=keys + ['NSeeds', 'NNonEmptySeeds'] +
                            [f'{c}_{stat}' for c in columns for stat in ('mean', 'std')])
    grouped = frame.groupby(keys, sort=True, dropna=False)
    result = grouped[columns].agg(['mean', 'std'])
    result.columns = [f'{name}_{stat}' for name, stat in result.columns]
    result['NSeeds'] = grouped.Seed.nunique()
    if 'N' in frame:
        result['NNonEmptySeeds'] = frame.loc[frame.N.gt(0)].groupby(keys).Seed.nunique()
        result['NNonEmptySeeds'] = result.NNonEmptySeeds.fillna(0).astype(int)
    return result.reset_index()


def band_statistics(frame):
    labels = rul_band(frame.RUL_true)
    records = []
    for band in BANDS:
        part = frame.loc[labels == band]
        records.append({'Band': band, 'N': len(part), 'NEngines': part.unit_nr.nunique(),
                        **scores(part.RUL_true, part.RUL_pred)})
    if sum(row['N'] for row in records) != len(frame):
        raise ValueError('Faixas não preservaram todas as observações.')
    return records


def cap_contribution(frame):
    residual = frame.RUL_pred.to_numpy() - frame.RUL_true.to_numpy()
    high = frame.RUL_true.to_numpy() > 125
    sse, sae = np.sum(residual**2), np.sum(np.abs(residual))
    return {'N': len(frame), 'NHigh': int(high.sum()), 'HighFraction': float(high.mean()),
            'SSE': float(sse), 'SSEHigh': float(np.sum(residual[high]**2)),
            'SSEHighShare': float(np.sum(residual[high]**2) / sse) if sse else 0.,
            'SAEHighShare': float(np.sum(np.abs(residual[high])) / sae) if sae else 0.}


def select_engines(units):
    """Primeiro, central inferior e último ID; nunca consulta erros ou previsões."""
    ids = sorted(set(map(int, units)))
    return sorted({ids[0], ids[(len(ids) - 1) // 2], ids[-1]}) if ids else []


class SavedResults:
    """Leitor somente leitura, com registro dos hashes de todos os insumos usados."""

    def __init__(self, result_root, model_root, data_root, experiment):
        self.result_root, self.model_root, self.data_root = map(Path, (result_root, model_root, data_root))
        self.experiment = experiment
        self.inputs, self.cache = {}, {}

    def track(self, path, digest=None):
        path = Path(path)
        actual = file_hash(path)
        if digest is not None and actual != digest:
            raise ValueError(f'Hash divergente: {path}.')
        key = str(path.resolve())
        if key in self.inputs and self.inputs[key] != actual:
            raise ValueError(f'Arquivo alterado durante a análise: {path}.')
        self.inputs[key] = actual

    def read(self, path, digest=None):
        self.track(path, digest)
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def manifest(self, folder, names):
        manifest = self.read(folder / 'manifest.json')
        if manifest['protocol'] != PROTOCOL or set(manifest['sha256']) != set(names):
            raise ValueError(f'Manifesto incompatível: {folder}.')
        for name in names:
            self.track(folder / name, manifest['sha256'][name])
        return manifest

    def metadata(self, dataset, protocol):
        key = (dataset, protocol)
        if key not in self.cache:
            folder = self.data_root / dataset
            manifest = self.read(folder / 'manifest.json')
            config = self.read(folder / 'config.json', manifest['sha256']['config.json'])
            splits = self.read(folder / 'splits.json', manifest['sha256']['splits.json'])
            features = self.read(folder / 'features.json', manifest['sha256']['features.json'])
            if (manifest['pipeline'] != 'rul_v2' or manifest['schema_version'] != 1
                    or manifest['Dataset'] != dataset or config != manifest['config']
                    or config['dataset'] != dataset or config['rul_cap'] != 125 or config['window_size'] != 30):
                raise ValueError('Identidade do pré-processamento incompatível.')
            dev = [set(splits[s]) for s in ('train', 'validation', 'calibration')]
            if any(dev[i] & dev[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError('Motores compartilhados entre divisões de desenvolvimento.')
            name = f'test_{protocol}_metadata.csv'
            self.track(folder / name, manifest['sha256'][name])
            meta = pd.read_csv(folder / name)
            if (meta.empty or set(meta.unit_nr) != set(splits['test'])
                    or meta.duplicated(JOIN_KEYS).any()
                    or not meta.Dataset.eq(dataset).all() or not meta.SourceSplit.eq('test').all()
                    or not meta.Split.eq('test_' + protocol).all()):
                raise ValueError('Coorte/metadados de teste incompatíveis.')
            self.cache[key] = (manifest, config, splits, features, meta)
        return self.cache[key]

    def load(self, dataset, model, seed, protocol):
        folder = self.result_root / dataset / model / self.experiment / f'seed_{seed}' / protocol
        manifest = self.manifest(folder, ('metrics.json', 'predictions.csv', 'predictions.png'))
        record = self.read(folder / 'metrics.json')
        if (record != manifest['record'] or record['Dataset'] != dataset or record['Model'] != model
                or record['Seed'] != seed or record['Experiment'] != self.experiment
                or record['Protocol'] != protocol or record['Target'] != 'y_original_uncapped'
                or record['TrainingProtocol']['protocol'] != PROTOCOL):
            raise ValueError('Identidade das métricas incompatível.')
        _, preconfig, splits, features, meta = self.metadata(dataset, protocol)
        digest = self.inputs[str((self.data_root / dataset / 'manifest.json').resolve())]
        if record['PreprocessingManifest'] != digest:
            raise ValueError('Avaliação e dados processados pertencem a experimentos diferentes.')
        source = self.model_root / dataset / model / self.experiment / f'seed_{seed}'
        model_manifest = self.manifest(source, ('model.keras', 'config.json', 'history.json', 'history.png'))
        config = self.read(source / 'config.json')
        if (model_manifest['seed'] != seed or model_manifest['model'] != model
                or model_manifest['experiment'] != self.experiment or config['seed'] != seed
                or config['model'] != model or config['experiment'] != self.experiment
                or config['data'] != model_manifest['data'] or config['data']['Dataset'] != dataset
                or config['data']['manifest_sha256'] != digest
                or config['data']['preprocessing_config'] != preconfig
                or config['data']['features'] != features['selected']
                or config['data']['input_shape'] != [30, len(features['selected'])]
                or record['ModelSHA256'] != model_manifest['sha256']['model.keras']):
            raise ValueError('Modelo/configuração e previsão não correspondem.')
        for name, value in record['TrainingProtocol'].items():
            if config.get(name) != value:
                raise ValueError(f'Protocolo de treinamento divergente: {name}.')
        if config['training_target'] != 'y_capped_125':
            raise ValueError('Target de treinamento diferente de cap 125.')
        frame = pd.read_csv(folder / 'predictions.csv')
        if (frame.empty or frame.duplicated(JOIN_KEYS).any()
                or not frame.Model.eq(model).all() or not frame.Seed.eq(seed).all()
                or not frame.Experiment.eq(self.experiment).all()
                or len(frame) != record['N'] or set(frame.unit_nr) != set(splits['test'])):
            raise ValueError('Identidade, tamanho ou duplicação nas previsões.')
        numeric = frame[JOIN_KEYS[-3:] + ['RUL_true', 'RUL_pred', 'AbsoluteError']].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError('Valores não finitos nas previsões.')
        rul_band(frame.RUL_true)
        keys = JOIN_KEYS
        ordered = frame.sort_values(keys).reset_index(drop=True)
        expected = meta.sort_values(keys).reset_index(drop=True)
        if (len(ordered) != len(expected) or not ordered[keys].equals(expected[keys])
                or not np.array_equal(ordered.RUL_true, expected.RUL_original)
                or not np.array_equal(ordered.RUL_original, expected.RUL_original)
                or not np.array_equal(ordered.RUL_capped, np.minimum(ordered.RUL_true, 125))):
            raise ValueError('Desalinhamento de janelas, motores, ciclos ou target.')
        if protocol == 'terminal' and not ordered.unit_nr.is_unique:
            raise ValueError('Terminal exige uma previsão por motor.')
        for _, engine in ordered.groupby('unit_nr'):
            if ((engine.end_cycle + engine.RUL_true).nunique() != 1
                    or not engine.end_cycle.is_monotonic_increasing):
                raise ValueError('Alinhamento temporal inválido.')
        recalculated = scores(ordered.RUL_true, ordered.RUL_pred)
        if not np.allclose(ordered.AbsoluteError, np.abs(ordered.RUL_pred - ordered.RUL_true), atol=TOLERANCE, rtol=0):
            raise ValueError('Erro absoluto salvo não corresponde à previsão.')
        if any(not np.isclose(recalculated[m], record[m], atol=TOLERANCE, rtol=0) for m in METRICS[:-1]):
            raise ValueError('Métricas salvas não correspondem às previsões.')
        columns = JOIN_KEYS + ['Model', 'Seed', 'Experiment', 'RUL_true', 'RUL_pred']
        return ordered[columns], config, recalculated


def ensemble(frame):
    """Dispersão das previsões no mesmo motor/ciclo, não incerteza estatística."""
    keys = ['Dataset', 'Model', 'Protocol'] + JOIN_KEYS[1:]
    if frame.duplicated(keys + ['Seed']).any():
        raise ValueError('Previsões duplicadas na agregação temporal.')
    if (frame.groupby(keys).RUL_true.nunique() != 1).any():
        raise ValueError('Targets diferentes para a mesma janela.')
    return frame.groupby(keys, sort=True).agg(RUL_true=('RUL_true', 'first'),
               PredictionMean=('RUL_pred', 'mean'), PredictionSD=('RUL_pred', 'std'),
               NSeeds=('Seed', 'nunique')).reset_index()


def write_table(frame, folder, name):
    frame.to_csv(folder / f'{name}.csv', index=False, mode='x', float_format='%.10g')


def make_figures(tables, folder, datasets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'cnn_lstm': '#276aab', 'transformer': '#dc772a'}
    names = {'cnn_lstm': 'CNN-LSTM', 'transformer': 'Transformer'}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / f'{name}.png', dpi=160)
        plt.close(fig)

    for protocol in ('terminal', 'windows'):
        summary = tables[f'{protocol}_summary']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ('RMSE', 'MAE')):
            for i, model in enumerate(MODELS):
                part = summary.loc[summary.Model.eq(model)].set_index('Dataset').reindex(datasets)
                ax.bar(np.arange(len(datasets)) + (i - .5) * .32, part[f'{metric}_mean'], .32,
                       yerr=part[f'{metric}_std'].fillna(0), label=names[model], color=colors[model], capsize=3)
            ax.set_xticks(np.arange(len(datasets)), datasets)
            ax.set(ylabel=f'{metric} (cycles)', title=f'{protocol}: uncapped target')
            ax.legend(fontsize=8)
        save(fig, f'{protocol}_metrics')

    band = tables['bands_summary'].query("Protocol == 'windows'")
    for metric in ('RMSE', 'Bias'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, dataset in zip(axes.flat, datasets):
            for i, model in enumerate(MODELS):
                part = band.loc[band.Dataset.eq(dataset) & band.Model.eq(model)].set_index('Band').reindex(BANDS)
                ax.bar(np.arange(3) + (i - .5) * .32, part[f'{metric}_mean'], .32,
                       yerr=part[f'{metric}_std'].fillna(0), label=names[model], color=colors[model], capsize=3)
            ax.set_xticks(range(3), ['0–30', '31–125', '>125'])
            ax.axhline(0, color='black', linewidth=.6)
            ax.set(title=dataset, xlabel='True RUL band', ylabel=f'{metric} (cycles)')
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(datasets):]:
            ax.set_visible(False)
        save(fig, 'bands_' + metric.lower())

    individual = tables['metrics_individual']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, protocol in enumerate(('terminal', 'windows')):
        for col, metric in enumerate(('RMSE', 'MAE')):
            ax = axes[row, col]
            for i, model in enumerate(MODELS):
                for x, dataset in enumerate(datasets):
                    values = individual.loc[individual.Dataset.eq(dataset) & individual.Model.eq(model)
                                            & individual.Protocol.eq(protocol)].sort_values('Seed')[metric]
                    ax.scatter(x + (i - .5) * .25 + np.linspace(-.035, .035, len(values)), values,
                               color=colors[model], label=names[model] if x == 0 else None, s=22)
            ax.set_xticks(range(len(datasets)), datasets)
            ax.set(title=f'{protocol}: all seeds', ylabel=metric)
            ax.legend(fontsize=8)
    save(fig, 'seed_stability')

    caps = tables['cap_complement_summary'].query("Protocol == 'windows'")
    original = tables['windows_summary']
    contribution = tables['cap_contribution_summary'].query("Protocol == 'windows'")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    labels, uncapped, capped, shares = [], [], [], []
    for dataset in datasets:
        for model in MODELS:
            mask = lambda f: f.Dataset.eq(dataset) & f.Model.eq(model)
            a, b, c = original.loc[mask(original)], caps.loc[mask(caps)], contribution.loc[mask(contribution)]
            if a.empty or b.empty or c.empty:
                continue
            labels.append(f'{dataset}\n{names[model]}')
            uncapped.append(a.RMSE_mean.iloc[0]); capped.append(b.RMSE_mean.iloc[0])
            shares.append(c.SSEHighShare_mean.iloc[0] * 100)
    x = np.arange(len(labels))
    axes[0].bar(x - .18, uncapped, .36, label='Original target (main)')
    axes[0].bar(x + .18, capped, .36, label='Capped target (complement)')
    axes[0].set(ylabel='Window RMSE', title='Same predictions, DIFFERENT targets')
    axes[0].legend(fontsize=8)
    axes[1].bar(x, shares, color='#637b64')
    axes[1].set(ylabel='% of total squared error', title='Contribution of true RUL > 125', ylim=(0, 100))
    for ax in axes:
        ax.set_xticks(x, labels, rotation=60, fontsize=8)
    save(fig, 'cap_effect')

    engine = tables['per_engine_summary'].query("Protocol == 'windows'")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(MODELS):
        for x, dataset in enumerate(datasets):
            values = engine.loc[engine.Dataset.eq(dataset) & engine.Model.eq(model), 'MAE_mean']
            if len(values):
                boxes = ax.boxplot([values], positions=[x + (i - .5) * .28], widths=.23,
                                   patch_artist=True, manage_ticks=False)
                boxes['boxes'][0].set_facecolor(colors[model])
    for model in MODELS:
        ax.plot([], [], color=colors[model], label=names[model], linewidth=8)
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set(ylabel='Window MAE per engine (mean across seeds)', title='Error distribution BETWEEN engines')
    ax.legend(fontsize=8)
    save(fig, 'engine_error_distribution')

    dispersion = tables['prediction_dispersion_summary'].query("Protocol == 'windows'")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, model in enumerate(MODELS):
        part = dispersion.loc[dispersion.Model.eq(model)].set_index('Dataset').reindex(datasets)
        ax.bar(np.arange(len(datasets)) + (i - .5) * .32, part.MeanPredictionSD, .32,
               label=names[model], color=colors[model])
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set(ylabel='Mean prediction SD across seeds (cycles)', title='Same engine/window: initialization variability')
    ax.legend()
    save(fig, 'prediction_dispersion')

    trajectories = tables['trajectory_predictions']
    selected = tables['selected_engines']
    for dataset in datasets:
        units = selected.loc[selected.Dataset.eq(dataset), 'unit_nr'].tolist()
        if not units:
            continue
        fig, axes = plt.subplots(len(units), 1, figsize=(11, 3.3 * len(units)), squeeze=False)
        for ax, unit in zip(axes.flat, units):
            part = trajectories.loc[trajectories.Dataset.eq(dataset) & trajectories.unit_nr.eq(unit)]
            reference = part.drop_duplicates('end_cycle').sort_values('end_cycle')
            ax.plot(reference.end_cycle, reference.RUL_true, color='black', label='True RUL (uncapped)')
            ax.axhline(125, color='gray', linestyle='--', linewidth=.8, label='Training cap = 125')
            for model in MODELS:
                values = part.loc[part.Model.eq(model)].sort_values('end_cycle')
                if values.empty:
                    continue
                x, y, sd = values.end_cycle.to_numpy(), values.PredictionMean.to_numpy(), values.PredictionSD.to_numpy()
                ax.plot(x, y, color=colors[model], label=names[model])
                ax.fill_between(x, y - sd, y + sd, color=colors[model], alpha=.18)
            ax.set(title=f'{dataset}, engine {unit}: mean ± sample SD across seeds',
                   xlabel='Observed cycle (window end)', ylabel='RUL (cycles)')
            ax.legend(fontsize=8, ncol=2)
        save(fig, f'trajectories_{dataset.lower()}')


def analyze(result_root=RESULT_ROOT, model_root=MODEL_ROOT, data_root=DATA_ROOT,
            datasets=DATASETS, seeds=(1, 2, 3, 4, 5), experiment='baseline', name='baseline_a_v1'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Nome da análise inválido.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', experiment):
        raise ValueError('Nome do experimento inválido.')
    if (not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS)
            or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError('Datasets/seeds inválidos.')
    output = Path(result_root) / 'analysis' / name
    if output.exists():
        raise FileExistsError(f'Análise já existe e será preservada: {output}. Use outro --name.')
    reader = SavedResults(result_root, model_root, data_root, experiment)
    records, bands, capped, contributions, engines, integrity, frames, protocols = [], [], [], [], [], [], [], []
    references = {}
    for dataset in datasets:
        for model in MODELS:
            for seed in seeds:
                for protocol in ('terminal', 'windows'):
                    key = {'Dataset': dataset, 'Model': model, 'Seed': seed, 'Protocol': protocol}
                    try:
                        frame, config, calculated = reader.load(dataset, model, seed, protocol)
                        reference = frame[JOIN_KEYS + ['RUL_true']]
                        shared = (dataset, protocol)
                        if shared in references and not reference.equals(references[shared]):
                            raise ValueError('Janelas/targets diferentes entre arquiteturas/seeds.')
                        references[shared] = reference
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        integrity.append({**key, 'Status': 'missing_or_inconsistent', 'Detail': str(error)})
                        continue
                    integrity.append({**key, 'Status': 'ok', 'Detail': ''})
                    row = {**key, 'N': len(frame), 'NEngines': frame.unit_nr.nunique()}
                    records.append({**row, **calculated})
                    bands += [{**key, **b} for b in band_statistics(frame)]
                    capped.append({**row, 'Target': 'capped125_complement_no_prediction_clipping',
                                   **scores(np.minimum(frame.RUL_true, 125), frame.RUL_pred)})
                    contributions.append({**key, **cap_contribution(frame)})
                    unit_scores = []
                    for unit, part in frame.groupby('unit_nr', sort=True):
                        calculated_unit = scores(part.RUL_true, part.RUL_pred)
                        unit_scores.append(calculated_unit)
                        engines.append({**key, 'unit_nr': unit, 'N': len(part), **calculated_unit})
                    records[-1]['EngineBalancedMAE'] = float(np.mean([r['MAE'] for r in unit_scores]))
                    records[-1]['EngineBalancedRMSE'] = float(np.sqrt(np.mean([r['RMSE']**2 for r in unit_scores])))
                    frame['Protocol'] = protocol
                    frames.append(frame)
                    protocols.append({**key, 'NFeatures': len(config['data']['features']),
                                      'SamplerSeed': config['sampler_seed'], 'TotalTrials': config['total_trials'],
                                      'Epochs': config['epochs'], 'Patience': config['patience'],
                                      'FinalEpochs': config['final_epochs'], 'FinalPatience': config['final_patience'],
                                      'BestTrial': config['best']['trial'],
                                      'BestValidationMSE': config['best']['validation_mse'],
                                      'Parameters': json.dumps(config['best']['params'], sort_keys=True),
                                      'PreprocessingManifest': config['data']['manifest_sha256']})
    tables = {'integrity': pd.DataFrame(integrity), 'metrics_individual': pd.DataFrame(records),
              'bands_individual': pd.DataFrame(bands), 'cap_complement_individual': pd.DataFrame(capped),
              'cap_contribution_individual': pd.DataFrame(contributions),
              'per_engine_individual': pd.DataFrame(engines), 'protocols': pd.DataFrame(protocols)}
    if records:
        individual = tables['metrics_individual']
        for protocol in ('terminal', 'windows'):
            tables[protocol + '_summary'] = aggregate(individual.loc[individual.Protocol.eq(protocol)], GROUP,
                                                      ['N', 'NEngines', 'EngineBalancedMAE', 'EngineBalancedRMSE'] + METRICS)
        tables['bands_summary'] = aggregate(tables['bands_individual'], GROUP + ['Band'], ['N', 'NEngines'] + METRICS)
        tables['cap_complement_summary'] = aggregate(tables['cap_complement_individual'], GROUP,
                                                    ['N', 'NEngines'] + METRICS)
        tables['cap_contribution_summary'] = aggregate(tables['cap_contribution_individual'], GROUP,
                ['N', 'NHigh', 'HighFraction', 'SSE', 'SSEHigh', 'SSEHighShare', 'SAEHighShare'])
        tables['per_engine_summary'] = aggregate(tables['per_engine_individual'], GROUP + ['unit_nr'], ['N'] + METRICS)
        all_predictions = pd.concat(frames, ignore_index=True)
        combined = ensemble(all_predictions)
        dispersion = []
        for key, part in combined.groupby(GROUP, sort=True):
            sd = part.PredictionSD.dropna()
            dispersion.append(dict(zip(GROUP, key)) | {'N': len(part), 'NWithSD': len(sd),
                              'MeanPredictionSD': sd.mean(), 'P95PredictionSD': sd.quantile(.95),
                              'MaxPredictionSD': sd.max(), 'MinSeeds': int(part.NSeeds.min())})
        tables['prediction_dispersion_summary'] = pd.DataFrame(dispersion)
        selected, trajectories, cohorts = [], [], []
        for dataset in datasets:
            part = combined.loc[combined.Dataset.eq(dataset) & combined.Protocol.eq('windows')]
            units = select_engines(part.unit_nr)
            selected += [{'Dataset': dataset, 'unit_nr': u, 'Selection': 'sorted_ID_first_lower_middle_last'} for u in units]
            trajectories.append(part.loc[part.unit_nr.isin(units)])
        for (dataset, protocol), part in references.items():
            counts = part.groupby('unit_nr').size()
            cohorts.append({'Dataset': dataset, 'Protocol': protocol, 'N': len(part),
                            'NEngines': len(counts), 'MinWindowsPerEngine': int(counts.min()),
                            'MaxWindowsPerEngine': int(counts.max()), 'WeightRatio': float(counts.max() / counts.min()),
                            'MinRUL': float(part.RUL_true.min()), 'MaxRUL': float(part.RUL_true.max()),
                            'HighRULFraction': float(part.RUL_true.gt(125).mean())})
        tables['selected_engines'] = pd.DataFrame(selected, columns=['Dataset', 'unit_nr', 'Selection'])
        tables['trajectory_predictions'] = pd.concat(trajectories, ignore_index=True)
        tables['cohorts'] = pd.DataFrame(cohorts)
    # Não publica resultados se os insumos mudaram enquanto eram analisados.
    for path, digest in list(reader.inputs.items()):
        reader.track(path, digest)
    output.mkdir(parents=True, exist_ok=False)
    table_dir, figure_dir = output / 'tables', output / 'figures'
    table_dir.mkdir(); figure_dir.mkdir()
    for label, table in tables.items():
        write_table(table, table_dir, label)
    if records:
        make_figures(tables, figure_dir, list(datasets))
    summary = {'analysis': 'rul_v2_saved_results_stage_a_1', 'experiment': experiment,
               'datasets': list(datasets), 'models': list(MODELS), 'seeds': list(seeds),
               'expected_evaluations': len(datasets) * len(MODELS) * len(seeds) * 2,
               'validated_evaluations': len(records),
               'issues': [r for r in integrity if r['Status'] != 'ok'],
               'status': 'complete' if all(r['Status'] == 'ok' for r in integrity) else 'partial',
               'metric_absolute_tolerance': TOLERANCE, 'std_ddof': 1,
               'target_main': 'y_original_uncapped', 'prediction_clipping': False,
               'trajectory_selection': 'sorted_ID_first_lower_middle_last_before_inspecting_errors',
               'inputs_sha256': reader.inputs, 'analysis_code_sha256': file_hash(__file__),
               'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
               'tables': sorted(p.name for p in table_dir.iterdir()),
               'figures': sorted(p.name for p in figure_dir.iterdir())}
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'{output}: {len(records)}/{summary["expected_evaluations"]} avaliações válidas; '
          f'{len(summary["issues"])} inconsistências; {len(summary["figures"])} gráficos.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--experiment', default='baseline')
    parser.add_argument('--name', default='baseline_a_v1', help='Nome novo; nunca sobrescreve outra análise.')
    args = parser.parse_args()
    analyze(datasets=args.datasets, seeds=args.seeds, experiment=args.experiment, name=args.name)


if __name__ == '__main__':
    main()
