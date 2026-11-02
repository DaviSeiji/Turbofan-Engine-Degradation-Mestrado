"""Prepara RUL v2 por motor. Não importa ou executa treinamento de redes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.paths import ROOT
from src.rul_prediction.v2.utils import (
    ARRAY_SPLITS, DATASETS, DEVELOPMENT_SPLITS, INPUT_COLUMNS, RAW_COLUMNS, WINDOW_SIZE,
    Config, RULPreprocessor, file_hash, output_target, software_versions,
    split_engines, validate_trajectories,
)

RAW_DIR = ROOT / 'data' / 'raw_data'
OUTPUT_DIR = ROOT / 'data' / 'processed_rul_v2'
META_COLUMNS = ['Dataset', 'Split', 'SourceSplit', 'unit_nr', 'start_cycle', 'end_cycle',
                'RUL_original', 'RUL_capped', 'regime', 'regime_transition_count',
                'is_padded', 'padding_cycles', 'observed_cycles']


def add_rul(data, source_split, rul_cap=125, final_rul=None):
    data = validate_trajectories(data)
    if not isinstance(rul_cap, int) or rul_cap <= 0:
        raise ValueError('Cap de RUL deve ser inteiro positivo.')
    last = data.groupby('unit_nr').time_cycles.transform('max')
    if source_split == 'train':
        if final_rul is not None:
            raise ValueError('Treino completo não recebe vetor externo de RUL.')
        original = last - data.time_cycles
    elif source_split == 'test':
        if final_rul is None or set(final_rul) != set(data.unit_nr.unique()):
            raise ValueError('Vetor RUL deve corresponder exatamente aos motores de teste.')
        values = np.array(list(final_rul.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError('RUL final deve conter ciclos inteiros, finitos e não negativos.')
        original = last + data.unit_nr.map(final_rul) - data.time_cycles
    else:
        raise ValueError('SourceSplit deve ser train ou test.')
    data['RUL_original'] = original.astype(np.int64)
    data['RUL_capped'] = np.minimum(data.RUL_original, rul_cap).astype(np.int64)
    return data


def read_sources(config, raw_dir):
    raw_dir = Path(raw_dir)
    frames = {}
    for split in ['train', 'test']:
        frame = pd.read_csv(raw_dir / f'{split}_{config.dataset}.txt', sep=r'\s+', header=None)
        if frame.shape[1] != len(RAW_COLUMNS):
            raise ValueError(f'{split}: esperadas exatamente {len(RAW_COLUMNS)} colunas.')
        frame.columns = RAW_COLUMNS
        frames[split] = frame
    rul = pd.read_csv(raw_dir / f'RUL_{config.dataset}.txt', sep=r'\s+', header=None)
    if rul.shape[1] != 1:
        raise ValueError('Arquivo RUL deve ter exatamente uma coluna.')
    final = dict(enumerate(rul.iloc[:, 0].tolist(), start=1))
    return (add_rul(frames['train'], 'train', config.rul_cap),
            add_rul(frames['test'], 'test', config.rul_cap, final))


def make_windows(transformed, features, dataset, split, terminal=False):
    """start/end são ciclos reais; padding nunca inventa ciclos negativos."""
    if dataset not in DATASETS or split not in ARRAY_SPLITS or terminal != (split == 'test_terminal'):
        raise ValueError('Split/protocolo de janelas inválido.')
    data = validate_trajectories(transformed)
    required = {'RUL_original', 'RUL_capped', 'regime'}
    if required - set(data):
        raise ValueError(f'Metadados ausentes: {sorted(required - set(data))}.')
    if (not features or len(features) != len(set(features))
            or set(features) - set(INPUT_COLUMNS) or set(features) - set(data)):
        raise ValueError('Lista de features inválida.')
    targets = data[['RUL_original', 'RUL_capped', 'regime']].to_numpy(dtype=float)
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError('RUL/regime não finito ou negativo.')
    windows, rows = [], []
    for unit, group in data.groupby('unit_nr', sort=True):
        values = group[features].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Features não finitas ou fora da capacidade de float32.')
        cycles = group.time_cycles.to_numpy()
        ends = [len(group) - 1] if terminal or len(group) < WINDOW_SIZE else range(WINDOW_SIZE - 1, len(group))
        for end in ends:
            start = max(0, end - WINDOW_SIZE + 1)
            sequence = values[start:end + 1]
            padding = WINDOW_SIZE - len(sequence)
            if padding:
                sequence = np.vstack([np.repeat(sequence[:1], padding, axis=0), sequence])
            windows.append(sequence)
            regimes = group.regime.iloc[start:end + 1].to_numpy()
            last = group.iloc[end]
            rows.append([dataset, split, 'train' if split in DEVELOPMENT_SPLITS else 'test',
                         int(unit), int(cycles[start]), int(cycles[end]),
                         int(last.RUL_original), int(last.RUL_capped), int(last.regime),
                         int(np.count_nonzero(regimes[1:] != regimes[:-1])),
                         bool(padding), padding, end - start + 1])
    metadata = pd.DataFrame(rows, columns=META_COLUMNS)
    X = np.stack(windows).astype(np.float32, copy=False)
    arrays = {'X': X, 'y': metadata.RUL_capped.to_numpy(dtype=np.float32),
              'y_original': metadata.RUL_original.to_numpy(dtype=np.int64),
              'features': np.asarray(features, dtype=str)}
    return arrays, metadata


def prepare_frames(train, test, config):
    """Treino/teste já têm RUL; a reserva só é transformada, nunca ajustada."""
    train, test = validate_trajectories(train), validate_trajectories(test)
    ids = split_engines(train.unit_nr.unique(), config)
    processor = RULPreprocessor(config)
    processor.fit(train[train.unit_nr.isin(ids['train'])])
    result = {}
    for split in DEVELOPMENT_SPLITS:
        frame = train[train.unit_nr.isin(ids[split])]
        result[split] = make_windows(processor.transform(frame), processor.features_, config.dataset, split)
    transformed_test = processor.transform(test)
    for split in ['test_terminal', 'test_windows']:
        result[split] = make_windows(transformed_test, processor.features_, config.dataset,
                                    split, terminal=split == 'test_terminal')
    ids['test'] = sorted(map(int, test.unit_nr.unique()))
    return processor, ids, result


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def process_dataset(config, raw_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    # Verifica destino antes de ler dados ou ajustar qualquer transformador.
    target = output_target(output_dir, raw_dir, config.dataset)
    inputs = [Path(raw_dir) / f'{name}_{config.dataset}.txt' for name in ['train', 'test', 'RUL']]
    input_hashes = {p.name: file_hash(p) for p in inputs}
    train, test = read_sources(config, raw_dir)
    processor, ids, datasets = prepare_frames(train, test, config)
    if any(file_hash(p) != input_hashes[p.name] for p in inputs):
        raise RuntimeError('Dados brutos mudaram durante a preparação; saída não publicada.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    # A pasta é exclusiva desta execução; manifesto por último indica conclusão.
    write_json(target / 'config.json', config.to_dict())
    write_json(target / 'splits.json', ids)
    write_json(target / 'features.json', {'selected': processor.features_, 'dropped': processor.dropped_features_})
    with (target / 'preprocessor.joblib').open('xb') as stream:
        joblib.dump(processor, stream)
    summaries = {}
    for split, (arrays, metadata) in datasets.items():
        with (target / f'{split}.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with (target / f'{split}_metadata.csv').open('x', encoding='utf-8', newline='') as stream:
            metadata.to_csv(stream, index=False)
        summaries[split] = {'windows': len(metadata), 'engines': metadata.unit_nr.nunique(),
                            'padded_windows': int(metadata.is_padded.sum()),
                            'normalized_values_below_zero': int(np.count_nonzero(arrays['X'] < 0)),
                            'normalized_values_above_one': int(np.count_nonzero(arrays['X'] > 1))}
    manifest = {'pipeline': 'rul_v2', 'schema_version': 1, 'Dataset': config.dataset,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'config': config.to_dict(), 'software': software_versions(),
                'input_sha256': input_hashes,
                'code_sha256': {name: file_hash(Path(__file__).parent / name) for name in ['preprocessing.py', 'utils.py']},
                'sha256': {p.name: file_hash(p) for p in sorted(target.iterdir())},
                'splits_summary': summaries}
    write_json(target / 'manifest.json', manifest)
    return target


def validate_saved_dataset(folder):
    """Só lê os artefatos v2; não abre dados brutos nem ajusta transformadores."""
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('pipeline') != 'rul_v2' or manifest.get('schema_version') != 1:
        raise ValueError('Manifesto não corresponde ao esquema RUL v2.')
    expected = {'config.json', 'splits.json', 'features.json', 'preprocessor.joblib'}
    expected |= {f'{split}{suffix}' for split in ARRAY_SPLITS for suffix in ['.npz', '_metadata.csv']}
    if set(manifest['sha256']) != expected:
        raise ValueError('Manifesto com conjunto de artefatos incompleto ou inesperado.')
    for name, digest in manifest['sha256'].items():
        if file_hash(folder / name) != digest:
            raise ValueError(f'Hash divergente: {name}.')
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config != manifest['config'] or config['dataset'] != manifest['Dataset']:
        raise ValueError('Configuração diverge do manifesto.')
    cfg = Config(**{name: config[name] for name in Config.__dataclass_fields__})
    if cfg.to_dict() != config:
        raise ValueError('Configuração contém protocolo incompatível com esta versão.')
    ids = json.loads((folder / 'splits.json').read_text(encoding='utf-8'))
    if set(ids) != set(DEVELOPMENT_SPLITS) | {'test'}:
        raise ValueError('Divisões de motores inválidas.')
    for name, values in ids.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f'IDs ausentes ou duplicados em {name}.')
    for i, a in enumerate(DEVELOPMENT_SPLITS):
        for b in DEVELOPMENT_SPLITS[i + 1:]:
            if set(ids[a]) & set(ids[b]):
                raise ValueError('Motor compartilhado entre divisões de desenvolvimento.')
    processor = joblib.load(folder / 'preprocessor.joblib')
    features = json.loads((folder / 'features.json').read_text(encoding='utf-8'))
    if (not isinstance(processor, RULPreprocessor) or not processor.fitted_
            or processor.config != cfg or processor.features_ != features['selected']
            or processor.dropped_features_ != features['dropped'] or processor.train_units_ != ids['train']):
        raise ValueError('Transformador incompatível com a configuração/treino.')
    summary = {}
    for split in ARRAY_SPLITS:
        metadata = pd.read_csv(folder / f'{split}_metadata.csv')
        if list(metadata.columns) != META_COLUMNS or metadata.empty:
            raise ValueError(f'{split}: esquema de metadados inválido.')
        source = 'train' if split in DEVELOPMENT_SPLITS else 'test'
        expected_ids = ids[split] if source == 'train' else ids['test']
        if (not metadata.Dataset.eq(cfg.dataset).all() or not metadata.Split.eq(split).all()
                or not metadata.SourceSplit.eq(source).all() or set(metadata.unit_nr) != set(expected_ids)
                or metadata.duplicated(['unit_nr', 'end_cycle']).any()):
            raise ValueError(f'{split}: identidade das janelas inválida.')
        with np.load(folder / f'{split}.npz', allow_pickle=False) as arrays:
            if (set(arrays.files) != {'X', 'y', 'y_original', 'features'}
                    or arrays['X'].shape != (len(metadata), WINDOW_SIZE, len(processor.features_))
                    or arrays['X'].dtype != np.float32 or not np.isfinite(arrays['X']).all()
                    or arrays['features'].tolist() != processor.features_
                    or not np.array_equal(arrays['y'], metadata.RUL_capped)
                    or not np.array_equal(arrays['y_original'], metadata.RUL_original)):
                raise ValueError(f'{split}: arrays e metadados não estão alinhados.')
        if not np.array_equal(metadata.RUL_capped, np.minimum(metadata.RUL_original, cfg.rul_cap)):
            raise ValueError(f'{split}: cap de RUL incorreto.')
        if (metadata.end_cycle - metadata.start_cycle + 1 != metadata.observed_cycles).any():
            raise ValueError(f'{split}: ciclos inválidos.')
        if ((metadata.observed_cycles + metadata.padding_cycles != WINDOW_SIZE).any()
                or not metadata.is_padded.eq(metadata.padding_cycles.gt(0)).all()):
            raise ValueError(f'{split}: padding inválido.')
        if split == 'test_terminal' and metadata.unit_nr.duplicated().any():
            raise ValueError('Teste terminal deve ter exatamente uma janela por motor.')
        summary[split] = {'engines': len(expected_ids), 'windows': len(metadata)}
    terminal = pd.read_csv(folder / 'test_terminal_metadata.csv')
    all_windows = pd.read_csv(folder / 'test_windows_metadata.csv')
    last = all_windows.groupby('unit_nr', sort=True).tail(1).reset_index(drop=True)
    compare_columns = [c for c in META_COLUMNS if c != 'Split']
    if not terminal[compare_columns].equals(last[compare_columns]):
        raise ValueError('Teste terminal não corresponde à última janela do teste complementar.')
    return processor, summary


def load_preprocessor(folder):
    """Carrega somente após conferir hashes e consistência dos artefatos locais."""
    return validate_saved_dataset(folder)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--train-fraction', type=float, default=.70)
    parser.add_argument('--validation-fraction', type=float, default=.15)
    parser.add_argument('--calibration-fraction', type=float, default=.15)
    parser.add_argument('--rul-cap', type=int, default=125)
    parser.add_argument('--constant-tolerance', type=float, default=1e-12)
    parser.add_argument('--validate-only', action='store_true', help='Só valida artefatos v2 existentes; não lê dados brutos.')
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Não repita datasets.')
    try:
        configs = [Config(d, args.seed, args.train_fraction, args.validation_fraction,
                          args.calibration_fraction, args.rul_cap, args.constant_tolerance) for d in args.datasets]
    except ValueError as exc:
        parser.error(str(exc))
    if not args.validate_only:
        # Confere todos os destinos antes de iniciar o primeiro dataset.
        for cfg in configs:
            output_target(args.output_dir, args.raw_dir, cfg.dataset)
    for cfg in configs:
        if args.validate_only:
            _, summary = validate_saved_dataset(args.output_dir / cfg.dataset)
            print(f'{cfg.dataset}: artefatos íntegros — {summary}')
        else:
            target = process_dataset(cfg, args.raw_dir, args.output_dir)
            print(f'{cfg.dataset}: RUL v2 salvo em {target}')


if __name__ == '__main__':
    main()
