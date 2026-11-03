"""Prepara somente janelas nominais AE dentro de RUL train; sem redes ou scores."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection.v2.utils import (
    Config, CoordinatedNominalPreprocessor, OUTPUT_DIR, PIPELINE, RAW_DIR, ROLES, RUL_DIR,
    load_rul_reference, output_target, read_json, split_inside_rul_train, write_json,
)
from src.rul_prediction.v2.utils import (
    DATASETS, INPUT_COLUMNS, RAW_COLUMNS, SETTINGS, file_hash, software_versions, validate_trajectories,
)

PHYSICAL_KEYS = ['Dataset', 'SourceSplit', 'unit_nr', 'start_cycle', 'end_cycle']
META_COLUMNS = ['Dataset', 'SourceSplit', 'Split', 'AERole', 'unit_nr', 'start_cycle',
                'end_cycle', 'observed_cycles', 'padding_cycles', 'is_padded']
UNIT_COLUMNS = ['Dataset', 'SourceSplit', 'Split', 'AERole', 'unit_nr', 'total_cycles',
                'nominal_cycles', 'eligible', 'nominal_windows']
ARTIFACTS = {'config.json', 'splits_ae.json', 'features.json', 'preprocessor.joblib', 'nominal_units.csv'}
ARTIFACTS |= {f'nominal_{suffix}{ext}' for suffix in ('fit', 'inner_validation')
              for ext in ('.npz', '_metadata.csv')}


def role_of(unit, splits):
    for role in ROLES:
        if unit in splits[role]:
            return role
    return 'not_used_for_ae_development'


def nominal_metadata(units, dataset, role):
    rows = []
    for unit in units.loc[units.AERole.eq(role) & units.eligible].itertuples():
        for end in range(30, unit.nominal_cycles + 1):
            rows.append([dataset, 'train', 'train', role, unit.unit_nr, end - 29, end, 30, 0, False])
    return pd.DataFrame(rows, columns=META_COLUMNS)


def prepare_aligned_windows(raw, rul_metadata, processor, *, allow_test=False, allow_calibration=False):
    """Contrato futuro de scoring: transforma sensores brutos, nunca executa o AE."""
    if not isinstance(processor, CoordinatedNominalPreprocessor) or not processor.fitted_:
        raise ValueError('Use o pré-processador AE coordenado congelado.')
    required = set(META_COLUMNS) - {'AERole'}
    if not required <= set(rul_metadata) or rul_metadata.empty:
        raise ValueError('Metadados RUL ausentes/incompletos.')
    meta = rul_metadata[[c for c in META_COLUMNS if c != 'AERole']].copy().reset_index(drop=True)
    if not meta.Dataset.eq(processor.config.dataset).all() or meta.Split.nunique() != 1:
        raise ValueError('Alinhe um dataset e um protocolo RUL por chamada.')
    split = meta.Split.iloc[0]
    if split not in ('train', 'validation', 'calibration', 'test_windows', 'test_terminal'):
        raise ValueError('Papel RUL inválido.')
    if split.startswith('test_') and not allow_test:
        raise ValueError('Teste exige allow_test=True, somente no estágio futuro explicitamente autorizado.')
    if split == 'calibration' and not allow_calibration:
        raise ValueError('Reserva exige allow_calibration=True; não é usada automaticamente.')
    source = 'test' if split.startswith('test_') else 'train'
    ids = processor.splits['rul_splits']['test' if source == 'test' else split]
    if (not meta.SourceSplit.eq(source).all() or not set(meta.unit_nr) <= set(ids)
            or meta.duplicated(PHYSICAL_KEYS).any() or meta.duplicated(['unit_nr', 'end_cycle']).any()):
        raise ValueError('Namespace, papel dos motores ou chaves inválidos.')
    for col in ['unit_nr', 'start_cycle', 'end_cycle', 'observed_cycles', 'padding_cycles']:
        values = meta[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError('Ciclos/metadados devem ser inteiros finitos.')
        meta[col] = values.astype(np.int64)
    if (not meta.is_padded.isin([True, False]).all() or (meta.start_cycle < 1).any()
            or (meta.observed_cycles < 1).any() or (meta.observed_cycles > 30).any()
            or (meta.padding_cycles < 0).any()
            or (meta.end_cycle - meta.start_cycle + 1 != meta.observed_cycles).any()
            or (meta.observed_cycles + meta.padding_cycles != 30).any()
            or not meta.is_padded.eq(meta.padding_cycles.gt(0)).all()
            or (meta.loc[meta.is_padded, 'start_cycle'] != 1).any()):
        raise ValueError('Janela/padding incompatível com a política RUL v2.')
    if split == 'test_terminal' and not meta.unit_nr.is_unique:
        raise ValueError('Terminal requer uma janela por motor.')
    if not set(RAW_COLUMNS + ['Dataset', 'SourceSplit']) <= set(raw):
        raise ValueError('Forneça sensores brutos com Dataset e SourceSplit explícitos, não X normalizado do RUL.')
    if not raw.Dataset.eq(processor.config.dataset).all() or not raw.SourceSplit.eq(source).all():
        raise ValueError('A origem física dos sensores não corresponde aos metadados.')
    # Descarta linhas posteriores ao último final solicitado ANTES de validar/transformar.
    last = meta.groupby('unit_nr').end_cycle.max()
    selected = raw.loc[raw.unit_nr.isin(last.index) & raw.time_cycles.le(raw.unit_nr.map(last)), RAW_COLUMNS]
    if set(selected.unit_nr) != set(last.index):
        raise ValueError('Sensores ausentes para motores solicitados.')
    transformed = processor.transform(selected)
    groups = {int(unit): group.set_index('time_cycles') for unit, group in transformed.groupby('unit_nr')}
    windows = []
    for row in meta.itertuples():
        group = groups[row.unit_nr]
        cycles = np.arange(row.start_cycle, row.end_cycle + 1)
        if not set(cycles) <= set(group.index):
            raise ValueError('Ciclos brutos ausentes na janela solicitada.')
        sequence = group.loc[cycles, processor.features_].to_numpy(dtype=np.float32)
        if row.padding_cycles:
            sequence = np.vstack([np.repeat(sequence[:1], row.padding_cycles, axis=0), sequence])
        windows.append(sequence)
    X = np.stack(windows).astype(np.float32, copy=False)
    if not np.isfinite(X).all() or X.shape != (len(meta), 30, len(processor.features_)):
        raise ValueError('Janelas AE não finitas ou shape incompatível.')
    meta['AERole'] = [role_of(int(u), processor.splits) if source == 'train'
                      else 'not_used_for_ae_development' for u in meta.unit_nr]
    return {'X': X, 'features': np.asarray(processor.features_, dtype=str)}, meta[META_COLUMNS]


def prepare_frames(raw_train, config, rul_splits, reference):
    """Separa motores antes do ajuste e não transforma validação/reserva RUL."""
    if (('Dataset' in raw_train and not raw_train.Dataset.eq(config.dataset).all())
            or ('SourceSplit' in raw_train and not raw_train.SourceSplit.eq('train').all())):
        raise ValueError('A preparação nominal recebe somente o arquivo original de treino.')
    splits = split_inside_rul_train(rul_splits, config)
    expected = set().union(*(set(rul_splits[s]) for s in ('train', 'validation', 'calibration')))
    if set(raw_train.unit_nr) != expected:
        raise ValueError('Inventário do arquivo original de treino diverge dos splits RUL.')
    # RUL, duração e qualquer coluna extra nunca são features.
    development = validate_trajectories(raw_train.loc[raw_train.unit_nr.isin(rul_splits['train']), RAW_COLUMNS])
    rows = []
    for unit, group in development.groupby('unit_nr', sort=True):
        count = int(np.floor(len(group) * config.nominal_fraction))
        rows.append([config.dataset, 'train', 'train', role_of(int(unit), splits), int(unit),
                     len(group), count, count >= 30, max(0, count - 29)])
    units = pd.DataFrame(rows, columns=UNIT_COLUMNS)
    nominal = {}
    for role in ROLES:
        limits = units.loc[units.AERole.eq(role) & units.eligible].set_index('unit_nr').nominal_cycles
        if limits.empty:
            assigned = units.loc[units.AERole.eq(role), 'unit_nr'].tolist()
            raise ValueError(f'{role}: nenhum motor com pelo menos 30 ciclos nominais: {assigned}; não há reposição por holdouts.')
        nominal[role] = development.loc[development.unit_nr.isin(limits.index)
                                        & development.time_cycles.le(development.unit_nr.map(limits))].copy()
    fit_limits = {int(u): int(n) for u, n in units.loc[units.AERole.eq('ae_fit') & units.eligible,
                                                    ['unit_nr', 'nominal_cycles']].itertuples(index=False, name=None)}
    processor = CoordinatedNominalPreprocessor(config, splits, reference, fit_limits).fit(nominal['ae_fit'])
    arrays, metadata = {}, {}
    for role in ROLES:
        raw = nominal[role].assign(Dataset=config.dataset, SourceSplit='train')
        arrays[role], metadata[role] = prepare_aligned_windows(raw, nominal_metadata(units, config.dataset, role), processor)
    return processor, splits, units, arrays, metadata


def process_dataset(config, raw_dir=RAW_DIR, rul_dir=RUL_DIR, output_dir=OUTPUT_DIR):
    target = output_target(output_dir, raw_dir, rul_dir, config.dataset)
    ids, reference = load_rul_reference(config.dataset, rul_dir)
    path = Path(raw_dir) / f'train_{config.dataset}.txt'
    if file_hash(path) != reference['raw_train_sha256']:
        raise ValueError('Arquivo bruto de treino diferente da referência RUL v2.')
    raw = pd.read_csv(path, sep=r'\s+', header=None)
    if raw.shape[1] != len(RAW_COLUMNS):
        raise ValueError('Arquivo de treino deve ter exatamente 26 colunas.')
    raw.columns = RAW_COLUMNS
    processor, splits, units, arrays, metadata = prepare_frames(raw, config, ids, reference)
    if file_hash(path) != reference['raw_train_sha256'] or load_rul_reference(config.dataset, rul_dir) != (ids, reference):
        raise ValueError('Insumos alterados durante a preparação.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    write_json(target / 'config.json', config.to_dict())
    write_json(target / 'splits_ae.json', splits)
    write_json(target / 'features.json', {'selected': processor.features_, 'dropped': processor.dropped_features_})
    with (target / 'preprocessor.joblib').open('xb') as stream:
        joblib.dump(processor, stream)
    with (target / 'nominal_units.csv').open('x', encoding='utf-8', newline='') as stream:
        units.to_csv(stream, index=False)
    summary = {}
    for role, suffix in zip(ROLES, ('fit', 'inner_validation')):
        with (target / f'nominal_{suffix}.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays[role])
        with (target / f'nominal_{suffix}_metadata.csv').open('x', encoding='utf-8', newline='') as stream:
            metadata[role].to_csv(stream, index=False)
        group = units.loc[units.AERole.eq(role)]
        summary[role] = {'assigned_engines': len(group), 'eligible_engines': int(group.eligible.sum()),
                         'excluded_engines': group.loc[~group.eligible, 'unit_nr'].tolist(),
                         'nominal_cycles': int(group.loc[group.eligible, 'nominal_cycles'].sum()),
                         'windows': len(metadata[role])}
    # Escrito por último: pasta parcial nunca é considerada uma execução concluída.
    manifest = {'pipeline': PIPELINE, 'schema_version': 1, 'Dataset': config.dataset,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config.to_dict(),
                'rul_reference': reference, 'software': software_versions(), 'summary': summary,
                'origins': {'raw_train': str(path.resolve()), 'rul_folder': str((Path(rul_dir) / config.dataset).resolve())},
                'code_sha256': {str(p.relative_to(Path(__file__).parents[3])): file_hash(p) for p in
                               (Path(__file__), Path(__file__).with_name('utils.py'),
                                Path(__file__).parents[2] / 'rul_prediction/v2/utils.py')},
                'sha256': {name: file_hash(target / name) for name in sorted(ARTIFACTS)}}
    write_json(target / 'manifest.json', manifest)
    print(config.dataset, summary)
    return target


def validate_saved_dataset(folder, rul_dir=RUL_DIR, expected_options=None):
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    if manifest['pipeline'] != PIPELINE or manifest['schema_version'] != 1 or set(manifest['sha256']) != ARTIFACTS:
        raise ValueError('Manifesto AE v2 inválido/incompleto.')
    for name, digest in manifest['sha256'].items():
        if file_hash(folder / name) != digest:
            raise ValueError(f'Hash AE divergente: {name}.')
    saved = read_json(folder / 'config.json')
    config = Config(**{k: saved[k] for k in Config.__dataclass_fields__})
    if saved != config.to_dict() or saved != manifest['config'] or config.dataset != folder.name or manifest['Dataset'] != config.dataset:
        raise ValueError('Configuração AE incompatível.')
    if any(saved.get(k) != v for k, v in (expected_options or {}).items()):
        raise ValueError('Opções solicitadas diferem da configuração salva.')
    ids, reference = load_rul_reference(config.dataset, rul_dir)
    splits = read_json(folder / 'splits_ae.json')
    if reference != manifest['rul_reference'] or splits != split_inside_rul_train(ids, config):
        raise ValueError('Referência/divisões RUL ou AE incompatíveis.')
    units = pd.read_csv(folder / 'nominal_units.csv')
    if (list(units.columns) != UNIT_COLUMNS or set(units.unit_nr) != set(ids['train'])
            or units.unit_nr.duplicated().any() or not units.Dataset.eq(config.dataset).all()
            or not units.SourceSplit.eq('train').all() or not units.Split.eq('train').all()):
        raise ValueError('Inventário nominal inválido.')
    for col in ('unit_nr', 'total_cycles', 'nominal_cycles', 'nominal_windows'):
        values = units[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < 0).any():
            raise ValueError('Contagens nominais inválidas.')
    if (not units.eligible.isin([True, False]).all() or (units.total_cycles < 1).any()
            or not np.array_equal(units.nominal_cycles, np.floor(units.total_cycles * config.nominal_fraction))
            or not units.eligible.eq(units.nominal_cycles.ge(30)).all()
            or not np.array_equal(units.nominal_windows, np.maximum(0, units.nominal_cycles - 29))
            or units.AERole.tolist() != [role_of(int(u), splits) for u in units.unit_nr]):
        raise ValueError('Região nominal ou papel interno incompatível.')
    features = read_json(folder / 'features.json')
    selected = features['selected']
    if (not selected or selected != [f for f in INPUT_COLUMNS if f in selected]
            or features['dropped'] != [f for f in INPUT_COLUMNS if f not in selected]):
        raise ValueError('Ordem/lista de features inválida.')
    processor = joblib.load(folder / 'preprocessor.joblib')
    limits = {int(u): int(n) for u, n in units.loc[units.AERole.eq('ae_fit') & units.eligible,
                                                ['unit_nr', 'nominal_cycles']].itertuples(index=False, name=None)}
    if (not isinstance(processor, CoordinatedNominalPreprocessor) or not processor.fitted_
            or processor.config != config or processor.splits != splits or processor.reference != reference
            or processor.fit_limits != limits or processor.train_units_ != sorted(limits)
            or processor.features_ != selected or processor.features_at_fit_ != tuple(selected)
            or processor.dropped_features_ != features['dropped']):
        raise ValueError('Identidade do pré-processador incompatível.')
    multi = config.dataset in ('FD002', 'FD004')
    if (set(processor.scalers_) != set(range(6) if multi else [0])
            or (processor.kmeans_ is not None) != multi):
        raise ValueError('Regimes/scalers inválidos.')
    if multi and (processor.kmeans_.n_clusters != 6 or processor.kmeans_.random_state != config.seed
                  or processor.kmeans_.n_init != 10 or list(processor.kmeans_.feature_names_in_) != SETTINGS
                  or len(processor.kmeans_.labels_) != sum(limits.values())):
        raise ValueError('KMeans incompatível com AE fit nominal.')
    for scaler in processor.scalers_.values():
        if list(scaler.feature_names_in_) != selected or scaler.clip or scaler.feature_range != (0, 1):
            raise ValueError('Scaler incompatível com as features nominais.')
    if sum(int(s.n_samples_seen_) for s in processor.scalers_.values()) != sum(limits.values()):
        raise ValueError('Scalers ajustados com quantidade incompatível de linhas.')
    summary = {}
    for role, suffix in zip(ROLES, ('fit', 'inner_validation')):
        meta = pd.read_csv(folder / f'nominal_{suffix}_metadata.csv')
        expected = nominal_metadata(units, config.dataset, role)
        if expected.empty or not meta.equals(expected):
            raise ValueError(f'{role}: metadados não correspondem às regiões nominais herdadas.')
        with np.load(folder / f'nominal_{suffix}.npz', allow_pickle=False) as arrays:
            if (set(arrays.files) != {'X', 'features'} or arrays['features'].tolist() != selected
                    or arrays['X'].shape != (len(meta), 30, len(selected))
                    or arrays['X'].dtype != np.float32 or not np.isfinite(arrays['X']).all()):
                raise ValueError(f'{role}: NPZ/features/shape inválidos.')
        group = units.loc[units.AERole.eq(role)]
        summary[role] = {'assigned_engines': len(group), 'eligible_engines': int(group.eligible.sum()),
                         'excluded_engines': group.loc[~group.eligible, 'unit_nr'].tolist(),
                         'nominal_cycles': int(group.loc[group.eligible, 'nominal_cycles'].sum()), 'windows': len(meta)}
    if summary != manifest['summary']:
        raise ValueError('Resumo nominal divergente dos artefatos.')
    return processor, summary


def load_preprocessor(folder, rul_dir=RUL_DIR):
    """Somente objetos locais confiáveis: hashes não garantem autenticidade do pickle."""
    return validate_saved_dataset(folder, rul_dir)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
    parser.add_argument('--seed', type=int, help='Padrão 42 na preparação; na validação, compara se informado.')
    parser.add_argument('--nominal-fraction', type=float, help='Padrão 0.30.')
    parser.add_argument('--inner-validation-fraction', type=float, help='Padrão 0.20, somente sobre RUL train.')
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--rul-dir', type=Path, default=RUL_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--validate-only', action='store_true', help='Valida artefatos AE salvos; não lê dados brutos ou teste.')
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Não repita datasets.')
    options = {k: getattr(args, k) for k in ('seed', 'nominal_fraction', 'inner_validation_fraction') if getattr(args, k) is not None}
    configs = [Config(dataset, **options) for dataset in args.datasets]
    if not args.validate_only:
        for config in configs:
            output_target(args.output_dir, args.raw_dir, args.rul_dir, config.dataset)
    for config in configs:
        if args.validate_only:
            _, summary = validate_saved_dataset(args.output_dir / config.dataset, args.rul_dir, options)
            print(config.dataset, summary)
        else:
            process_dataset(config, args.raw_dir, args.rul_dir, args.output_dir)


if __name__ == '__main__':
    main()
