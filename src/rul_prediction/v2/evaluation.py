"""Avaliação RUL v2 separada do treinamento: terminal, janelas e comparação."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.rul_prediction.v2.experiment import (
    DATA_ROOT, JOIN_KEYS, MODEL_ROOT, MODELS, PROTOCOL, RESULT_ROOT,
    assert_model_shape, check_output_root, experiment_path, load_data, read_json, write_json,
)
from src.rul_prediction.v2.utils import DATASETS, file_hash

METRICS = ('RMSE', 'MAE', 'MedianAE', 'P95AE', 'MaxAE')


def metrics(y_true, y_pred):
    truth, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if (truth.ndim != 1 or truth.shape != prediction.shape or not len(truth)
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError('Targets/previsões devem ser vetores finitos e alinhados.')
    error = np.abs(truth - prediction)
    return dict(zip(METRICS, map(float, (np.sqrt(np.mean(error**2)), np.mean(error),
                                       np.median(error), np.percentile(error, 95), np.max(error)))))


def prediction_table(part, prediction, model, seed, experiment):
    prediction = np.asarray(prediction)
    if prediction.shape == (len(part['y_original']), 1):
        prediction = prediction[:, 0]
    metrics(part['y_original'], prediction)  # Valida antes de montar a tabela.
    table = part['metadata'].copy()
    table['Model'], table['Seed'], table['Experiment'] = model, seed, experiment
    table['RUL_true'], table['RUL_pred'] = part['y_original'], prediction
    table['AbsoluteError'] = np.abs(table.RUL_true - table.RUL_pred)
    if table.duplicated(JOIN_KEYS).any():
        raise ValueError('Chaves de associação duplicadas nas previsões.')
    return table


def load_model(path):
    # O import registra RUL>PositionalEmbedding antes do load_model.
    from src.rul_prediction.models.transformer_layers import PositionalEmbedding  # noqa: F401
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def plot_predictions(table, destination):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(table.RUL_true, table.RUL_pred, s=8, alpha=.5)
    limits = [min(table.RUL_true.min(), table.RUL_pred.min()),
              max(table.RUL_true.max(), table.RUL_pred.max())]
    ax.plot(limits, limits, '--', color='black')
    ax.set(xlabel='True RUL (uncapped)', ylabel='Predicted RUL')
    fig.tight_layout(); fig.savefig(destination); plt.close(fig)


def clear_runtime():
    from tensorflow import keras
    keras.backend.clear_session()


def evaluate(data_root, model_root, result_root, dataset, model, experiment, seeds,
             protocol='terminal', batch_size=128):
    check_output_root(result_root, RESULT_ROOT)
    if protocol not in ('terminal', 'windows') or batch_size < 1:
        raise ValueError('Protocolo/batch size inválido.')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Informe seeds únicas.')
    split = 'test_' + protocol
    data, provenance = load_data(Path(data_root) / dataset, (split,))
    parent = experiment_path(model_root, dataset, model, experiment)
    output = experiment_path(result_root, dataset, model, experiment)
    # Valida todas as seeds antes de qualquer inferência.
    manifests = {}
    for seed in seeds:
        source = parent / f'seed_{seed}'
        manifest = read_json(source / 'manifest.json')
        if (manifest['protocol'] != PROTOCOL or manifest['data'] != provenance
                or manifest['model'] != model or manifest['experiment'] != experiment
                or manifest['seed'] != seed):
            raise ValueError('Modelo e pré-processamento/experimento incompatíveis.')
        if set(manifest['sha256']) != {'model.keras', 'config.json', 'history.json', 'history.png'}:
            raise ValueError('Manifesto do modelo incompleto.')
        for name, digest in manifest['sha256'].items():
            if file_hash(source / name) != digest:
                raise ValueError(f'Artefato do modelo alterado: {name}.')
        config = read_json(source / 'config.json')
        if (config['data'] != provenance or config['model'] != model
                or config['seed'] != seed or config['experiment'] != experiment):
            raise ValueError('Configuração final inconsistente.')
        if (output / f'seed_{seed}' / protocol).exists():
            raise FileExistsError('Resultado existente será preservado; use outro experimento.')
        manifests[seed] = (manifest, config)
    for seed in seeds:
        source = parent / f'seed_{seed}'
        target = output / f'seed_{seed}' / protocol
        target.mkdir(parents=True, exist_ok=False)
        fitted = load_model(source / 'model.keras')
        try:
            assert_model_shape(fitted, provenance['input_shape'])
            prediction = fitted.predict(data[split]['X'], batch_size=batch_size, verbose=0)
        finally:
            clear_runtime()
        table = prediction_table(data[split], prediction, model, seed, experiment)
        scores = metrics(table.RUL_true.to_numpy(), table.RUL_pred.to_numpy())
        table.to_csv(target / 'predictions.csv', index=False, mode='x')
        plot_predictions(table, target / 'predictions.png')
        model_manifest, config = manifests[seed]
        training_protocol = {key: config[key] for key in
                             ('protocol', 'sampler_seed', 'total_trials', 'epochs', 'patience',
                              'final_epochs', 'final_patience', 'software', 'code_sha256')}
        record = {'Dataset': dataset, 'Model': model, 'Experiment': experiment, 'Seed': seed,
                  'Protocol': protocol, 'Target': 'y_original_uncapped',
                  'PreprocessingManifest': provenance['manifest_sha256'],
                  'ModelSHA256': model_manifest['sha256']['model.keras'],
                  'TrainingProtocol': training_protocol,
                  'N': len(table), **scores}
        write_json(target / 'metrics.json', record)
        write_json(target / 'manifest.json', {
            'protocol': PROTOCOL, 'record': record,
            'sha256': {name: file_hash(target / name) for name in
                       ('predictions.csv', 'predictions.png', 'metrics.json')}})
        print(record)
        del fitted


def compare(result_root, datasets, experiment, seeds, protocol, comparison):
    """Exige ambas as arquiteturas e todas as seeds; nunca seleciona a melhor."""
    check_output_root(result_root, RESULT_ROOT)
    if (not seeds or len(set(seeds)) != len(seeds) or len(set(datasets)) != len(datasets)
            or protocol not in ('terminal', 'windows')):
        raise ValueError('Datasets/seeds/protocolo inválidos.')
    # Valida o nome sem permitir caminhos arbitrários para a tabela.
    experiment_path(result_root, 'FD001', 'cnn_lstm', comparison)
    output = Path(result_root) / 'comparisons' / comparison / protocol
    if output.exists():
        raise FileExistsError('Comparação existente será preservada.')
    records = []
    for dataset in datasets:
        reference = None
        for model in MODELS:
            for seed in seeds:
                folder = experiment_path(result_root, dataset, model, experiment) / f'seed_{seed}' / protocol
                manifest = read_json(folder / 'manifest.json')
                if manifest['protocol'] != PROTOCOL:
                    raise ValueError('Protocolo de resultados incompatível.')
                for name in ('metrics.json', 'predictions.csv', 'predictions.png'):
                    if file_hash(folder / name) != manifest['sha256'][name]:
                        raise ValueError('Resultado alterado desde a avaliação.')
                record = read_json(folder / 'metrics.json')
                if (record != manifest['record'] or record['Dataset'] != dataset
                        or record['Model'] != model or record['Seed'] != seed
                        or record['Experiment'] != experiment or record['Protocol'] != protocol
                        or record['Target'] != 'y_original_uncapped'):
                    raise ValueError('Identidade dos resultados incompatível.')
                table = pd.read_csv(folder / 'predictions.csv')
                current = (record['PreprocessingManifest'], record['TrainingProtocol'], table[JOIN_KEYS].to_dict('list'),
                           table.RUL_true.tolist())
                if reference is not None and current != reference:
                    raise ValueError('Comparação exige exatamente o mesmo pré-processamento e janelas.')
                reference = current
                records.append(record)
    individual = pd.DataFrame(records)
    if not np.isfinite(individual[list(METRICS)].to_numpy()).all():
        raise ValueError('Métricas não finitas.')
    summary = individual.groupby(['Dataset', 'Model', 'Protocol'])[list(METRICS)].agg(['mean', 'std'])
    summary.columns = [f'{name}_{stat}' for name, stat in summary.columns]
    # Desvio padrão amostral (ddof=1); uma única seed resulta em NaN.
    output.mkdir(parents=True, exist_ok=False)
    individual.to_csv(output / 'individual.csv', index=False, mode='x')
    summary.to_csv(output / 'summary.csv', mode='x')
    write_json(output / 'config.json', {'datasets': datasets, 'models': list(MODELS),
               'experiment': experiment, 'seeds': seeds, 'protocol': protocol, 'std_ddof': 1})
    print(summary.to_string())
    return individual, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('terminal', 'windows', 'compare'):
        p = sub.add_parser(command)
        p.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
        p.add_argument('--experiment', default='baseline')
        p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
        if command == 'compare':
            p.add_argument('--protocol', choices=('terminal', 'windows'), default='terminal')
            p.add_argument('--comparison', required=True, help='Nome novo para preservar tabelas existentes.')
        else:
            p.add_argument('--model', choices=MODELS, required=True)
            p.add_argument('--data-dir', type=Path, default=DATA_ROOT)
            p.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Não repita datasets.')
    if args.command == 'compare':
        compare(RESULT_ROOT, args.datasets, args.experiment, args.seeds, args.protocol, args.comparison)
    else:
        for dataset in args.datasets:
            evaluate(args.data_dir, MODEL_ROOT, RESULT_ROOT, dataset, args.model,
                     args.experiment, args.seeds, args.command, args.batch_size)


if __name__ == '__main__':
    main()
