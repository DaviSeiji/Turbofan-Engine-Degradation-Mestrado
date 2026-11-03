"""Executa o pipeline RUL v2 configurado abaixo. --dry-run apenas mostra o plano."""

import argparse
import logging
import os
import warnings

from src.rul_prediction.v2 import evaluation, train
from src.rul_prediction.v2.experiment import (
    DATA_ROOT, MODEL_ROOT, RESULT_ROOT, MODELS, PROTOCOL, experiment_path, read_json,
)
from src.rul_prediction.v2.utils import DATASETS as VALID_DATASETS, file_hash


# Datasets completos são conferidos e pulados automaticamente.
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
EXPERIMENTO = 'baseline'
SEEDS = [1, 2, 3, 4, 5]
SAMPLER_SEED = 42
TOTAL_TRIALS = 30
EPOCAS_OPTUNA = 50
EPOCAS_FINAL = 50
PATIENCE = 5


def configure_tensorflow_logging():
    # Silencia INFO/WARNING, mas mantém erros e tracebacks visíveis.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message=r'The structure of .inputs. doesn.t match.*',
                            category=UserWarning, module=r'keras\..*')
    warnings.filterwarnings('ignore', message=r'TensorFlow GPU support is not available.*')


def checked_manifest(folder, names):
    manifest = read_json(folder / 'manifest.json')
    if manifest['protocol'] != PROTOCOL or set(manifest['sha256']) != set(names):
        raise ValueError(f'Manifesto incompatível: {folder}.')
    for name in names:
        if file_hash(folder / name) != manifest['sha256'][name]:
            raise ValueError(f'Artefato alterado: {folder / name}.')
    return manifest


def dataset_complete(dataset):
    """Só pula quando todas as seeds/avaliações existem, são íntegras e compatíveis."""
    folders = []
    for model in MODELS:
        parent = experiment_path(MODEL_ROOT, dataset, model, EXPERIMENTO)
        output = experiment_path(RESULT_ROOT, dataset, model, EXPERIMENTO)
        for seed in SEEDS:
            folders += [parent / f'seed_{seed}']
            folders += [output / f'seed_{seed}' / p for p in ('terminal', 'windows')]
    if not any(path.exists() for path in folders):
        return False
    if not all((path / 'manifest.json').is_file() for path in folders):
        raise FileExistsError(f'{dataset}: execução parcial encontrada. Nada será sobrescrito; '
                              'continue pelas CLIs individuais antes de usar o runner.')
    preprocessing_hash = file_hash(DATA_ROOT / dataset / 'manifest.json')
    for model in MODELS:
        parent = experiment_path(MODEL_ROOT, dataset, model, EXPERIMENTO)
        output = experiment_path(RESULT_ROOT, dataset, model, EXPERIMENTO)
        study = read_json(parent / 'study_config.json')
        expected = train.study_config(study['data'], model, EXPERIMENTO, SAMPLER_SEED,
                                      TOTAL_TRIALS, EPOCAS_OPTUNA, PATIENCE)
        if study != expected or study['data']['manifest_sha256'] != preprocessing_hash:
            raise ValueError(f'{dataset}/{model}: configuração/dados/código diferentes dos salvos.')
        for seed in SEEDS:
            source = parent / f'seed_{seed}'
            manifest = checked_manifest(source, ('model.keras', 'config.json', 'history.json', 'history.png'))
            config = read_json(source / 'config.json')
            if (manifest['data'] != study['data'] or manifest['seed'] != seed
                    or manifest['model'] != model or manifest['experiment'] != EXPERIMENTO
                    or any(config.get(key) != value for key, value in study.items())
                    or config['seed'] != seed or config['final_epochs'] != EPOCAS_FINAL
                    or config['final_patience'] != PATIENCE):
                raise ValueError(f'Modelo final incompatível: {source}.')
            for protocol in ('terminal', 'windows'):
                result = checked_manifest(output / f'seed_{seed}' / protocol,
                                          ('metrics.json', 'predictions.csv', 'predictions.png'))
                record = result['record']
                if (record['Dataset'] != dataset or record['Model'] != model
                        or record['Seed'] != seed or record['Experiment'] != EXPERIMENTO
                        or record['Protocol'] != protocol or record['Target'] != 'y_original_uncapped'
                        or record['PreprocessingManifest'] != preprocessing_hash
                        or record['ModelSHA256'] != manifest['sha256']['model.keras']
                        or read_json(output / f'seed_{seed}' / protocol / 'metrics.json') != record):
                    raise ValueError(f'Avaliação incompatível: {dataset}/{model}/{seed}/{protocol}.')
    return True


def comparison_complete(comparison, protocol):
    folder = RESULT_ROOT / 'comparisons' / comparison / protocol
    if not folder.exists():
        return False
    expected = {'datasets': DATASETS, 'models': list(MODELS), 'experiment': EXPERIMENTO,
                'seeds': SEEDS, 'protocol': protocol, 'std_ddof': 1}
    if (not all((folder / name).is_file() and (folder / name).stat().st_size > 0
                for name in ('config.json', 'individual.csv', 'summary.csv'))
            or read_json(folder / 'config.json') != expected):
        raise ValueError(f'Comparação existente incompleta/incompatível: {folder}.')
    return True


def run_pipeline(dry_run=False):
    if (not DATASETS or len(set(DATASETS)) != len(DATASETS)
            or not set(DATASETS) <= set(VALID_DATASETS)):
        raise ValueError('Configure datasets válidos, sem repetições.')
    if (not SEEDS or len(set(SEEDS)) != len(SEEDS)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in SEEDS)
            or not 0 <= SAMPLER_SEED < 2**32
            or min(TOTAL_TRIALS, EPOCAS_OPTUNA, EPOCAS_FINAL) < 1 or PATIENCE < 0):
        raise ValueError('Seeds, trials, épocas ou patience inválidos.')
    comparison = EXPERIMENTO + '_' + '_'.join(d.lower() for d in DATASETS)
    experiment_path(RESULT_ROOT, DATASETS[0], MODELS[0], comparison)
    print(f'Datasets: {DATASETS}; modelos: {MODELS}; experimento: {EXPERIMENTO}', flush=True)
    print(f'Optuna: {TOTAL_TRIALS} trials totais; treino final: seeds {SEEDS}.', flush=True)
    print('Ordem: otimizar e treinar todos os modelos -> avaliar terminal/janelas -> comparar.', flush=True)
    if dry_run:
        print('Somente plano: nenhum dado carregado e nenhum experimento executado.')
        return

    configure_tensorflow_logging()
    pending = []
    for dataset in DATASETS:
        if dataset_complete(dataset):
            print(f'{dataset}: já concluído e conferido; pulando.', flush=True)
        else:
            pending.append(dataset)
    comparisons = {p: comparison_complete(comparison, p) for p in ('terminal', 'windows')}
    if pending and any(comparisons.values()):
        raise ValueError('Comparação existente, mas faltam datasets completos; confira os artefatos.')

    # O campeão vem do Optuna, sem precisar informar o número manualmente.
    for dataset in pending:
        for model in MODELS:
            print(f'\nOtimizando {dataset}/{model}...', flush=True)
            snapshot = train.optimize(DATA_ROOT, MODEL_ROOT, dataset, model, EXPERIMENTO,
                                      seed=SAMPLER_SEED, trials=TOTAL_TRIALS,
                                      epochs=EPOCAS_OPTUNA, patience=PATIENCE)
            best_trial = read_json(snapshot)['trial']
            print(f'Treinando {dataset}/{model}, campeão {best_trial}...', flush=True)
            train.train_final(DATA_ROOT, MODEL_ROOT, dataset, model, EXPERIMENTO,
                              seeds=SEEDS, best_trial=best_trial,
                              epochs=EPOCAS_FINAL, patience=PATIENCE)

    # Só acessa o teste depois de concluir todos os ajustes configurados.
    for dataset in pending:
        for model in MODELS:
            for protocol in ('terminal', 'windows'):
                print(f'\nAvaliando {dataset}/{model}: {protocol}...', flush=True)
                evaluation.evaluate(DATA_ROOT, MODEL_ROOT, RESULT_ROOT, dataset, model,
                                    EXPERIMENTO, seeds=SEEDS, protocol=protocol)
    for protocol in ('terminal', 'windows'):
        if comparisons[protocol]:
            print(f'Comparação {protocol}: já existe; pulando.', flush=True)
        else:
            evaluation.compare(RESULT_ROOT, DATASETS, EXPERIMENTO, SEEDS, protocol, comparison)
    print('\nPipeline concluído. Resultados em results/rul_prediction/v2/.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Mostra o plano sem carregar dados ou executar etapas.')
    args = parser.parse_args()
    run_pipeline(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
