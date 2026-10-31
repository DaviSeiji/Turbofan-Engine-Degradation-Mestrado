# Reorganização estrutural

Referência: `main`, commit `11a5b51`. A organização separa previsão de RUL,
detecção de anomalias e XAI, mantendo os protocolos e os artefatos existentes.

## Movimentações

| Origem | Destino |
| --- | --- |
| `src/ml/preproc_data_nasa.py` | `src/common/preprocessing/preproc_data_nasa.py` |
| `src/ml/{baseline_models,model_builders,transformer_layers}.py` | `src/rul_prediction/models/` |
| `src/ml/{tune_cnn,tune_hybrid,tune_transformer}.py` | `src/rul_prediction/training/` |
| `src/ml/testar_modelos.py` | `src/rul_prediction/evaluation/testar_modelos.py` |
| `src/ml/anomaly_preprocessing.py` | `src/anomaly_detection/preprocessing.py` |
| `src/ml/autoencoder_models.py` | `src/anomaly_detection/models/autoencoder_models.py` |
| `src/ml/tune_autoencoder.py` | `src/anomaly_detection/training/tune_autoencoder.py` |
| `src/ml/{detect_anomalies,evaluate_anomaly_detection}.py` | `src/anomaly_detection/evaluation/` |
| Sete explicadores de `src/xai/` | `src/xai/explainers/` |
| `src/xai/{xai_estabilidade,xai_estabilidade_modelos}.py` | `src/xai/evaluation/` |
| `notebooks/ml/` | `notebooks/rul_prediction/` |
| `tests/test_evaluate_anomaly_detection.py` | `tests/anomaly_detection/test_evaluate_anomaly_detection.py` |
| `results/anomaly_detection/FD00X/` | `results/anomaly_detection/detection/FD00X/` |
| `results/anomaly_evaluation/` | `results/anomaly_detection/evaluation/` |
| `results/xai_estabilidade/` | `results/xai/stability/` |
| `results/xai_estabilidade_modelos/` | `results/xai/model_stability/` |
| `results/ml/` (vazia) | `results/rul_prediction/` |

A lista individual e os hashes estão em
[structural_refactor_manifest.json](structural_refactor_manifest.json).
Os resultados antigos são registros históricos; seus comandos e caminhos internos
foram preservados integralmente. Os comandos atuais estão no [README](../README.md).

## Compatibilidade e caminhos

- `src.common.paths.ROOT` centraliza a raiz; dados, modelos e bancos locais
  continuam nas mesmas pastas `data/` e `models/`.
- `src.rul_prediction.experiment_utils` reúne o registro `MODELOS` e as funções
  `carregar_best_params` e `salvar_csv_atomico`, extraídas sem alterar o cálculo.
  XAI e benchmark importam essas mesmas funções.
- `src/ml/anomaly_preprocessing.py` permanece como ponte para a classe
  `NominalPreprocessor` referenciada pelos `joblib` antigos. Os imports atuais
  usam `src.anomaly_detection.preprocessing`.
- `PositionalEmbedding` mantém o registro Keras `RUL>PositionalEmbedding`.
- Os notebooks usam a raiz centralizada e conservam outputs e metadados.
- As regras `/data/` e `/models/` do `.gitignore` se aplicam somente à raiz,
  permitindo versionar o código dos subpacotes chamados `models`.
- Dois caches `.pyc` foram removidos somente do índice Git; as cópias locais
  foram preservadas. Nenhum treinamento ou estudo Optuna foi executado.

## Verificação

```powershell
python -m unittest discover -s tests -v
python -m src.anomaly_detection.training.tune_autoencoder --help
python -m src.anomaly_detection.evaluation.detect_anomalies --help
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --help
```

A suíte verifica imports, compilação de fontes e células dos notebooks,
cálculos existentes, consenso sintético e carregamento dos artefatos locais.
Os scripts de RUL e de estabilidade XAI não possuem interface `argparse`;
seus imports são testados sem executar os respectivos blocos principais.
Não se deve passar `--help` a esses scripts como teste, pois eles iniciariam
os experimentos mesmo com esse argumento.

A auditoria anterior à movimentação confirmou hashes idênticos de 277 arquivos
de resultados, dados e modelos e a preservação dos outputs/metadados dos sete
notebooks. Nenhum resultado foi recalculado durante esta reorganização.
