# Turbofan Engine Degradation — Mestrado

Pesquisa sobre degradação de motores turbofan usando NASA C-MAPSS (FD001–FD004).
O repositório reúne três áreas complementares: previsão de vida útil remanescente
(RUL), detecção não supervisionada de anomalias e explicabilidade (XAI).
A organização permite integração futura sem misturar os protocolos experimentais.

## Organização

```text
src/
├── common/
│   ├── paths.py                       # Raiz única do projeto
│   └── preprocessing/preproc_data_nasa.py
├── rul_prediction/
│   ├── models/                        # Baselines, CNN-LSTM e Transformer
│   ├── training/                      # Otimização e treinamento
│   ├── evaluation/testar_modelos.py    # Benchmark com seeds (inclui treino)
│   └── experiment_utils.py            # Registro e utilitários usados por RUL/XAI
├── anomaly_detection/
│   ├── preprocessing.py
│   ├── models/autoencoder_models.py
│   ├── training/tune_autoencoder.py
│   └── evaluation/                    # Detecção e avaliação dos CSVs
├── xai/
│   ├── xai_baseExplainer.py
│   ├── xai_utils.py
│   ├── explainers/                    # SHAP, IG, Saliency, LIME e outros
│   └── evaluation/                    # Estabilidade do consenso
└── ml/anomaly_preprocessing.py         # Ponte para joblib antigos; não é implementação duplicada

notebooks/{rul_prediction,anomaly_detection,xai}/
tests/{rul_prediction,anomaly_detection,xai}/
results/
├── rul_prediction/
├── anomaly_detection/{detection,evaluation}/
└── xai/{stability,model_stability}/
data/                                  # Local, não versionado
models/                                # Local, não versionado
docs/                                  # Mapa de movimentações e auditoria
```

Os pacotes Python possuem `__init__.py`; execute scripts com `python -m`, a partir
da raiz. Os caminhos de dados/modelos são ancorados em `src.common.paths.ROOT`.
Nos notebooks, a primeira célula localiza essa raiz sem depender de o kernel
iniciar na raiz do projeto ou na pasta do notebook. Outputs históricos foram
preservados; não significam uma nova execução após a reorganização.

## Previsão de RUL

Modelos de referência ficam em `src/rul_prediction/models/baseline_models.py`;
os notebooks em `notebooks/rul_prediction/` incluem baselines, CNN e CNN-LSTM,
além do notebook de referência do artigo. O notebook de referência conserva suas
rotinas originais de download e processamento; não foi executado na refatoração.

Os comandos abaixo **executam experimentos e podem treinar/salvar modelos**.
São apresentados como interface de uso, não foram executados na reorganização.

```powershell
# Pré-processamento RUL: recria NPZ; não execute se quiser apenas usar os existentes.
python -m src.common.preprocessing.preproc_data_nasa

# CNN: KerasTuner; CNN-LSTM e Transformer: Optuna.
python -m src.rul_prediction.training.tune_cnn
python -m src.rul_prediction.training.tune_hybrid
python -m src.rul_prediction.training.tune_transformer

# Benchmark por seeds: treina novas instâncias, não é apenas inferência.
python -m src.rul_prediction.evaluation.testar_modelos
```

Datasets/seeds continuam definidos no início dos respectivos scripts. As
arquiteturas, espaços de busca, quantidade de trials, splits e EarlyStopping
foram mantidos. `testar_modelos` e XAI compartilham `MODELOS`,
`carregar_best_params` e `salvar_csv_atomico` em `experiment_utils.py`.
O carregamento dos melhores parâmetros continua usando cópia temporária do banco
Optuna, sem modificar o estudo original.

## Detecção de anomalias

O LSTM Autoencoder reconstrói janelas de 30 ciclos. A região nominal assumida
corresponde aos primeiros 30% das trajetórias completas de desenvolvimento;
RUL é apenas metadado de avaliação. O baseline mantém MSE de reconstrução,
P99 nominal de validação e alerta após três janelas consecutivas acima do threshold.
FD002/FD004 mantêm normalização por regime com o KMeans já ajustado.

```powershell
# Treinamento/Optuna — somente quando desejar um novo experimento.
python -m src.anomaly_detection.training.tune_autoencoder --datasets FD001 FD002 FD003 FD004

# Inferência com modelos salvos; gera resultados de detecção.
python -m src.anomaly_detection.evaluation.detect_anomalies --datasets FD001 FD002 FD003 FD004

# Inspeção da interface, sem executar treinamento ou inferência.
python -m src.anomaly_detection.training.tune_autoencoder --help
python -m src.anomaly_detection.evaluation.detect_anomalies --help
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --help
```

O pré-processamento específico do Autoencoder é reutilizado pelo treinamento;
não é necessário executar `preprocessing.py` separadamente.

## XAI

As implementações dos explicadores ficam em `src/xai/explainers/`; a classe base
e as funções gerais permanecem em `src/xai/`. Os notebooks de análise mantêm
suas explicações, métricas, gráficos embutidos e resultados anteriores.

```powershell
# Atenção: ambos os experimentos de estabilidade abaixo incluem retreinamentos.
python -m src.xai.evaluation.xai_estabilidade
python -m src.xai.evaluation.xai_estabilidade_modelos
```

As saídas ficam em `results/xai/stability/` e `results/xai/model_stability/`.
As imagens de Fidelity exportadas anteriormente para a pasta de estabilidade
foram movidas junto com ela, preservando seus nomes. Imagens vinculadas aos
notebooks em `notebooks/xai/figuras/` permanecem no lugar.
O consenso mantém Top-10 por método, pontuação 10→1, soma dos scores e Top-5.

## Dados, modelos e compatibilidade

- `data/raw_data/`, `data/processed_data/` e `data/processed_anomaly/`: mesmos
  arquivos e formatos anteriores; nenhum dado foi reprocessado na reorganização.
- `models/`: pesos, estudos Optuna/KerasTuner e CSVs históricos continuam nos
  mesmos caminhos. Inclusive os CSVs de benchmark em `models/` foram preservados
  para não interromper a retomada dos experimentos existentes.
- `results/`: resultados versionáveis separados por área. `results/rul_prediction/`
  corresponde à antiga pasta `results/ml/`, que estava vazia.
- `.gitignore`: exclui dados, modelos, ambientes virtuais, caches, checkpoints
  de notebooks, arquivos temporários, pesos serializados e bancos locais.
- Dois `.pyc` antes versionados foram retirados **somente do índice**, preservando
  as cópias locais.

`src/ml/anomaly_preprocessing.py` é mantido exclusivamente porque os arquivos
`preprocessor.joblib` antigos guardam esse caminho da classe `NominalPreprocessor`.
Os imports atuais usam `src.anomaly_detection.preprocessing`; a ponte impede
que a reorganização exija recriar os artefatos. A camada `PositionalEmbedding`
mantém o registro Keras `RUL>PositionalEmbedding` para os Transformers existentes.

Relatórios e JSONs de resultados antigos são registros históricos: podem conter
comandos/caminhos anteriores ou hashes da versão que os gerou. Seu conteúdo não
foi reescrito. Use os comandos deste README e o mapa em
[docs/structural_refactor.md](docs/structural_refactor.md).

## Validação da organização

```powershell
python -m unittest discover -s tests -v
```

A suíte verifica sintaxe, imports, caminhos, métricas, consenso sintético e
compatibilidade dos artefatos locais quando disponíveis. Ela não executa `fit`,
Optuna ou salva modelos. Testes de carregamento de artefatos ausentes são pulados.
Os hashes e as movimentações estão em
`docs/structural_refactor_manifest.json`; os resultados antigos não foram recalculados.

## Etapa 2: avaliação da detecção de anomalias

A avaliação usa os resultados existentes do LSTM Autoencoder. Não treina modelos,
não executa Optuna e não modifica os resultados em `results/anomaly_detection/detection/`.
Execute os comandos na raiz do repositório, no ambiente Python do projeto.

```powershell
# Todos os datasets.
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --datasets FD001 FD002 FD003 FD004

# Somente um dataset.
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --datasets FD001

# Repetir uma avaliação já existente (atualiza somente a nova pasta de avaliação).
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --datasets FD001 FD002 FD003 FD004 --overwrite

# Testes sintéticos, sem carregar ou treinar redes.
python -m unittest discover -s tests -p "test_evaluate_anomaly_detection.py" -v
```

Requer os CSVs e `detection_config.json` em `results/anomaly_detection/detection/FD00X/`.
Utiliza os dados brutos e os artefatos em `data/processed_anomaly/FD00X/` quando
disponíveis. Ausências de artefatos opcionais são registradas no relatório.
Sem dados brutos, o inventário fica limitado aos motores/períodos presentes nos CSVs.
Dados essenciais inconsistentes interrompem a execução com uma mensagem de erro.

### Saídas

Tudo é salvo em `results/anomaly_detection/evaluation/`, separado dos resultados originais:

- `FD00X/motor_metrics.csv`: indicadores por motor, split, percentil e persistência.
- `rul_coverage.csv`: cobertura em grupos **cumulativos** de RUL final.
- `nominal_alerts.csv` e `nominal_motor_metrics.csv`: diagnóstico nominal agregado e por motor.
- `persistence_comparison.csv`: desaparecimento/atraso de alertas ao passar de 3 para 5 janelas.
- `threshold_sensitivity.csv`: percentis 95, 97, 99 e 99,5, somente com calibração verificável.
- `regime_analysis.csv`, `regime_windows.csv` e `transition_profile.csv`: análise dos regimes em FD002/FD004.
- `evaluation_summary.csv` e `evaluation_config.json`: resumo, parâmetros, hashes e auditoria.
- `evaluated_windows_p99_k3.csv`: baseline recalculado; figuras PNG a 300 dpi.
- `comparison/`: tabelas reunidas e gráficos comparativos dos datasets solicitados na execução.
- `evaluation_report.md`: resultados e limitações metodológicas.

Uma execução individual atualiza o relatório/comparação para o dataset solicitado;
as pastas dos demais datasets são preservadas. Use os quatro datasets no comando
para reconstruir a comparação completa. `--overwrite` substitui arquivos de mesmo nome
da avaliação, mas nunca os modelos nem os CSVs originais.

### Definições e cuidados

- O baseline é P99 da validação nominal e 3 janelas consecutivas acima do threshold.
  A versão vetorizada da persistência é testada contra a função original, inclusive
  com lacunas, motores distintos e reinicialização por split. A auditoria também
  compara os indicadores recalculados com os CSVs de detecção existentes.
- A cobertura usa motores com ao menos `k` janelas consecutivas disponíveis.
  Motores insuficientes são separados de motores elegíveis sem alerta. Ausências
  de alerta/RUL permanecem vazias nos CSVs (NaN), não são substituídas por zero.
- O RUL no alerta do teste é `RUL_final + ultimo_ciclo_observado - ciclo_alerta`.
  Último ciclo observado não significa falha no teste truncado.
- O diagnóstico nominal considera apenas janelas inteiramente nos primeiros 30%
  da **vida completa de treino**. O denominador nominal é o número de motores com
  pelo menos uma janela nominal completa. A divisão `fit_train`/`validation` é
  recuperada dos artefatos salvos. Não se aplica essa hipótese ao teste truncado.
- Os percentis adicionais vêm do cruzamento dos metadados nominais de validação
  com seus erros já salvos. Hashes, separação dos motores e reprodução do P99 são
  verificados. Nenhum percentil é calculado sobre o teste. RUL não entra na calibração.
- A mesma validação foi usada no desenvolvimento do AE; portanto não é uma
  calibração independente. Os resultados não selecionam a melhor configuração.
- Os regimes vêm do KMeans salvo, sem novo ajuste. Mudança interna está entre o
  segundo e o último ciclo da janela; proximidade ao fim usa os três ciclos finais.
  As associações não demonstram causalidade. Janelas e eventos se sobrepõem.
- Cobertura não é acurácia/recall de falhas; antecedência não é início da degradação.

### Recuperação opcional da calibração (somente inferência)

Normalmente **não é necessária**: os CSVs existentes já permitem recuperar os erros
nominais de validação. Se eles não contiverem essas janelas, mas o modelo e os
artefatos originais existirem, execute separadamente:

```powershell
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --datasets FD001 --recover-calibration-only
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --datasets FD001 --overwrite
```

Essa opção apenas carrega o modelo salvo e reconstrói `X_val` nominal já salvo,
verificando hashes e o P99 original. Não ajusta modelo, normalizador ou KMeans,
não salva o modelo novamente e não inicia a avaliação na mesma execução.
Se a origem da calibração não puder ser confirmada, a análise principal mantém
somente o P99 original e registra a omissão dos percentis adicionais.
