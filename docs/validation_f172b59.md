# Validação final da refatoração — f172b59

Validação realizada em 02/11/2026 sobre `f172b59` (`organizacao das pastas`).
O diretório de trabalho estava limpo no início. Ambiente: CPython 3.13.14,
Windows 11 x64, TensorFlow 2.20.0, Keras 3.15.0.

## Resultado

| Verificação | Aprovados | Pulados | Falhas |
| --- | ---: | ---: | ---: |
| Suíte `unittest discover -s tests -v` | 21 | 0 | 0 |
| Carregamento adicional de todos os arquivos Keras locais | 17 | 0 | 0 |
| Carregamento adicional de todos os arquivos joblib locais | 8 | 0 | 0 |
| Interfaces `--help` dos módulos Autoencoder | 3 | 0 | 0 |

Os carregamentos adicionais são verificações fora da suíte, não novos casos
`unittest`. Nenhum treinamento, busca Optuna, salvamento de modelo ou execução
dos notebooks foi realizado.

## Escopo dos testes e artefatos

- A suíte verificou imports dos módulos principais, compilação das fontes Python
  e das células dos sete notebooks, caminhos, métricas RUL, consenso sintético,
  persistência, cobertura, RUL dos alertas e compatibilidade do pré-processamento.
- Foram carregados, com `compile=False`, os quatro campeões de cada família
  (CNN, CNN-LSTM, Transformer e Autoencoder) e o modelo temporário que já existia
  em `models/temp_stability/`: 17 arquivos `.keras`.
- Os quatro `preprocessor.joblib` de FD001–FD004 foram carregados e reconhecidos
  como `NominalPreprocessor`. A ponte `src.ml.anomaly_preprocessing` funcionou.
- Os quatro baselines joblib (dois SVR e dois XGBoost) também foram carregados.
- Durante essa checagem adicional, os hashes SHA-256 dos 278 arquivos presentes
  em `data/`, `models/` e `results/` foram comparados antes/depois: nenhuma mudança.
- `PositionalEmbedding` manteve o registro `RUL>PositionalEmbedding`.

Carregar modelos confirma compatibilidade de desserialização e seus shapes;
não equivale a repetir a avaliação preditiva ou os experimentos XAI.

## Imports, caminhos e documentação

Os imports reorganizados e os caminhos exercitados pela suíte passaram. A raiz
é obtida por `src.common.paths.ROOT`; os caminhos dos resultados apontam para
`results/anomaly_detection/{detection,evaluation}/` e `results/xai/`.
As referências antigas em código ativo limitam-se à ponte intencional para
joblib e ao teste dessa compatibilidade. Registros históricos não foram editados.

Os comandos de execução do README correspondem aos módulos atuais. As interfaces
abaixo encerraram com código zero:

```powershell
python -m src.anomaly_detection.training.tune_autoencoder --help
python -m src.anomaly_detection.evaluation.detect_anomalies --help
python -m src.anomaly_detection.evaluation.evaluate_anomaly_detection --help
```

Os scripts RUL e de estabilidade XAI não oferecem `argparse`: sua importação
foi validada, mas seus blocos principais não foram executados. Esses comandos
continuam explicitamente identificados no README como experimentos com treino.

## Dependências

O `requirements.txt`, antes vazio, passou a listar versões instaladas das
dependências do código e dos notebooks, incluindo as versões relevantes para
serialização e XAI. Não se exportou indiscriminadamente todo o ambiente:
PyTorch, Captum e Quantus instalados localmente não aparecem nos imports atuais
dos scripts e notebooks deste commit e não foram incluídos.

O notebook `solucao_paper_prepared_for_lift_off.ipynb` importa `statsmodels`
e `pwlf`, ausentes no ambiente. Foram acrescentados `statsmodels==0.15.0`
e `pwlf==2.7.0`, com wheels/resolução compatíveis com o Python utilizado.
São as únicas dependências diretas listadas que não estavam instaladas;
seus imports e a execução desse notebook permanecem pendentes após instalação.

Verificações executadas:

```powershell
python -m pip check
python -m pip install --dry-run --no-deps --only-binary=:all: statsmodels pwlf
python -m pip install --dry-run -r requirements.txt
```

O `pip check` não encontrou dependências quebradas. A resolução do requirements
terminou com código zero e propôs apenas as duas dependências ausentes e suas
dependências transitivas (`formulaic`, `interface_meta` e `patsy`). Nenhum pacote
foi instalado ou atualizado. Não foi testada uma instalação em ambiente limpo;
o arquivo fixa o ambiente usado e não constitui um lock completo de transitivas.

## Avisos e limitações técnicas

- Os baselines XGBoost foram carregados, mas houve aviso de serialização de
  uma versão anterior. Manter os originais; uma eventual migração de formato
  deve ser feita em tarefa própria, usando o ambiente que os produziu.
- Há avisos de depreciação no caminho `optuna.integration.tfkeras`, na integração
  TimeSHAP/Altair e no TensorFlow/Keras. Não causaram falhas nesta validação.
- O notebook de referência contém `sep='\s+'`, que gera `SyntaxWarning` em
  Python 3.13. A compilação passou; o conteúdo científico não foi alterado.
- O teste de sintaxe não executa imports/células dos notebooks. Em particular,
  as duas dependências ausentes do notebook do artigo foram identificadas
  separadamente, sem confundir isso com testes pulados ou aprovados de execução.

Não foi encontrada falha funcional da refatoração que exigisse alterar `src/`
ou os cálculos dos testes. As alterações desta validação são somente
`requirements.txt`, a seção de ambiente do README e este relatório.

## Observações metodológicas — sem alterações

- RUL utiliza `validation_split=0.2` sobre janelas. Isso, isoladamente, não
  garante separação por motor; essa propriedade exige auditoria dos metadados
  e da divisão dos dados em uma etapa científica específica.
- No Autoencoder, a validação nominal é usada no desenvolvimento do modelo
  e na calibração do threshold. Ela não constitui calibração independente.
- Os primeiros 30% como região nominal são uma hipótese operacional;
  cobertura de alertas não comprova o instante de início da degradação.
- Janelas sobrepostas e as seeds de estabilidade não representam observações
  independentes da mesma natureza. Interpretar as métricas conforme o protocolo
  documentado, sem inferir causalidade ou generalização apenas destes testes.

Esses pontos foram apenas registrados. Nenhum protocolo RUL, Autoencoder ou
XAI, arquitetura, hiperparâmetro, seed, métrica ou resultado foi modificado.
