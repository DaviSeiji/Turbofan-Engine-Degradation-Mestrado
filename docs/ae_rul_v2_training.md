# Etapa E — Optuna e treinamento do LSTM Autoencoder coordenado

## Escopo e independência

Implementação em `src/anomaly_detection/v2/{experiment,train}.py`, separada do
AE histórico e das baselines RUL. **Nenhum treinamento, estudo Optuna real,
inferência ou reprocessamento C-MAPSS foi executado nesta entrega.**

Pré-requisito: artefatos da [Etapa C](ae_rul_v2_coordinated_preprocessing.md)
já preparados e validados em `data/processed_anomaly_v2/FD00X/`. O treinamento
não modifica esses arquivos, não recalcula a preparação e não ajusta novamente
scalers/KMeans. Reutiliza a validação estrutural da Etapa C, que compara a
divisão salva com sua regra determinística, sem salvar novos splits.

Entradas de treinamento exclusivamente:

- `nominal_fit.npz`: janelas nominais dos motores `ae_fit`.
- `nominal_inner_validation.npz`: janelas nominais de outros motores, dentro
  de RUL train, usadas na seleção de hiperparâmetros e no EarlyStopping.

Ambos contêm somente `X` e a ordem de `features`. O target é **o próprio X**:
`fit(X_fit, X_fit, validation_data=(X_inner_val, X_inner_val))`.
O treinamento final mantém os dois grupos separados; não existe
`validation_split` nem concatenação dos grupos.

JSONs/CSVs, joblib e hashes estruturais AE são verificados para autenticar
identidade, features, shapes, finitude, cobertura e motores permitidos. A
referência RUL abre somente seus três JSONs estruturais, nunca arrays ou
rótulos RUL. Os números de motores de teste/reserva nas listas servem apenas
às verificações de identidade/disjunção, não a ajustes ou seleção.

Não se abrem dados brutos, `calibration.npz`, `test_terminal.npz`, outros NPZ
RUL, nem trajetórias completas de degradação. Nenhum `y` ou `y_original`
é fornecido ao AE. Consulte também a [auditoria](ae_rul_v2_compatibility_audit.md)
e o [protocolo RUL v2](rul_protocol_v2.md).

## Arquitetura e objetivo

Reutiliza, sem alterações, `build_lstm_autoencoder` em
`src/anomaly_detection/models/autoencoder_models.py`:

janela `(30, F_AE)` → encoder LSTM → Dense latente linear → RepeatVector →
decoder LSTM → TimeDistributed Dense linear → reconstrução `(30, F_AE)`.

O número de camadas é 1 ou 2 e o dropout segue o builder original. Adam e
loss MSE também vêm do builder. Uma checagem exige que entrada e saída tenham
exatamente o mesmo shape, com o número de features específico do FD.

O objetivo é **o menor `val_loss` do histórico**, MSE de reconstrução nominal
das janelas internas. Não é RMSE/MAE de RUL, nem taxa de detecção de falhas,
nem média com igual peso por motor. Cada janela participa da loss Keras.

## Espaço de busca e opções

| Parâmetro | Valores |
| --- | --- |
| latent_dim | 8, 16, 32, 64 |
| encoder_units | 32, 64, 128 |
| num_layers | 1, 2 |
| dropout_rate | 0.0, 0.1, 0.2, 0.3 |
| learning_rate | 1e-4 a 1e-2, log |
| batch_size | 32, 64, 128 |

Um estudo SQLite por dataset/experimento. Padrões: **30 trials totais**, até
50 épocas por trial, TPE seed 42, MedianPruner com `n_warmup_steps=5` e
`TFKerasPruningCallback` de `optuna-integration.tfkeras`. EarlyStopping monitora
`val_loss`, patience 5 e `restore_best_weights=True`. Os parâmetros históricos
não são automaticamente enfileirados, copiados ou escolhidos como campeão.

O comando final é independente, usa um snapshot campeão explícito, por padrão
50 épocas, patience 5 e seed final 42 (configuráveis). Reconstrói o modelo do
zero, sem continuar pesos de um trial. A seleção da época final utiliza apenas
a validação interna nominal; nenhum threshold P99 é calculado.

## Comandos manuais — começar por FD001

Executar na raiz do repositório, no ambiente Python do projeto. **Os comandos
`optimize` e `final` abaixo executarão treinamento quando o usuário os chamar.**

```powershell
# Ajuda e inspeção: não treinam, nem importam TensorFlow/Optuna.
python -m src.anomaly_detection.v2.train --help
python -m src.anomaly_detection.v2.train optimize --help
python -m src.anomaly_detection.v2.train final --help
python -m src.anomaly_detection.v2.train inspect --datasets FD001

# Busca somente no FD001.
python -m src.anomaly_detection.v2.train optimize --datasets FD001 --experiment baseline --seed 42 --trials 30 --epochs 50 --patience 5
```

Ao terminar a busca, o terminal informa `best_trial_N.json`, com número,
hiperparâmetros, MSE interno, orçamento concluído e identidade da configuração.
Não se assume que N seja 0. Para ler o campeão e executar o treino final:

```powershell
# Cada experimento concluído publica um único snapshot campeão.
$campeaoFD001 = Get-Content -Encoding UTF8 -Raw models/anomaly_v2/FD001/baseline/best_trial_*.json | ConvertFrom-Json
$campeaoFD001
python -m src.anomaly_detection.v2.train inspect --datasets FD001 --experiment baseline --best-trial $campeaoFD001.trial

# Executar só após revisar o campeão. Não executa nova busca.
python -m src.anomaly_detection.v2.train final --datasets FD001 --experiment baseline --best-trial $campeaoFD001.trial --seed 42 --epochs 50 --patience 5

# Conferir hashes e carregar o .keras, SEM predict, fit ou novo save.
python -m src.anomaly_detection.v2.train inspect --datasets FD001 --experiment baseline --model-seed 42
```

Para os demais FDs, depois de conferir FD001:

```powershell
python -m src.anomaly_detection.v2.train inspect --datasets FD002 FD003 FD004
python -m src.anomaly_detection.v2.train optimize --datasets FD002 FD003 FD004 --experiment baseline --seed 42 --trials 30 --epochs 50 --patience 5

# Cada FD pode ter um número de trial campeão diferente.
foreach ($datasetAE in @('FD002', 'FD003', 'FD004')) {
    $campeaoAE = Get-Content -Encoding UTF8 -Raw "models/anomaly_v2/$datasetAE/baseline/best_trial_*.json" | ConvertFrom-Json
    python -m src.anomaly_detection.v2.train final --datasets $datasetAE --experiment baseline --best-trial $campeaoAE.trial --seed 42 --epochs 50 --patience 5
    if ($LASTEXITCODE -ne 0) { throw "Falha no treinamento final de $datasetAE" }
}
python -m src.anomaly_detection.v2.train inspect --datasets FD002 FD003 FD004 --experiment baseline --model-seed 42
```

`--data-dir` e `--rul-dir` permitem referências preparadas em outra localização,
mas não fazem preparação. Os destinos de modelos/resultados são fixos na CLI.
As APIs internas aceitam pastas temporárias externas para testes; dentro do
repositório, somente os dois destinos AE v2 exclusivos são aceitos.

## Identidade, retomada e orçamento

`study_config.json` e o atributo `protocol` do estudo congelam:

- Dataset, SHA256 do manifesto AE, hashes dos nove artefatos e referência RUL.
- Listas de motores herdadas e internas, configuração, features e shapes.
- Arquitetura, espaço de busca, seed, orçamento, épocas, patience e objetivo.
- Versões de Python, TensorFlow/Keras, Optuna/integration, NumPy,
  scikit-learn/joblib/pandas, Matplotlib e SQLAlchemy.
- Hashes do builder, código AE v2 e helpers compartilhados efetivamente usados.

Alterar essas informações recusa a retomada e o treino final antes de qualquer
`fit()`. Use outro `--experiment`, por exemplo `baseline_v2`, para mudar o
orçamento, código, bibliotecas ou protocolo. O estudo histórico não é retomado.

Para continuar uma busca interrompida, repetir **exatamente** o mesmo comando
`optimize`. COMPLETE, PRUNED e FAIL consomem o orçamento. Se já existem 20
trials terminais com orçamento 30, somente 10 serão solicitados. Uma busca
concluída não adiciona trials quando chamada novamente.

RUNNING/WAITING bloqueiam a retomada. Não se presume que um trial RUNNING
esteja morto apenas por ser antigo. Após confirmar que o processo anterior
terminou, `--fail-interrupted` pode marcar RUNNING como FAIL, sem devolver
seu orçamento. WAITING continua recusado; esta CLI não enfileira trials.

```powershell
# SOMENTE após confirmar que o processo anterior não está mais ativo:
python -m src.anomaly_detection.v2.train optimize --datasets FD001 --experiment baseline --seed 42 --trials 30 --epochs 50 --patience 5 --fail-interrupted
```

O `.lock` exclusivo bloqueia dois processos do mesmo estudo, inclusive na fase
final. Uma exceção normal remove o lock; um encerramento abrupto pode deixá-lo.
Nesse caso, conferir o processo e remover manualmente **somente esse lock**,
nunca o banco/modelo ou a pasta do experimento para contornar a proteção.
Não há execução distribuída/paralela de workers.

Erros não tratados podem encerrar uma chamada Optuna, deixando um trial FAIL
contabilizado. Após investigar a causa e mantendo a configuração compatível,
é possível repetir o comando para completar os trials restantes. Uma busca
sem nenhum COMPLETE não publica campeão. O snapshot é publicado somente
depois de consumir todo o orçamento, com todos os trials terminais.

O SQLite é mutável durante a busca/retomada. Configurações, snapshots e saídas
finais não são sobrescritos. Se o snapshot já existe, seu conteúdo deve ser
idêntico. Um banco sem `study_config.json` não é adotado automaticamente.
Os metadados do banco precisam concordar com os do arquivo.

## Artefatos finais e recarregamento

```text
models/anomaly_v2/FD001/baseline/
  study.db
  study_config.json
  best_trial_N.json
  seed_42/
    model.keras
    config.json
    manifest.json
results/anomaly_detection/v2/FD001/baseline/seed_42/
  history.json
  history.png
  summary.json
```

A configuração final registra o estudo inteiro, seed final, épocas, patience,
trial e hiperparâmetros. O resumo registra melhor MSE interno, melhor época e
épocas executadas. O manifesto final identifica o pré-processamento e contém
SHA256 de modelo, configuração, histórico, gráfico e resumo. É escrito por
último: uma pasta parcial não representa uma execução concluída.

Pasta de seed existente em **qualquer um** dos destinos é recusada, mesmo vazia
ou parcial. Não existe `--overwrite` nem retomada de pesos de treino final.
Outra seed é outro ajuste, não uma forma de continuar o anterior. Se for
necessário repetir exatamente a mesma seed, use um experimento distinto e
conclua uma busca compatível nele; não apague resultados antigos.

`inspect --model-seed 42` verifica pré-processamento/configuração/snapshot e os
hashes finais, depois usa `keras.models.load_model(..., compile=False)` e
confere os shapes. Não calcula reconstruções ou scores e não salva novamente.
O builder não tem camadas customizadas que exijam registro adicional.

Uso em Python para a futura etapa, ainda sem inferência:

```python
from src.anomaly_detection.v2.experiment import (
    DATA_ROOT, MODEL_ROOT, RESULT_ROOT, load_frozen_model,
)
model = load_frozen_model(DATA_ROOT, MODEL_ROOT, RESULT_ROOT,
                          'FD001', 'baseline', 42)
```

Somente modelos/joblib locais confiáveis devem ser carregados. Hashes
comprovam integridade relativamente aos manifestos, não autenticidade de
artefatos de terceiros.

## Reprodutibilidade e limites metodológicos

Python, NumPy e Keras recebem seed; operações determinísticas são solicitadas
quando suportadas. A seed do trial é `(sampler_seed + trial.number) % 2**32`.
A sessão Keras é limpa e o coletor de memória é chamado entre modelos,
inclusive se o ajuste falhar. O treino final usa sua própria seed configurada.

Essas medidas não garantem igualdade numérica entre CPUs/GPUs, bibliotecas e
drivers diferentes. Uma API de determinismo indisponível gera aviso; uma
operação sem implementação determinística pode falhar, sem fallback silencioso.
Recriar o TPESampler ao retomar também não promete reproduzir a mesma sequência
de sugestões de uma busca ininterrupta.

Os primeiros 30% são hipótese nominal retrospectiva, não rótulo de saúde
garantido. A validação interna foi usada na busca e no EarlyStopping final:
seu MSE é uma medida de seleção/desenvolvimento, **não uma avaliação externa
independente** de detecção. As janelas se sobrepõem dentro de cada motor; os
motores, não as janelas aleatórias, permanecem separados entre fit/validação.

Um futuro score contínuo será calculado por janela com AE/preprocessador
congelados sobre sensores brutos alinhados aos metadados RUL. Esta etapa não
implementa scoring, P99, alertas, métricas de anomalia, fusão, cross-fitting ou
XAI. A diferença entre scores in-sample/out-of-sample e o papel da reserva
continuam decisões futuras, conforme a auditoria. Nenhum protocolo RUL foi
alterado, nem se usou o teste para selecionar o AE.

## Testes desta entrega

```powershell
python -B -m unittest tests.anomaly_detection.test_v2_training tests.anomaly_detection.test_coordinated_preprocessing tests.rul_prediction.test_v2_preprocessing -v
```

A suíte nova usa fixtures sintéticas temporárias e mocks de Keras/Optuna,
incluindo SQLite simulado. Não roda estudos reais. Exercita leitura nominal,
isolamento de dados externos, `fit(X,X)`, shapes por FD, espaço de busca,
objetivo mínimo de validação, callbacks, seeds, limpeza após falhas, orçamento,
RUNNING/WAITING, compatibilidade, locks, snapshots, artefatos finais,
sobrescrita e carregamento simulado sem inferência. As CLIs `--help` e
`inspect` também são testadas com imports TensorFlow/Optuna bloqueados.

Resultado: **67 testes aprovados**, sem falhas ou testes pulados: 28 novos
testes de treinamento simulado, 23 da preparação AE coordenada e 16 da
preparação RUL v2. Uma checagem antiga de caminho foi isolada com mock para
não depender da ausência de FD001 já processado no computador do usuário;
nenhum dado ou código de pré-processamento foi alterado.

O teste de `.keras` usa um arquivo substituto e um loader mockado: **não é uma
verificação de round-trip de pesos reais**. O builder histórico foi preservado.
O comando manual `inspect --model-seed` permite verificar o primeiro modelo
real depois que o usuário concluir seu treinamento.

Próximo passo: revisar o código, inspecionar FD001, executar sua busca e treino
final manualmente e conferir o carregamento. Depois ampliar para FD002–FD004.
A geração de scores congelados será uma etapa independente.
