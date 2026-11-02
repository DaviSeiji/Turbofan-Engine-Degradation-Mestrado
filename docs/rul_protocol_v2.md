# RUL v2 — pré-processamento, treinamento e avaliação

A etapa 1 implementa preparação e persistência de dados. A etapa 2 acrescenta
treinamento e avaliação independentes, descritos ao final deste documento.
Os dados da etapa 1 já foram processados e validados manualmente pelo pesquisador.
A implementação da etapa 2 foi verificada somente com dados sintéticos e mocks:
nenhuma rede, estudo Optuna ou avaliação real foi executado nesta entrega.

## Implementação independente e preservação da v1

O código novo está em `src/rul_prediction/v2/`:

- `utils.py`: configuração, divisão de motores, validação de trajetórias e classe
  serializável `RULPreprocessor`.
- `preprocessing.py`: leitura, RUL, janelas, gravação, validação de artefatos e CLI.
- `__init__.py`: identificação do pacote, sem executar processamento.
- `experiment.py`: carregamento por escopo, identidade e proteção de artefatos.
- `train.py`: inspeção, Optuna persistente e treinamento final explícito.
- `evaluation.py`: teste terminal, teste por janelas e comparação por seeds.

A v1 continua em `src/common/preprocessing/preproc_data_nasa.py`, com os mesmos
scripts e artefatos. Os scripts atuais de CNN-LSTM/Transformer ainda leem
`data/processed_data/`: **não passam a utilizar v2 automaticamente**.
O treinamento v2 usa comandos próprios e arrays explícitos de treino e validação,
em vez de `validation_split` sobre um array combinado.

## Divisão de motores

O teste oficial é mantido externo. Somente os motores do arquivo original de
treinamento são divididos, antes da seleção de features e de qualquer ajuste:

| Partição | Fração padrão | Papel nesta etapa |
| --- | ---: | --- |
| `train` | 70% | Única origem para seleção de features, scalers e KMeans |
| `validation` | 15% | Dados preparados para futura seleção de hiperparâmetros/EarlyStopping |
| `calibration` | 15% | Reserva preparada e armazenada; não utilizada para ajustes ou decisões |

As frações devem ser estritamente positivas e somar 1; são configuráveis na CLI.
A divisão usa IDs ordenados, `numpy.random.default_rng(seed=42)` e uma permutação.
Os tamanhos são calculados pelo método do maior resto: parte inteira de cada
fração, seguida da distribuição das vagas restantes pelos maiores restos.
Empates seguem a ordem treino, validação, reserva. Nenhuma divisão pode ficar vazia.
Assim, conjuntos pequenos podem não ter percentuais exatos; os IDs efetivos são
salvos em `splits.json` e não devem ser reconstruídos apenas pelas porcentagens.

Cada motor entra integralmente em apenas uma das três partições. Dentro de cada
partição, as janelas são ordenadas por motor e ciclo. Os IDs numéricos do teste
podem coincidir com IDs de desenvolvimento: `SourceSplit` distingue os arquivos.

Não há estratificação por modo de falha. Os rótulos de modo não são inferidos.
A adequação da composição dos grupos deverá ser revisada após a execução real,
sem escolher a divisão pelo desempenho no teste.

### Papel ainda provisório da reserva

A reserva recebe somente `transform` com os objetos do treino. Seus arrays e
rótulos são armazenados para manter rastreabilidade, mas não influenciam features,
escalas, clusters, thresholds, seleção ou métricas nesta etapa.
Isso **não calibra o Autoencoder** e não define ainda seu protocolo nominal.
Seu papel definitivo será decidido ao coordenar as partições de RUL e anomalias.

## Dados e validação de entrada

O leitor exige os arquivos `train_FD00X.txt`, `test_FD00X.txt` e `RUL_FD00X.txt`.
As trajetórias devem conter exatamente as 26 colunas C-MAPSS. Cada motor precisa
ter IDs/ciclos inteiros positivos, começar no ciclo 1 e ter ciclos consecutivos.
As linhas são ordenadas por motor e ciclo. Dados ausentes, Inf/NaN, duplicatas
e lacunas geram erro: não há remoção silenciosa de linhas.

O vetor externo de RUL associa a linha 1 ao motor de teste 1, e assim por diante;
o conjunto de IDs deve corresponder exatamente ao arquivo de teste. Não há
suposição fixa de 248/249 motores em FD004: valem os arquivos efetivamente lidos.

## Features, regimes e normalização

A lista candidata contém os três settings e os 21 sensores, nessa ordem.
`unit_nr`, ciclos e RUL nunca são entradas do normalizador ou features de `X`.
A seleção utiliza apenas a amplitude observada no treino: manter a coluna se
`max − min > 1e-12`. A tolerância absoluta é configurável e registrada; evita
reter uma constante devido a resíduos numéricos de desvio padrão.
Uma feature constante no treino permanece excluída mesmo se variar no holdout.

| Dataset | Tratamento fixado pelo protocolo |
| --- | --- |
| FD001 | MinMax global, sem KMeans |
| FD002 | KMeans com seis clusters e MinMax por regime |
| FD003 | MinMax global, sem KMeans |
| FD004 | KMeans com seis clusters e MinMax por regime |

Não existe detector baseado no número de pequenas variações dos settings.
Em FD002/FD004, KMeans usa os três ajustes operacionais **brutos dos motores de
treino**, `n_clusters=6`, `n_init=10` e a seed configurada. A rotina exige seis
condições distintas e seis clusters ocupados; não inventa regimes se os dados
de ajuste forem insuficientes.

As features são selecionadas globalmente no treino, mantendo a mesma dimensão
e ordem em todos os regimes e partições. Cada scaler é ajustado somente nas
linhas de treino atribuídas àquele regime. Nos demais conjuntos, há apenas
`KMeans.predict` e `MinMaxScaler.transform`. IDs dos clusters são rótulos internos
arbitrários, não nomes físicos de regimes.

O normalizador permite um único `fit`; uma tentativa de reajuste do objeto já
ajustado gera erro. Validação, reserva e teste não participam do ajuste.
Não há clipping de entradas: valores de holdout fora de `[0,1]` são preservados.
Arrays e transformações são verificados quanto a valores finitos; a conversão
para `float32` também deve permanecer finita.

## RUL e janelas de 30 ciclos

- Treino completo: `RUL_original = último_ciclo_do_motor − ciclo_atual`.
- Teste truncado: `RUL_original = último_ciclo_observado + RUL_final_fornecido − ciclo_atual`.
- Target padrão: `RUL_capped = min(RUL_original, 125)`.

O cap padrão é 125, configurável por `--rul-cap` e registrado. O RUL original é
preservado, inclusive para os motores de teste cujo RUL final excede o teto.
Nenhum dos dois alvos entra na seleção de features, no MinMax ou no KMeans.

Todas as janelas têm shape `(30, n_features)` e target no ciclo final.
As janelas são construídas separadamente por motor/partição, com passo de um ciclo.
Sobreposição dentro da mesma partição é esperada; não há motor compartilhado
entre os três conjuntos de desenvolvimento.

### Dois protocolos separados para o teste

| Saída | Composição | Uso futuro |
| --- | --- | --- |
| `test_terminal` | Exatamente a última janela de cada motor | Avaliação principal, uma previsão por motor |
| `test_windows` | Todas as janelas disponíveis de cada motor | Avaliação complementar ao longo da trajetória |

A janela terminal também aparece como última janela do protocolo complementar;
isso é intencional, pois são duas representações do mesmo teste externo, não dois
conjuntos independentes. O validador confere essa correspondência nos metadados.
As representações não devem ser somadas em uma única métrica.

### Trajetórias curtas

Um motor com menos de 30 ciclos gera uma única janela com repetição do primeiro
ciclo à esquerda, em qualquer partição. Não são usados dados de outro motor ou
ciclos futuros. Os metadados incluem `is_padded`, `padding_cycles` e
`observed_cycles`. `start_cycle` e `end_cycle` são ciclos **reais**, por exemplo
1 e 19 para um motor com 19 observações; os 11 ciclos repetidos não recebem
tempos negativos artificiais. A política deve ser mantida igual entre os modelos
comparados e não implica que o AE existente tenha um score para essa janela.

## Artefatos futuros

Quando o pesquisador executar o processamento, cada dataset terá:

```text
data/processed_rul_v2/
└── FD001/                       # Também FD002, FD003 e FD004 quando solicitados
    ├── config.json
    ├── splits.json
    ├── features.json
    ├── preprocessor.joblib
    ├── train.npz
    ├── train_metadata.csv
    ├── validation.npz
    ├── validation_metadata.csv
    ├── calibration.npz
    ├── calibration_metadata.csv
    ├── test_terminal.npz
    ├── test_terminal_metadata.csv
    ├── test_windows.npz
    ├── test_windows_metadata.csv
    └── manifest.json
```

Os cinco NPZ usam as chaves:

- `X`: `float32`, shape `(n_janelas, 30, n_features)`.
- `y`: RUL limitado, `float32`, target de treinamento.
- `y_original`: ciclos restantes originais, `int64`.
- `features`: nomes na ordem de `X`, sem objetos pickle no NPZ.

O formato é deliberadamente identificado como v2; não usa a chave histórica
`dados`. Os futuros loaders de treinamento deverão ler essas chaves explicitamente.
As linhas dos CSVs correspondem, na mesma ordem, às linhas de seus NPZ.

Metadados por janela:

`Dataset`, `Split`, `SourceSplit`, `unit_nr`, `start_cycle`, `end_cycle`,
`RUL_original`, `RUL_capped`, `regime`, `regime_transition_count`,
`is_padded`, `padding_cycles`, `observed_cycles`.

`Split` identifica a saída (`train`, `validation`, `calibration`, `test_terminal`
ou `test_windows`). `SourceSplit` identifica o arquivo original (`train`/`test`).
`regime` é o regime do **último ciclo**, não um regime atribuído a toda a janela;
em condição única usa 0 como marcador fixo. A contagem de transições considera
os ciclos realmente observados, ignorando a repetição artificial do padding.

`splits.json` lista IDs de treino, validação, reserva e teste. `features.json`
registra colunas selecionadas e removidas. `preprocessor.joblib` contém os
scalers, o KMeans quando aplicável, configuração e IDs usados no ajuste.
A classe é definida em `src.rul_prediction.v2.utils`, mantendo um caminho estável
mesmo quando a preparação é chamada por `python -m`.

O manifesto contém SHA-256 dos três arquivos brutos, dos 14 artefatos gerados
(exceto o próprio manifesto), dos dois módulos de implementação, configuração,
versões de Python/bibliotecas, plataforma, data UTC e contagens por saída.
Também registra quantos valores de `X` ficaram abaixo de 0 ou acima de 1.
Essas contagens incluem repetições em janelas sobrepostas; não são contagens de
observações físicas únicas nem critérios para selecionar configuração.

### Proteção contra sobrescrita

Não há `--overwrite`. Qualquer pasta de dataset existente, até mesmo vazia ou
incompleta, é recusada antes da leitura/ajuste. Os diretórios históricos de dados,
modelos, resultados, código e notebooks são protegidos como destinos.
Todos os destinos da CLI são conferidos antes de começar o primeiro dataset.

A preparação fica em memória antes da gravação. O diretório final é criado de
forma exclusiva; cada arquivo usa criação exclusiva e o manifesto é escrito por
último. Se uma falha ocorrer durante a gravação, pode permanecer uma pasta parcial
sem manifesto: ela é preservada e uma nova execução não a sobrescreve. Inspecione
a falha e use outra pasta de saída. Não há exclusão automática de artefatos.
Uma chamada com vários datasets não é uma transação única: um FD concluído é
preservado se o seguinte falhar.

### Carregamento posterior sem ajuste

```python
from src.rul_prediction.v2.preprocessing import load_preprocessor

processor = load_preprocessor('data/processed_rul_v2/FD001')
# processor.transform(novas_trajetorias)  # Nunca chamar fit novamente.
```

O loader confere os artefatos e retorna o transformador salvo; não refaz ajustes.
A validação lê os arrays/CSVs, portanto pode ter custo de I/O. Hashes demonstram
integridade, não autenticidade: carregue joblib somente de origem confiável.

## CLI para execução manual posterior

Na raiz do projeto, no ambiente indicado no README:

```powershell
python -m src.rul_prediction.v2.preprocessing --help

# Condição única e primeiro dataset com seis regimes.
python -m src.rul_prediction.v2.preprocessing --datasets FD001 FD002 --seed 42

# Demais datasets; preserva as pastas já produzidas acima.
python -m src.rul_prediction.v2.preprocessing --datasets FD003 FD004 --seed 42

# Frações e diretórios explícitos, quando necessário.
python -m src.rul_prediction.v2.preprocessing --datasets FD001 --seed 42 --train-fraction 0.70 --validation-fraction 0.15 --calibration-fraction 0.15 --raw-dir data/raw_data --output-dir data/processed_rul_v2

# Apenas verificar artefatos existentes, sem ler dados brutos nem ajustar nada.
python -m src.rul_prediction.v2.preprocessing --datasets FD001 FD002 FD003 FD004 --validate-only
```

`--validate-only` consulta a configuração salva; não altera seed ou frações dos
artefatos existentes. Para uma nova configuração, selecione outra pasta via
`--output-dir`. Não é preciso repetir os argumentos de configuração para validar.

**Os comandos de processamento real acima não foram executados nesta entrega.**

## Testes realizados nesta etapa

```powershell
python -B -m unittest tests.rul_prediction.test_v2_preprocessing -v
```

Foram usados somente DataFrames sintéticos pequenos e arquivos brutos sintéticos
em `TemporaryDirectory`. Resultado: **16 testes aprovados, zero pulados e zero
falhas**. Os casos verificam:

1. Partições completas, disjuntas, configuráveis e reproduzíveis por motor.
2. Scalers/seleção de features ajustados apenas no treino; holdout fora dos limites.
3. Ausência de KMeans em FD001/FD003 mesmo com ruído nos settings.
4. Seis regimes em FD002/FD004, com uma única chamada de ajuste em linhas de treino.
5. Ordem de features, entrada embaralhada e recusa de colunas ausentes.
6. RUL original/limitado, ciclos, sobreposição interna e limites de motores.
7. Padding e última janela única por motor de teste.
8. Namespace distinto para IDs de teste numericamente iguais aos de desenvolvimento.
9. Salvamento/carregamento sem reajuste e detecção de corrupção por hash.
10. Recusa de sobrescrita e preservação de arquivos históricos de controle.
11. Serialização da CLI e `--validate-only` em processos separados, com fixtures sintéticas.
12. Recarregamento dos scalers/KMeans de FD002/FD004 e reprodução exata das janelas sintéticas salvas, sem reajuste.

Os testes ajustam somente scalers e KMeans dessas fixtures. Não importam modelos
de RUL, não treinam redes e não executam o Autoencoder. A suíte histórica completa
não é necessária para esta etapa e não foi executada.

## Integração futura com o Autoencoder e decisões pendentes

A chave para a janela física será
`(Dataset, SourceSplit, unit_nr, start_cycle, end_cycle)`, com `Split` preservado
para distinguir a partição/protocolo. A mesma janela pode aparecer nas duas
representações de teste; a integração deve escolher uma por avaliação.

Será necessário alinhar as listas de motores entre RUL e AE, preservar os hashes
dos dados brutos e a identidade dos transformadores e decidir como produzir scores
fora do ajuste do AE quando apropriado. Os dois ramos mantêm seus próprios
normalizadores; não se deve passar `X` de RUL diretamente ao AE existente.

Continuam pendentes: o papel definitivo da reserva, a divisão de desenvolvimento
do AE, a calibração independente de flags, a estratégia de scores para motores
curtos e a eventual necessidade de cross-fitting. A reserva não se torna uma
calibração independente válida apenas por possuir esse nome.

Também devem ser revisadas após o processamento manual a representação dos seis
regimes no treino e as distribuições dos grupos. Nenhuma decisão deve ser escolhida
pelo desempenho do teste. Reprodutibilidade numérica exige versões/ambiente
compatíveis; seed fixa não garante identidade em todo hardware ou versão.

O pipeline atual materializa os arrays em memória por dataset. Isso é adequado à
estrutura atual, mas o custo real de RAM/tempo ainda não foi medido. O manifesto
não substitui testes reais de qualidade após a revisão do pesquisador.

As seções anteriores descrevem a etapa 1. A etapa 2 está abaixo; explicações XAI
e fusão com o Autoencoder continuam fora do escopo.

## Etapa 2 — treinamento e avaliação independentes

### Dados e identidade

O treinamento abre **somente** `train.npz`, `validation.npz` e seus metadados.
Confere SHA256 desses arquivos, `config.json`, `features.json`, `splits.json` e
`preprocessor.joblib` (este último é apenas conferido por hash, não desserializado).
Confere ainda features ordenadas, shapes `(N, 30, F)`, arrays finitos, alvos,
cap 125, alinhamento linha a linha, IDs disjuntos de desenvolvimento e padding.
Não reajusta transformadores. `calibration.npz` e seus metadados nunca são abertos.
Os IDs de calibração em `splits.json` servem apenas à checagem de disjunção.

O SHA256 do manifesto completo identifica a versão do pré-processamento. Os
hashes efetivos são conferidos **por escopo**, para não abrir arrays de teste
durante seleção. A avaliação confere somente o protocolo de teste solicitado,
os mesmos arquivos estruturais e a identidade registrada no modelo. A validação
completa da etapa 1 continua disponível, mas não é chamada pelo treinamento.

### Arquiteturas, busca e validação

Reutilizam-se, sem alterações, `build_cnn_lstm` e `build_transformer` de
`src/rul_prediction/models/model_builders.py`. Ambos usam Adam, MSE e MAE,
entrada `(30, F)` e saída linear escalar. A CNN-LSTM é Conv1D → LSTM → Dropout →
Dense(32, ReLU) → Dense(1). O Transformer usa projeção, posição treinável,
encoders com atenção/residuais/normalização, pooling e cabeça de regressão.

Espaços históricos:

| Parâmetro | CNN-LSTM | Transformer |
| --- | --- | --- |
| filters / lstm_units | 32, 64, 128 | — |
| kernel_size | 3 a 5, inteiros | — |
| d_model | — | 32, 64, 128 |
| num_heads | — | 2, 4, 8; todos dividem d_model |
| ff_dim | — | 64, 128, 256 |
| num_layers | — | 1 a 3 |
| dropout_rate | 0,1 a 0,4; passo 0,1 | igual |
| learning_rate | 1e-4 a 1e-2, log | igual |
| batch_size | 32, 64, 128 | igual |

Padrões v2: 30 trials **totais**, até 50 épocas por trial e por treino final,
EarlyStopping em `val_loss`, patience 5, melhores pesos restaurados. Essas
opções são configuráveis nas CLIs. O objetivo é o menor MSE nas **janelas da
validação de motores independentes**, não uma média primeiro agregada por motor.
O treino final usa os mesmos grupos separados: não une treino e validação.
Não existe `validation_split` no pipeline novo.

Cada dataset/arquitetura/experimento possui estudo próprio, TPESampler com seed
padrão 42 e MedianPruner com `n_warmup_steps=5`, via TFKerasPruningCallback.
A seed de cada trial é `(sampler_seed + trial.number) % 2**32`.
A sessão Keras é limpa entre modelos, seeds Python/NumPy/TF são fixadas pelo
Keras e operações determinísticas são solicitadas. Isso não garante igualdade
entre ambientes/hardware diferentes. Reiniciar o TPESampler com a mesma seed
também não reproduz necessariamente a sequência de uma busca sem interrupção.

### Retomada e proteção

`study_config.json` congela protocolo, configuração de treino, orçamento,
dataset, features, SHA do manifesto, versões e hashes do código/arquiteturas.
Uma divergência é recusada **antes de abrir o banco SQLite**. Para mudar dados,
código, orçamento ou opções congeladas, use um novo `--experiment`.

COMPLETE, PRUNED e FAIL consomem orçamento. Com 20 trials contabilizados e
orçamento 30, são solicitados apenas 10. RUNNING e WAITING bloqueiam a retomada.
Após confirmar que não existe processo ativo, `--fail-interrupted` permite
marcar RUNNING como FAIL; esses trials continuam contabilizados. WAITING é
recusado (esta CLI não enfileira trials). Não há execução paralela de workers.

O arquivo `.lock` evita dois processos simultâneos no mesmo experimento. Uma
interrupção normal o remove. Se houver encerramento abrupto, confira primeiro
que o processo terminou e remova **apenas o `.lock` daquele experimento** antes
de retomar. Não apague bancos ou pastas de resultados para contornar erros.

O campeão gera um snapshot imutável `best_trial_N.json`; `N` é mostrado no
terminal ao concluir o Optuna. A retomada pode gerar outro snapshot campeão.
O comando `final --best-trial N` usa explicitamente aquele snapshot, sem abrir
o teste nem reexecutar o Optuna. Use o campeão informado pela busca concluída.
O SQLite é o único artefato científico mutável durante a retomada normal.

Cada seed final grava modelo, configuração com hiperparâmetros, histórico,
curva de MSE e manifesto com hashes e menor loss de validação. Pastas de seed
existentes são recusadas, inclusive incompletas; não existe `--overwrite`.
O mesmo vale para avaliações e tabelas comparativas. Falhas podem deixar uma
pasta parcial protegida, que exige inspeção manual antes de nova tentativa.

### Avaliação e comparação

O comando `terminal` usa `test_terminal.npz`, exatamente uma previsão por motor,
e compara com **`y_original` sem cap**. O comando `windows` usa as janelas de
`test_windows.npz`, também com `y_original`, em outra pasta. Não há clipping
de target ou previsão. Treinar com alvo limitado e avaliar sem cap é uma decisão
explícita do protocolo: erros acima de 125 também contam; não compare diretamente
esses números com métricas históricas calculadas com target limitado.

Ambos calculam RMSE, MAE, mediana, percentil 95 (interpolação linear do NumPy)
e máximo do erro absoluto. Terminal dá o mesmo peso a cada motor. Por janelas,
motores longos contribuem mais; esses resultados são complementares, não devem
ser misturados com os terminais. CSVs preservam todos os metadados, incluindo
`Dataset, SourceSplit, Split, unit_nr, start_cycle, end_cycle`, além de Model,
Seed, Experiment, RUL_true, RUL_pred e AbsoluteError. Um scatter real/predito é
salvo, sem exibição interativa.

`compare` exige as duas arquiteturas, todas as seeds solicitadas e exatamente
o mesmo manifesto de dados e chaves/targets de teste, além de orçamento, épocas,
patience, seed do sampler e versões/código do protocolo iguais. Não descarta seeds
ausentes: falha explicitamente. Grava individuais e média/desvio padrão amostral
(`ddof=1`) por dataset/modelo/protocolo. Com uma seed, o desvio não é estimável
e fica vazio no CSV. Não seleciona melhor seed por teste nem agrega FDs distintos.

Organização futura:

```text
models/rul_v2/FD001/cnn_lstm/baseline/
  study_config.json
  study.db
  best_trial_N.json
  seed_1/{model.keras,config.json,history.json,history.png,manifest.json}
results/rul_prediction/v2/FD001/cnn_lstm/baseline/seed_1/
  terminal/{predictions.csv,predictions.png,metrics.json,manifest.json}
  windows/{predictions.csv,predictions.png,metrics.json,manifest.json}
results/rul_prediction/v2/comparisons/comparacao_v2/terminal/
  individual.csv
  summary.csv
  config.json
```

### Comandos manuais — não executados nesta implementação

Na raiz do projeto, com o ambiente de dependências ativado. Comece em FD001;
depois repita para FD002, FD003 e FD004, nesta ordem. Não reprocesse os dados.

```powershell
# Ajuda e inspeção sem importar TensorFlow nem criar estudos
python -m src.rul_prediction.v2.train --help
python -m src.rul_prediction.v2.train inspect --datasets FD001
python -m src.rul_prediction.v2.evaluation --help

# Otimização separada por arquitetura (30 trials totais cada)
python -m src.rul_prediction.v2.train optimize --datasets FD001 --model cnn_lstm --experiment baseline --seed 42 --trials 30 --epochs 50 --patience 5
python -m src.rul_prediction.v2.train optimize --datasets FD001 --model transformer --experiment baseline --seed 42 --trials 30 --epochs 50 --patience 5
```

Após cada busca, anote o número `N` do campeão. No PowerShell, informe os números
reais (podem ser diferentes para cada arquitetura/dataset), sem assumir trial 0:

```powershell
$cnnTrial = [int](Read-Host 'Numero N de best_trial_N.json da CNN-LSTM FD001')
$transformerTrial = [int](Read-Host 'Numero N de best_trial_N.json do Transformer FD001')
python -m src.rul_prediction.v2.train final --datasets FD001 --model cnn_lstm --experiment baseline --best-trial $cnnTrial --seeds 1 2 3 4 5 --epochs 50 --patience 5
python -m src.rul_prediction.v2.train final --datasets FD001 --model transformer --experiment baseline --best-trial $transformerTrial --seeds 1 2 3 4 5 --epochs 50 --patience 5

# Apenas depois de fixar os hiperparâmetros e concluir todas as seeds
python -m src.rul_prediction.v2.evaluation terminal --datasets FD001 --model cnn_lstm --experiment baseline --seeds 1 2 3 4 5
python -m src.rul_prediction.v2.evaluation terminal --datasets FD001 --model transformer --experiment baseline --seeds 1 2 3 4 5
python -m src.rul_prediction.v2.evaluation windows --datasets FD001 --model cnn_lstm --experiment baseline --seeds 1 2 3 4 5
python -m src.rul_prediction.v2.evaluation windows --datasets FD001 --model transformer --experiment baseline --seeds 1 2 3 4 5
python -m src.rul_prediction.v2.evaluation compare --datasets FD001 --experiment baseline --seeds 1 2 3 4 5 --protocol terminal --comparison comparacao_v2
python -m src.rul_prediction.v2.evaluation compare --datasets FD001 --experiment baseline --seeds 1 2 3 4 5 --protocol windows --comparison comparacao_v2
```

Repita os comandos trocando FD001 pelo próximo FD e lendo os respectivos trials.
Para a tabela conjunta, após avaliar todos, use outro nome de comparação:

```powershell
python -m src.rul_prediction.v2.evaluation compare --datasets FD001 FD002 FD003 FD004 --experiment baseline --seeds 1 2 3 4 5 --protocol terminal --comparison todos_fd_v2
```

Para retomar Optuna, repita o comando original. Use `--fail-interrupted` somente
na situação documentada acima. O `final` aceita uma lista de FDs, mas um único
`--best-trial`: prefira um FD por comando para não confundir campeões distintos.

### Testes e limites da entrega

```powershell
python -B -m unittest tests.rul_prediction.test_v2_training -v
```

Verificação desta entrega: **35 testes aprovados**, sendo 16 de
`test_v2_preprocessing` e 19 de `test_v2_training`, sem falhas ou testes pulados.

Os testes usam dados sintéticos em diretórios temporários e mocks de `fit`,
`Study.optimize`, modelos salvos e `predict`. Exercitam contratos dos NPZ,
escopos sem teste/calibração, métricas, retomada, configuração, proteção da v1,
seeds, avaliação e comparação. Todas as CLIs `--help` são exercitadas com imports
de TensorFlow e Optuna bloqueados. Não são testes de convergência, desempenho,
serialização Keras real ou compatibilidade numérica GPU. A suíte histórica não
foi executada para evitar carregar modelos/dados reais fora deste escopo.

A reserva de calibração permanece intocada. Antes de associar os CSVs a anomaly
scores, alinhe os motores e protocolos do AE, que hoje possui partição própria.
As chaves preservadas não garantem, por si só, independência do treinamento do
AE. Nenhuma fusão, calibração automática ou experimento XAI foi acrescentado.
