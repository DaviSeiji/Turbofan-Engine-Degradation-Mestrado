# RUL v2 — etapa 1: pré-processamento por motor

Esta etapa implementa somente preparação e persistência de dados. O processamento
dos datasets reais fica a cargo do pesquisador após a revisão. Nenhum treinamento,
Optuna, avaliação preditiva ou experimento de integração é implementado aqui.

## Implementação independente e preservação da v1

O código novo está em `src/rul_prediction/v2/`:

- `utils.py`: configuração, divisão de motores, validação de trajetórias e classe
  serializável `RULPreprocessor`.
- `preprocessing.py`: leitura, RUL, janelas, gravação, validação de artefatos e CLI.
- `__init__.py`: identificação do pacote, sem executar processamento.

A v1 continua em `src/common/preprocessing/preproc_data_nasa.py`, com os mesmos
scripts e artefatos. Os scripts atuais de CNN-LSTM/Transformer ainda leem
`data/processed_data/`: **não passam a utilizar v2 automaticamente**.
O treinamento v2 será uma etapa posterior, usando arrays explícitos de treino e
validação, em vez de `validation_split` sobre um array combinado.

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

Esta entrega termina na preparação dos dados. Treinamento, Optuna, avaliação
preditiva, explicações XAI e fusão de modelos ficam para etapas separadas.
