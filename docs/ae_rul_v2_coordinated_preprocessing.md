# Etapa C — preparação do Autoencoder coordenado ao RUL v2

## O que foi implementado

Uma preparação independente em `src/anomaly_detection/v2/`, sem treinamento,
Optuna, inferência de redes, calibração de thresholds ou geração de anomaly scores.
O destino padrão é `data/processed_anomaly_v2/FD00X/`. Os pipelines antigos e os
dados/modelos/resultados RUL v2 não são modificados.

A [auditoria da Etapa B](ae_rul_v2_compatibility_audit.md) mostrou que os AEs
antigos utilizaram motores da validação e da reserva RUL em seu desenvolvimento.
Congelar esses pesos ou apenas recalibrar thresholds não desfaz essa exposição.
O novo ramo deve aprender sua representação com grupos coordenados, antes de
investigar se seus scores acrescentam informação à previsão de RUL.

Esta entrega **não processou FD001–FD004 reais**. Os comandos abaixo são para
execução manual após revisão. Foram executados somente testes sintéticos.

## 1. Grupos herdados e divisão interna

A preparação lê diretamente três JSONs da referência RUL:

- `data/processed_rul_v2/FD00X/manifest.json`;
- `config.json`;
- `splits.json`.

Confere pipeline/schema, dataset, configuração e SHA256 de configuração/splits.
Não lê os NPZ, os scalers RUL ou os metadados de teste. A referência completa é
identificada pelo hash do manifesto e pelo hash declarado do treino bruto.
Antes de preparar o AE, o SHA256 efetivo de `train_FD00X.txt` deve coincidir.

Os IDs de `splits.json` são obrigatórios. Não se sorteiam novamente os motores
entre train/validation/calibration. São verificadas listas não vazias, IDs
inteiros únicos e disjunção entre os três grupos de desenvolvimento. O teste
é outro namespace e pode repetir os mesmos números de IDs.

Somente os motores de **RUL train** são permutados para dois grupos internos:

| Grupo | Papel nesta preparação | Uso futuro |
| --- | --- | --- |
| `ae_fit` | Ajusta features/scalers/KMeans usando apenas sua região nominal | Ajuste dos pesos AE |
| `ae_inner_validation` | Somente transformação com objetos de AE fit | Seleção AE e EarlyStopping AE |
| RUL validation | Nenhuma transformação na preparação padrão | Validação do futuro preditor integrado, não desenvolvimento AE |
| RUL calibration | Somente IDs registrados; nenhum dado preparado por padrão | Possível calibração independente futura, se aprovada |
| Teste oficial | Somente IDs registrados; nenhum arquivo de teste aberto | Aplicação futura explícita, após congelar decisões |

Algoritmo interno: ordenar os IDs de RUL train, permutar com
`np.random.default_rng(seed)` e reservar os primeiros
`ceil(n_motores * inner_validation_fraction)` para a validação interna.
O restante é AE fit. As listas são salvas ordenadas. Seed padrão 42, fração
interna padrão 0,20. Exigem-se dois grupos não vazios, sem reposição de motores.

Com 70 motores RUL train, isso produz 56 AE fit e 14 inner-validation. Para
182 motores, são 145/37; para 174, 139/35. São contagens **atribuídas pela regra**,
não resultados de um processamento real executado nesta entrega. A elegibilidade
nominal pode reduzir a quantidade efetivamente representada, sem mudar os IDs
atribuídos e sem buscar substitutos na validação/reserva/teste.

O arquivo bruto de treino contém fisicamente todos os motores de desenvolvimento.
Ele é lido para conferir seu inventário, mas as linhas de RUL validation/calibration
são removidas antes da preparação nominal e não entram em fit ou transform.
Nenhum arquivo `test_FD00X.txt` ou `RUL_FD00X.txt` é necessário ao fluxo padrão.

## 2. Região nominal e motores não elegíveis

Para cada motor de RUL train, separadamente por papel interno:

`nominal_cycles = floor(n_ciclos_da_trajetória_completa * nominal_fraction)`.

A fração padrão é 0,30. A seleção não consulta os valores de RUL: considera
os primeiros ciclos físicos da trajetória completa de desenvolvimento. Uma
janela nominal tem exatamente 30 ciclos, passo 1, sem padding. Um motor só
contribui se tiver ao menos 30 ciclos nominais; sua quantidade de janelas é
`max(0, nominal_cycles - 29)`.

`nominal_units.csv` registra **todos** os motores atribuídos de RUL train,
papel interno, total de ciclos, ciclos nominais, elegibilidade e janelas. Motores
não elegíveis permanecem nesse inventário e nos splits; não aparecem nos NPZ
nominais. O resumo imprime seus IDs. Se um grupo inteiro não tiver elegíveis,
a execução falha antes da publicação dos artefatos, sem completar com holdouts.

`AERole` indica a atribuição interna. Para verificar exposição efetiva ao ajuste
em casos de inelegibilidade, consultar também o inventário e `train_units_` do
pré-processador, que contém somente os motores elegíveis AE fit.

**Limitação:** conhecer os primeiros 30% exige saber retrospectivamente a duração
total. Isso é uma hipótese nominal de construção offline, não uma regra online
para reconhecer saúde ou recalibrar o modelo de um motor novo. Não equivale a
um rótulo verdadeiro de ausência de falha. A inferência futura de uma janela
não poderá depender da duração final ou de ciclos posteriores ao seu final.

## 3. Normalizador próprio e seleção de features

`CoordinatedNominalPreprocessor`, definido em `v2/utils.py`, cria seus próprios
objetos. Reutiliza por herança apenas a implementação estável de normalização
de `RULPreprocessor`; **não reutiliza instâncias, features selecionadas ou
parâmetros ajustados do RUL**. O contexto, a população nominal, a configuração
e o arquivo joblib são do AE. Esse reuso também é registrado nos hashes do código.

O método fit exige exatamente os IDs elegíveis AE fit e os comprimentos nominais
declarados para cada motor. Não aceita inner-validation, motores externos ou
ciclos degradados posteriores ao corte. Um segundo fit é recusado.

- Candidatas: três settings e 21 sensores, na ordem canônica do C-MAPSS.
- Seleção nova e própria: amplitude `max - min > 1e-12` **no AE fit nominal**.
  Essa decisão evita as constantes que o critério histórico `std > 0` podia
  reter por resíduos numéricos. Não modifica a seleção dos experimentos antigos.
- FD001/FD003: um MinMaxScaler global, sem KMeans.
- FD002/FD004: KMeans de seis clusters nos settings brutos AE fit nominais,
  `n_init=10`, seed configurada; MinMaxScaler por cluster. Exige seis condições
  distintas e seis clusters ocupados, em vez de inventar regimes ausentes.
- Scalers com `clip=False`: valores fora de [0,1] nas demais fases são preservados.
- Inner-validation recebe exclusivamente transform com objetos congelados.

Features, ordem e colunas descartadas são persistidas em `features.json`. A ordem
também é conferida com os scalers e com cada NPZ. RUL, unit_nr, ciclos, duração
e indicadores retrospectivos nunca são entradas do tensor AE.

Os normalizadores AE/RUL podem diferir legitimamente, inclusive com features
de mesmo nome. Os labels dos clusters também não são intercambiáveis. A futura
inferência deverá partir da janela **bruta**, passando por cada ramo com seu
próprio transformador, nunca entregar `X` normalizado pelo RUL ao AE.

## 4. Contrato das janelas alinhadas — sem scoring nesta etapa

A função `prepare_aligned_windows(raw, rul_metadata, processor, ...)` recebe
sensores brutos, metadados de um único dataset/protocolo RUL e o processador AE
congelado. Não carrega redes e não calcula reconstruction errors.

A chave física é:

`(Dataset, SourceSplit, unit_nr, start_cycle, end_cycle)`.

O dataframe bruto deve conter `Dataset` e `SourceSplit` explícitos, além das
26 colunas brutas. Uma divergência de origem é recusada mesmo quando os números
de motores são iguais. A referência de splits também restringe os motores
permitidos por papel. Não se juntam train/test ou os protocolos terminal/windows
na mesma chamada.

A função verifica chaves únicas, ciclos, tamanho real, padding e shape. Preserva
a ordem das linhas dos metadados RUL, necessária para a associação com seus NPZ.
Retorna apenas:

- arrays com `X` float32, shape `(n_janelas, 30, n_features_ae)`, e `features`;
- metadados `Dataset, SourceSplit, Split, AERole, unit_nr, start_cycle,
  end_cycle, observed_cycles, padding_cycles, is_padded`.

Não copia RUL verdadeiro, flags retrospectivos ou scores de CSVs antigos para
as entradas. `Split` continua sendo o papel RUL; `AERole` é o papel interno AE
ou `not_used_for_ae_development` nos grupos externos.

Os sensores devem fornecer o prefixo observado desde o ciclo 1, incluindo os
ciclos solicitados. Linhas posteriores ao maior ciclo final solicitado para
cada motor são removidas **antes** da validação/transformação. Os scalers e a
classificação de regime são congelados e ponto a ponto; cada janela recebe
somente seu intervalo real, sem usar linhas posteriores ao seu final. Os testes
conferem invariância ao adicionar/alterar ciclos futuros além do prefixo pedido.

Teste oficial exige `allow_test=True`. Reserva exige `allow_calibration=True`.
Essas permissões pertencem ao uso futuro explícito da função, **não habilitam
nenhum processamento de teste/reserva na CLI padrão**. Nem sequer há um comando
de inferência AE v2 nesta entrega. O estágio posterior também deverá autenticar
os arquivos brutos/metadados por hashes antes de chamar esse contrato em memória.

## 5. Motores curtos e padding

Para uma janela RUL com menos de 30 ciclos reais, repetir o primeiro vetor
observado à esquerda até completar 30 posições, usando as features e escalas AE.
Por exemplo, 19 observações geram 11 repetições à esquerda, sem ciclos negativos:
`start_cycle=1`, `end_cycle=19`, `observed_cycles=19`, `padding_cycles=11` e
`is_padded=True`.

Isso preserva a população de motores RUL e não usa ciclos de outro motor nem
observações futuras. A repetição **não é uma nova observação física**, nem uma
reconstrução perfeita. Não se inventa score zero e não se alteram thresholds.
Seu efeito no reconstruction error ainda precisará ser avaliado no protocolo
científico futuro; a implementação da política não demonstra sua eficácia.

Padding é suporte do contrato de **scoring futuro**, não forma de fabricar
regiões nominais suficientes para ajustar o AE. Foram testadas especificamente
fixtures FD002 e FD004, com motores de 12 e 19 ciclos e seis regimes no ajuste.

## 6. Formato persistido, identidade e proteção

```text
data/processed_anomaly_v2/FD001/
  config.json
  splits_ae.json
  features.json
  preprocessor.joblib
  nominal_fit.npz
  nominal_inner_validation.npz
  nominal_fit_metadata.csv
  nominal_inner_validation_metadata.csv
  nominal_units.csv
  manifest.json
```

`config.json` registra dataset, seed, frações, tamanho 30, método de divisão,
seleção nominal/features, dtype e política de padding. `splits_ae.json` guarda
os dois grupos internos e uma cópia das listas RUL herdadas, sem modificá-las.
Os NPZ têm **somente `X` e `features`**, sem target RUL; futuramente o AE será
ajustado para reconstruir X, não para prever RUL.

O manifesto identifica `pipeline=ae_coordinated_rul_v2`, schema 1, configuração,
versões Python/NumPy/pandas/scikit-learn/joblib, origem, hashes do código e dos
nove artefatos. Vincula ainda o manifesto, configuração, splits e treino bruto
da referência RUL por SHA256. Registra motores atribuídos/elegíveis, IDs excluídos,
linhas nominais efetivas e janelas por grupo. A classe serializável está em
`src.anomaly_detection.v2.utils`, não em `__main__`.

O destino é criado exclusivamente após preparar os dados em memória; cada
arquivo é criado com modo exclusivo. O manifesto é escrito por último. Uma
falha pode deixar pasta parcial, que é preservada e **não** conta como execução
concluída. Não há retomada implícita ou `--overwrite`. Uma nova versão deve usar
uma pasta nova, por exemplo `--output-dir data/processed_anomaly_v2/revisao_02`.

Dentro do projeto, a escrita só é permitida sob `data/processed_anomaly_v2/`;
pastas históricas e origens são protegidas, inclusive por resolução dos caminhos.
Pastas temporárias externas são aceitas para testes. Uma CLI com vários FDs
confere todos os destinos antes de começar; não é uma transação global se houver
falha posterior em outro dataset.

## 7. Validação e recarregamento sem novo fit

`validate_saved_dataset` confere:

1. Manifesto de conclusão, configuração e hashes dos nove artefatos.
2. Referência RUL atual e igualdade dos splits herdados.
3. Reprodução da divisão interna, disjunção e contenção em RUL train.
4. Inventário completo, contagens nominais e elegibilidade, sem substituições.
5. Classe/configuração/IDs/limites do processador e número de amostras dos scalers.
6. Features ordenadas, scalers e KMeans de acordo com o dataset.
7. Metadados correspondentes às regiões nominais e NPZ finitos com shape correto.

Não lê dados brutos, NPZ RUL ou teste, nem realiza novo fit/transform. Os hashes
dos brutos são vinculados à referência histórica; a validação sem brutos não
afirma ter conferido o conteúdo de um arquivo bruto que tenha sido editado depois.
Na preparação, o arquivo bruto é efetivamente autenticado antes e depois do ajuste.

```python
from src.anomaly_detection.v2.preprocessing import load_preprocessor

processor = load_preprocessor('data/processed_anomaly_v2/FD001')
# Recarregado e validado; não chamar fit novamente.
```

Somente carregar joblib de origem local confiável: SHA256 verifica integridade,
não torna seguro um pickle de origem desconhecida.

## 8. Comandos manuais — não executados em dados reais nesta entrega

Na raiz do repositório e no ambiente Python do projeto:

```powershell
# Apenas ajuda, sem ler dados ou importar TensorFlow/Optuna:
python -m src.anomaly_detection.v2.preprocessing --help

# Primeiro FD001, após revisão:
python -m src.anomaly_detection.v2.preprocessing --datasets FD001 --seed 42 --nominal-fraction 0.30 --inner-validation-fraction 0.20

# Valida a configuração efetivamente salva, sem novo ajuste:
python -m src.anomaly_detection.v2.preprocessing --datasets FD001 --validate-only

# Opcional: exigir também estas opções ao validar:
python -m src.anomaly_detection.v2.preprocessing --datasets FD001 --validate-only --seed 42 --nominal-fraction 0.30 --inner-validation-fraction 0.20

# Depois de conferir FD001, preparar os demais:
python -m src.anomaly_detection.v2.preprocessing --datasets FD002 FD003 FD004 --seed 42 --nominal-fraction 0.30 --inner-validation-fraction 0.20
python -m src.anomaly_detection.v2.preprocessing --datasets FD001 FD002 FD003 FD004 --validate-only
```

`--validate-only` lê a configuração salva quando opções não são fornecidas;
opções explicitamente informadas devem coincidir, ou há erro. Os parâmetros de
origem/destino opcionais são `--raw-dir`, `--rul-dir` e `--output-dir`. Na validação,
os brutos não precisam estar disponíveis.

Não incluir um FD já preparado em uma nova chamada de preparação: a proteção
contra sobrescrita o recusará. Use `--validate-only` para conferir os existentes.
Esta CLI não treina modelos, não reutiliza o fluxo AE histórico e não consome
automaticamente a reserva.

## 9. Testes sintéticos e próximos passos

```powershell
python -B -m unittest tests.anomaly_detection.test_coordinated_preprocessing -v
```

A suíte cobre herança de splits, 56/14 a partir de 70 motores, disjunção,
normalização/features/KMeans apenas em AE fit nominal, não acesso ao teste,
serialização/recarregamento, seis regimes, padding em FD002/FD004, chaves e
namespaces, ordem, ausência de RUL nas entradas, causalidade por prefixos,
reprodutibilidade, configurações/hashes incompatíveis e preservação de pastas
históricas/parciais. Usa somente pequenos MinMaxScaler/KMeans sintéticos.

Resultado da verificação: **39 testes aprovados**, sem falhas ou pulados: 23
novos de AE coordenado e 16 de regressão sintética RUL v2. A execução combinada
bloqueou imports de TensorFlow, Keras e Optuna. As CLIs foram exercitadas somente
com ajuda e fixtures temporárias, incluindo recarregamento em processo separado.

Ainda permanecem pendentes, **fora desta implementação**: executar e conferir
a preparação real; aprovar o protocolo de treino/seleção AE interno e o uso de
score contínuo; treinar os AEs coordenados; implementar inferência autenticada
por chave e validar o efeito de padding; decidir escala do score e eventual
cross-fitting; planejar a ablação/fusão com RUL. Thresholds, se usados no futuro,
exigirão um protocolo próprio; não são necessários para preparar os tensores.

Próximo passo recomendado: o pesquisador preparar manualmente FD001 e executar
`--validate-only`, revisar grupos/contagens, e só depois ampliar para FD002–FD004.
O treino do novo AE exige uma tarefa e autorização posteriores. As baselines
RUL v2 e os AEs antigos continuam preservados.
