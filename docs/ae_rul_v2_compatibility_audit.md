# Auditoria de compatibilidade LSTM Autoencoder × RUL v2 — Etapa B

## 1. Resumo executivo e decisão

**Os quatro Autoencoders atuais podem ser preservados para exploração e para seus
experimentos originais de anomalia, mas não devem ser reutilizados como extratores
da comparação integrada principal com independência por motor exigida no RUL v2.**
Não é um problema de shape apenas: seus pesos e transformadores foram ajustados
com motores da validação/reserva RUL, e a seleção/threshold também consultaram
outros motores desses grupos.

Para a futura comparação principal, os quatro FDs exigem **novos AEs e novos
transformadores nominais coordenados com os splits RUL v2**. Não é necessário
repetir a CNN-LSTM/Transformer baseline, mudar seus NPZ ou executar novas buscas
automaticamente. A necessidade é de uma versão AE para essa nova pergunta
científica, não de apagar ou invalidar os experimentos de anomalia existentes.

| Dataset | AE atual reutilizável? | Principal impedimento | Novo treino necessário? | Justificativa |
| --- | --- | --- | --- | --- |
| FD001 | Exploração: sim. Principal independente: não | AE train inclui 12/15 motores RUL validation e 11/15 da reserva | Sim, AE + transformadores para integração principal | Os demais 3/4 motores desses grupos entraram na validação/seleção/threshold AE |
| FD002 | Exploração: sim. Principal independente: não | AE train inclui 32/39 motores RUL validation e 27/39 da reserva | Sim, AE + transformadores para integração principal | Os demais 7/12 entraram na validação/seleção/threshold AE; faltam scores de 6 motores curtos de teste |
| FD003 | Exploração: sim. Principal independente: não | AE train inclui 12/15 motores RUL validation e 11/15 da reserva | Sim, AE + transformadores para integração principal | Os demais 3/4 entraram na validação/seleção/threshold AE |
| FD004 | Exploração: sim. Principal independente: não | AE train inclui 30/38 motores RUL validation e 28/37 da reserva | Sim, AE + transformadores para integração principal | Os demais 8/9 entraram na validação/seleção/threshold AE; faltam scores de 11 motores curtos de teste |

A **sobreposição dos grupos é confirmada**. A integração ainda não foi realizada:
logo, não se afirma que seus resultados já sofreram vazamento. Afirma-se que
reutilizar diretamente esses AEs **não satisfaria** a independência solicitada.
Isso também não é evidência de vazamento para as baselines RUL v2 já treinadas,
que não usam scores AE, nem de ajuste de pesos do AE no teste oficial.

### Escopo efetivamente executado

Revisão base Git: `a646f23`, considerando também os artefatos locais existentes.
Foi feita leitura do código e auditoria direta dos JSONs/CSVs, sem criar módulo
auxiliar. Foram conferidos hashes de **100 arquivos de entrada**, inclusive os
artefatos AE declarados em seus manifestos e os metadados RUL usados nas junções.

Não se carregaram redes: os quatro `.keras` foram conferidos por hash e tiveram
somente seu `config.json` interno lido como ZIP. Os quatro preprocessadores
joblib foram inspecionados como objetos de scalers/KMeans, **sem fit, transform
ou predict**. As consultas aos quatro bancos SQLite usaram exclusivamente
`mode=ro`, sem inicializar storage ou executar Optuna. TensorFlow, Keras e
Optuna ficaram bloqueados no processo de análise. Não foram materializados
novos tensores, reconstruções ou anomaly scores.

Referências locais principais:
[protocolo RUL v2](rul_protocol_v2.md),
[auditoria histórica](rul_protocol_audit.md) e
[análise da Etapa A](rul_v2_results_analysis.md).
Os números de interseção abaixo são do **RUL v2**, não do corte v1 auditado antes.

## 2. Descrição do Autoencoder atual e sua proveniência

O AE usa janelas de 30 ciclos. O encoder LSTM resume a janela em uma camada
Dense latente linear; o decoder recebe esse vetor repetido e reconstrói a janela,
com saída linear. A loss é MSE. O erro global por janela é:

`reconstruction_error = mean((X_AE - X_reconstruído)^2, eixos tempo e features)`.

Inclui todas as features do ramo AE, **inclusive settings**, não só sensores.
Há também erro por feature, mas não foi escolhido como entrada da integração.
Um score elevado não é, isoladamente, um rótulo verdadeiro de falha.

A preparação atual divide os motores do arquivo original de treino em 80%/20%
com `train_test_split`, seed 42. Em cada motor, a região nominal corresponde a
`floor(0,30 × total_cycles)` ciclos iniciais. Motores cuja região tenha menos de
30 ciclos seriam excluídos; **nenhum motor foi excluído nos quatro artefatos
inspecionados**. Features, scalers e KMeans são ajustados exclusivamente nas
linhas nominais dos motores AE train, não em AE validation ou no teste.

As janelas nominais AE train ajustam pesos. AE validation participa de:
seleção Optuna, EarlyStopping dos trials, EarlyStopping do treino final e,
novamente, calibração P99 do threshold. Não há terceiro conjunto independente
de calibração no AE atual.

| Dataset | P99 salvo | Janelas de calibração | Trials COMPLETE / PRUNED |
| --- | --- | --- | --- |
| FD001 | 0,020519751 | 633 | 22 / 8 |
| FD002 | 0,041536535 | 1557 | 21 / 9 |
| FD003 | 0,014737011 | 821 | 19 / 11 |
| FD004 | 0,028568823 | 2119 | 20 / 10 |

Os quatro estudos possuem 30 trials contabilizados, sem FAIL/RUNNING nas consultas.
Os fingerprints e hashes dos NPZ nominais nos atributos `protocol` dos bancos
coincidem com os manifestos. O threshold vincula o modelo por SHA256 e o
pré-processamento por fingerprint. O P99 foi reproduzido a partir dos **scores
já salvos**, associados às janelas nominais de validação; não houve inferência.
A fonte salva de avaliação confirma `independent_calibration=false`.

Não existe um manifesto de treinamento tão completo quanto o da baseline RUL v2
por seed. A cadeia disponível é: manifesto nominal → hashes dos metadados/NPZ/
joblib → atributo protocol do estudo → threshold com hash do modelo → configuração
de detecção → hashes dos CSVs na avaliação de anomalias. Essa cadeia sustenta a
identificação usada; não é um registro individual de cada batch histórico.

Evidência de código:
[preparação nominal](../src/anomaly_detection/preprocessing.py#L138),
[seleção e treino](../src/anomaly_detection/training/tune_autoencoder.py#L30),
[calibração após o treino](../src/anomaly_detection/training/tune_autoencoder.py#L103),
[erro de reconstrução](../src/anomaly_detection/evaluation/detect_anomalies.py#L29)
e [registro da avaliação](../src/anomaly_detection/evaluation/evaluate_anomaly_detection.py#L638).

## 3. Splits RUL v2 e identificação dos papéis

O RUL v2 reserva motores antes do pré-processamento: 70% treino, 15% validação,
15% reserva, seed 42 e arredondamento pelo maior resto. A seed igual à do AE
**não produz os mesmos conjuntos**, pois o algoritmo e as frações são diferentes.
Foram utilizadas as listas efetivamente salvas, sem reconstruí-las pela seed.

| Dataset | AE train | AE validation = calibration | Janelas AE train | Janelas AE validation | RUL train | RUL validation | RUL calibration | Teste oficial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | 80 | 20 | 2.611 | 633 | 70 | 15 | 15 | 100 |
| FD002 | 208 | 52 | 6.913 | 1.557 | 182 | 39 | 39 | 259 |
| FD003 | 80 | 20 | 3.649 | 821 | 70 | 15 | 15 | 100 |
| FD004 | 199 | 50 | 8.933 | 2.119 | 174 | 38 | 37 | 248 |

Fontes por FD:
`data/processed_anomaly/FD00X/{nominal_units.csv,train_metadata.csv,validation_metadata.csv,manifest.json}`
e `data/processed_rul_v2/FD00X/{splits.json,*_metadata.csv,manifest.json}`.
Os IDs únicos dos metadados AE coincidiram com o inventário elegível de
`nominal_units.csv`; AE train e validation são disjuntos.

Todos os motores do arquivo original de treino aparecem em AE train ou AE
validation, enquanto o RUL os reparte em três grupos. Assim, **100% da validação
e da reserva RUL já foram expostos ao desenvolvimento do AE**: parte no ajuste
direto dos pesos/transformadores e parte na seleção/calibração.

### Namespaces: não confundir arquivo de origem com papel de desenvolvimento

Nos scores AE, `Split=train` significa origem em `train_FD00X.txt`, **não**
que aquele motor entrou no ajuste dos pesos AE. Esse CSV contém também motores
AE validation e todos os papéis RUL de desenvolvimento. Para o cruzamento:

- AE `Split` deve ser tratado como `SourceSplit`.
- O `Split` RUL deve permanecer separado: train, validation, calibration,
  test_windows ou test_terminal.
- A chave física deve incluir `Dataset` e `SourceSplit`, além de unit_nr/ciclos.
  IDs numericamente iguais em train/test ou entre FDs não são o mesmo motor.

No código de detecção atual, a persistência é calculada separadamente por
arquivo de origem. Reutilizar sua função com tabelas concatenadas de vários
datasets/splits, sem agrupá-los antes, não manteria essa garantia.

## 4. Matrizes de interseção entre motores

Cada célula apresenta **n (percentual do grupo AE da linha; percentual do grupo
RUL da coluna)**. Todos os conjuntos desta seção têm `SourceSplit=train`.
A linha AE_threshold_calibration repete AE_validation porque **é o mesmo grupo**;
não deve ser somada como um terceiro conjunto independente de motores.

### FD001

| Papel AE (motores) | RUL train | RUL validation | RUL calibration |
| --- | --- | --- | --- |
| AE_train (80) | 57 (71,25%; 81,43%) | 12 (15,00%; 80,00%) | 11 (13,75%; 73,33%) |
| AE_validation (20) | 13 (65,00%; 18,57%) | 3 (15,00%; 20,00%) | 4 (20,00%; 26,67%) |
| AE_threshold_calibration (20) | 13 (65,00%; 18,57%) | 3 (15,00%; 20,00%) | 4 (20,00%; 26,67%) |

### FD002

| Papel AE (motores) | RUL train | RUL validation | RUL calibration |
| --- | --- | --- | --- |
| AE_train (208) | 149 (71,63%; 81,87%) | 32 (15,38%; 82,05%) | 27 (12,98%; 69,23%) |
| AE_validation (52) | 33 (63,46%; 18,13%) | 7 (13,46%; 17,95%) | 12 (23,08%; 30,77%) |
| AE_threshold_calibration (52) | 33 (63,46%; 18,13%) | 7 (13,46%; 17,95%) | 12 (23,08%; 30,77%) |

### FD003

| Papel AE (motores) | RUL train | RUL validation | RUL calibration |
| --- | --- | --- | --- |
| AE_train (80) | 57 (71,25%; 81,43%) | 12 (15,00%; 80,00%) | 11 (13,75%; 73,33%) |
| AE_validation (20) | 13 (65,00%; 18,57%) | 3 (15,00%; 20,00%) | 4 (20,00%; 26,67%) |
| AE_threshold_calibration (20) | 13 (65,00%; 18,57%) | 3 (15,00%; 20,00%) | 4 (20,00%; 26,67%) |

### FD004

| Papel AE (motores) | RUL train | RUL validation | RUL calibration |
| --- | --- | --- | --- |
| AE_train (199) | 141 (70,85%; 81,03%) | 30 (15,08%; 78,95%) | 28 (14,07%; 75,68%) |
| AE_validation (50) | 33 (66,00%; 18,97%) | 8 (16,00%; 21,05%) | 9 (18,00%; 24,32%) |
| AE_threshold_calibration (50) | 33 (66,00%; 18,97%) | 8 (16,00%; 21,05%) | 9 (18,00%; 24,32%) |

### IDs que impedem a independência de validação e reserva

| Dataset | Papel AE → papel RUL | IDs (SourceSplit=train) |
| --- | --- | --- |
| FD001 | AE_train → validation | 7, 12, 35, 50, 55, 59, 68, 75, 83, 86, 89, 99 |
| FD001 | AE_train → calibration | 9, 14, 15, 20, 36, 37, 42, 48, 65, 66, 67 |
| FD001 | AE_validation → validation | 31, 45, 91 |
| FD001 | AE_validation → calibration | 13, 23, 54, 78 |
| FD002 | AE_train → validation | 2, 12, 18, 24, 41, 43, 44, 54, 58, 61, 62, 64, 70, 77, 106, 110, 117, 123, 130, 138, 158, 161, 165, 177, 184, 204, 209, 223, 225, 230, 242, 257 |
| FD002 | AE_train → calibration | 15, 17, 23, 32, 35, 36, 37, 42, 45, 48, 50, 59, 65, 103, 111, 137, 141, 142, 164, 176, 183, 195, 196, 203, 236, 240, 244 |
| FD002 | AE_validation → validation | 11, 16, 143, 168, 212, 214, 224 |
| FD002 | AE_validation → calibration | 7, 20, 31, 78, 98, 114, 140, 182, 191, 229, 238, 243 |
| FD003 | AE_train → validation | 7, 12, 35, 50, 55, 59, 68, 75, 83, 86, 89, 99 |
| FD003 | AE_train → calibration | 9, 14, 15, 20, 36, 37, 42, 48, 65, 66, 67 |
| FD003 | AE_validation → validation | 31, 45, 91 |
| FD003 | AE_validation → calibration | 13, 23, 54, 78 |
| FD004 | AE_train → validation | 2, 12, 18, 24, 32, 41, 43, 44, 54, 58, 62, 64, 77, 106, 110, 123, 130, 143, 158, 161, 165, 168, 177, 192, 199, 210, 211, 212, 217, 229 |
| FD004 | AE_train → calibration | 15, 23, 35, 36, 37, 42, 45, 48, 50, 59, 65, 78, 103, 111, 114, 137, 140, 142, 164, 182, 183, 203, 206, 227, 230, 231, 236, 238 |
| FD004 | AE_validation → validation | 16, 17, 61, 70, 138, 191, 201, 216 |
| FD004 | AE_validation → calibration | 7, 20, 31, 98, 141, 176, 194, 195, 196 |

AE validation é também o conjunto de seleção/EarlyStopping/calibração. Para
identificar todos os papéis, as listas completas AE validation são:

- **FD001, AE validation/calibration:** 1, 5, 11, 13, 19, 23, 31, 32, 34, 40, 45, 46, 54, 71, 74, 77, 78, 81, 84, 91.
- **FD002, AE validation/calibration:** 7, 10, 11, 16, 19, 20, 25, 26, 31, 34, 46, 47, 69, 76, 78, 91, 93, 97, 98, 102, 105, 114, 115, 120, 140, 143, 145, 151, 155, 159, 168, 174, 178, 180, 182, 186, 191, 197, 202, 205, 206, 207, 212, 213, 214, 221, 224, 229, 237, 238, 243, 259.
- **FD003, AE validation/calibration:** 1, 5, 11, 13, 19, 23, 31, 32, 34, 40, 45, 46, 54, 71, 74, 77, 78, 81, 84, 91.
- **FD004, AE validation/calibration:** 7, 10, 11, 16, 17, 19, 20, 25, 26, 31, 34, 39, 46, 56, 61, 68, 69, 70, 97, 98, 105, 109, 112, 113, 115, 119, 121, 138, 141, 159, 160, 163, 173, 174, 176, 181, 184, 185, 191, 194, 195, 196, 197, 200, 201, 202, 216, 224, 240, 247.

Em cada FD, AE train é o complemento dessa lista no inventário de
`nominal_units.csv`, pois todos os motores foram elegíveis. Os demais IDs e
suas divisões RUL estão explicitamente em `splits.json`, não foram inferidos por
intervalos ou por seed. FD001/FD003 têm listas numéricas iguais nesta execução,
mas cada FD constitui uma população distinta.

## 5. Compatibilidade dos pré-processamentos

| Aspecto | AE atual | RUL v2 | Consequência para integração |
| --- | --- | --- | --- |
| Janela / stride | 30 ciclos / 1 | 30 ciclos / 1 | Alinhamento físico possível em janelas completas |
| Origem para ajustar transformadores | Primeiros 30% de AE train | Trajetórias completas de RUL train | Escalas distintas são legítimas, mas os grupos AE atuais são incompatíveis com o novo holdout |
| Features constantes | std > 0 no nominal | amplitude > 1e-12 no treino | Listas/dimensões podem diferir; não trocar entradas entre ramos |
| Settings | Candidatos e usados no KMeans | Candidatos e usados no KMeans | Score AE também contém erro sobre ajustes operacionais |
| FD001/FD003 | MinMax global | MinMax global | Mesmo tipo de transformação não significa mesmos parâmetros |
| FD002/FD004 | KMeans(6) em settings brutos nominais, MinMax por cluster | KMeans(6) em settings brutos de RUL train, MinMax por cluster | Centros/labels/escalas não são intercambiáveis |
| Clipping | Scalers salvos com clip=False | clip=False | Valores fora de [0,1] são permitidos; não devem ser recortados automaticamente |
| RUL | Metadado, sem cap, nunca entrada do AE | Target de treino capped125; avaliação original | RUL verdadeiro não pode ser adicionado como feature de fusão |
| Motor com <30 ciclos | Não cria janela artificial | Repete o primeiro ciclo à esquerda, uma janela | Há lacunas de score em parte do teste |

| Dataset | Features AE | Features RUL v2 | Só no AE | Scalers AE |
| --- | --- | --- | --- | --- |
| FD001 | 20 | 17 | s_5, s_10, s_16 | 1 |
| FD002 | 24 | 24 | nenhuma | 6 |
| FD003 | 20 | 18 | s_5, s_16 | 1 |
| FD004 | 24 | 24 | nenhuma | 6 |

Em FD001/FD003, a ordem relativa das features comuns também coincide. Em
FD002/FD004, as 24 features são idênticas em nomes e ordem, mas **as escalas e
as populações de ajuste continuam diferentes**. Não basta conferir o shape.

Nos scalers globais AE salvos, `s_5,s_10,s_16` de FD001 e `s_5,s_16` de FD003
têm amplitude zero e ainda estão nas entradas. Isso é compatível com a fragilidade
numérica de selecionar por std >0, já discutida na auditoria histórica. É um
achado confirmado de seleção de features; não demonstra vazamento de rótulos.
Não foi corrigido. Removê-las mudaria o shape e exigiria uma versão nova.
Constância dentro de um cluster nos FDs multirregime não equivale a constância
global, e não deve motivar remoção automática.

Os joblib inspecionados têm nomes de features dos scalers iguais aos manifestos;
há um scaler em FD001/FD003 e seis em FD002/FD004. Nestes últimos, o KMeans salvo
tem seis clusters e usa os três settings na ordem prevista. Os arquivos de
treino brutos têm o mesmo SHA256 nos dois ramos. Os labels dos clusters são
arbitrários: não juntar AE regime 0 com RUL regime 0 supondo condição física igual.

**Rota correta futura:** mesma janela física bruta → transformação RUL congelada
→ preditor; em paralelo, mesma janela bruta → transformação nominal AE congelada
→ AE → score associado pela chave. Não alimentar o AE com `X` já normalizado
pelo RUL. Tampouco tentar resolver a diferença apenas reordenando colunas.

A possibilidade de valores fora de [0,1] foi confirmada nos parâmetros e no
código; não se executou transformação de trajetórias para quantificar novos
mínimos/máximos. Um reconstruction error maior pode refletir escala, regime ou
degradação; a auditoria não identifica uma causa física única.

## 6. Alinhamento de janelas e cobertura observada de scores

A junção foi feita por `Dataset, SourceSplit, unit_nr, start_cycle, end_cycle`,
após filtrar o CSV AE pela origem correta. `Split` RUL foi preservado como papel.
Os arquivos AE são `results/anomaly_detection/detection/FD00X/reconstruction_errors.csv`;
não houve geração de scores.

| Dataset | Split RUL | Scores presentes / janelas RUL | Cobertura | Ausentes = padding |
| --- | --- | --- | --- | --- |
| FD001 | train | 12.377 / 12.377 | 100,00% | 0 |
| FD001 | validation | 2.725 / 2.725 | 100,00% | 0 |
| FD001 | calibration | 2.629 / 2.629 | 100,00% | 0 |
| FD001 | test_windows | 10.196 / 10.196 | 100,00% | 0 |
| FD001 | test_terminal | 100 / 100 | 100,00% | 0 |
| FD002 | train | 32.245 / 32.245 | 100,00% | 0 |
| FD002 | validation | 7.035 / 7.035 | 100,00% | 0 |
| FD002 | calibration | 6.939 / 6.939 | 100,00% | 0 |
| FD002 | test_windows | 26.505 / 26.511 | 99,98% | 6 |
| FD002 | test_terminal | 253 / 259 | 97,68% | 6 |
| FD003 | train | 15.376 / 15.376 | 100,00% | 0 |
| FD003 | validation | 3.253 / 3.253 | 100,00% | 0 |
| FD003 | calibration | 3.191 / 3.191 | 100,00% | 0 |
| FD003 | test_windows | 13.696 / 13.696 | 100,00% | 0 |
| FD003 | test_terminal | 100 / 100 | 100,00% | 0 |
| FD004 | train | 38.665 / 38.665 | 100,00% | 0 |
| FD004 | validation | 7.701 / 7.701 | 100,00% | 0 |
| FD004 | calibration | 7.662 / 7.662 | 100,00% | 0 |
| FD004 | test_windows | 34.081 / 34.092 | 99,97% | 11 |
| FD004 | test_terminal | 237 / 248 | 95,56% | 11 |

Em todos os casos, a cobertura por `end_cycle` coincide exatamente com a cobertura
pela janela completa. **Não foi encontrado caso em que relaxar start_cycle
recuperasse um score**. Os pares existentes têm os mesmos RUL originais no fim da
janela. Não houve chaves ou pares motor/ciclo final duplicados em cada namespace
nos scores AE ou nos metadados RUL utilizados.

As lacunas são exclusivamente:

- **FD002:** motores de teste 28, 59, 83, 122, 201 e 223, um score ausente por motor.
- **FD004:** motores de teste 10, 19, 28, 125, 141, 156, 164, 204, 229, 239 e 246,
  um score ausente por motor.
- FD001/FD003 não têm essa lacuna. Nenhum motor de desenvolvimento tem padding
  ou falta de score nesta comparação.

Cobertura de janelas muito próxima de 100% não deve esconder a perda terminal:
faltam **2,32% dos motores FD002 e 4,44% dos motores FD004** se for exigido score
existente. Cobertura numérica também **não significa elegibilidade científica**:
os scores de validação/reserva RUL disponíveis foram produzidos por AEs expostos
a esses motores no desenvolvimento.

Alternativas para motores curtos, a decidir antes do experimento:

| Alternativa | Vantagem | Limitação / consequência |
| --- | --- | --- |
| AE novo com a mesma janela preenchida do RUL | Preserva a coorte e permite score escalar | Reconstrução sobre ciclos repetidos muda sua distribuição; precisa validação em desenvolvimento, sem trocar silenciosamente o AE atual |
| Máscara de disponibilidade + imputação fixa definida em desenvolvimento | Mantém todos os motores sem dizer que houve reconstrução perfeita | Efeito da ausência/imputação deve ser separado do efeito do score; incluir controle equivalente sem informação AE |
| Usar somente o ramo RUL nesses casos | Preserva as previsões e coorte originais, sem score inventado | É uma política híbrida de fallback explícita, não o mesmo forward integrado para todos; deve ser avaliada como tal |
| Excluir motores sem score | Simplifica a análise pareada em subconjunto | Não substitui a comparação principal na coorte completa; baselines devem ser relatadas também no mesmo subconjunto |
| Esperar 30 ciclos para emitir score | Causal e simples em operação | Não resolve a avaliação terminal dos motores truncados antes disso |

Não se escolheu uma política definitiva. A falta não pode ser escondida em
`fillna(0)`: score zero sem indicador significaria reconstrução perfeita. Também
não existe score naqueles ciclos finais para recuperar por uma junção mais permissiva.

## 7. Reutilização dos pesos, scalers e thresholds existentes

**Exploração:** os quatro AEs, seus scalers e scores são reutilizáveis para
descrever padrões nos artefatos existentes, com os papéis de desenvolvimento
explícitos. Isso não exige nova inferência nas janelas já cobertas.

**Comparação principal solicitada:** os quatro precisam de nova versão ajustada
apenas com desenvolvimento interno de RUL train. Não basta congelar os pesos,
recalibrar P99, excluir colunas dos CSVs, ou passar a gerar scores somente na
validação: a informação dos motores já influenciou pesos/transformadores e/ou
seleção. Não há um subconjunto RUL validation completamente intocado pelo AE
atual, pois a união AE train/validation cobre todos os motores de desenvolvimento.

Fine-tuning de pesos contaminados também não desfaz automaticamente essa exposição.
Para a alegação estrita pretendida, iniciar os AEs novos de forma independente
e refazer seus transformadores com o novo escopo. A arquitetura, funções,
espaços de busca, lógica de scoring e testes conceituais podem ser reaproveitados.

A arquitetura/hiperparâmetros históricos podem servir como referência exploratória
e reduzir trabalho de implementação. Contudo, seus valores campeões foram
selecionados numa validação que contém motores RUL validation/reserva. Para uma
seleção AE estritamente independente desses grupos, justificar uma configuração
predefinida ou selecionar/revalidar configurações **somente dentro de RUL train**;
não usar o resultado histórico como certificado de independência.

O teste oficial não entrou no ajuste/seleção/calibração do código AE inspecionado.
Um protocolo alternativo que declarasse pré-treinamento não supervisionado em
todo o desenvolvimento poderia ser estudado, mas seria outro desenho, não a
independência por motor requerida. Não foi adotado.

Os thresholds atuais só são apropriados ao experimento AE original/exploratório:
usam a mesma validação que escolheu o modelo. **A reserva RUL v2 não é uma
calibração independente para esses pesos atuais**, porque já participou do
desenvolvimento AE. Para um AE novo ajustado/selecionado sem ela, essa reserva
pode cumprir um papel útil, desde que o protocolo de calibração seja fixado antes.

## 8. Vazamentos: achados confirmados, riscos e disponibilidade temporal

### Classificação dos achados

| ID | Classe | Achado | Interpretação |
| --- | --- | --- | --- |
| B1 | Incompatibilidade confirmada | AE weights/scalers e seleção expostos à validação/reserva RUL | Reutilização principal violaria o escopo novo; não é vazamento ocorrido em uma integração já executada |
| B2 | Limitação confirmada | Threshold calibrado na validação usada para Optuna/EarlyStopping | Não é calibração independente, apesar de não haver ajuste de pesos nos batches de validação |
| B3 | Lacuna confirmada | 6/11 motores curtos sem score em FD002/FD004 | Exige política explícita, não junção ou imputação silenciosa |
| B4 | Conteúdo retrospectivo confirmado; vazamento futuro condicional | CSVs propagam primeiro ciclo de alerta para linhas anteriores e têm duração completa de episódios | Não são features online causais; uso nos relatórios é legítimo |
| B5 | Diferença confirmada | Transformadores, população nominal e parte das features diferem | Não compartilhar arrays normalizados entre ramos |
| B6 | Risco potencial | Distribuição de scores de motores AE fit difere da de não vistos | Pode afetar generalização da fusão; não prova sobreajuste ou exige cross-fitting automaticamente |
| B7 | Limitação operacional confirmada | Regra nominal de 30% depende da duração final no treinamento offline | Não pode ser aplicada como seleção nominal online sem informação futura |
| B8 | Melhoria opcional | OOF/cross-fitting e controles adicionais | Escolher conforme hipótese/diagnóstico no desenvolvimento, não pelo teste |

### O que é causal no ciclo final t

A janela AE contém apenas `[t−29,t]`. Scalers/KMeans já congelados são aplicados
por observação; não se recalculam estatísticas usando o restante da trajetória.
Encoder/decoder podem reconstruir ciclos anteriores com o contexto de toda a
janela, mas o **score da janela só fica disponível em t**. Usá-lo em t é causal;
atribuir esse mesmo score ao seu ciclo inicial anteciparia informação.

`above_threshold(t)` usa score(t) e threshold fixo. A regra de três janelas
exige finais consecutivos e emite o primeiro alerta na terceira janela disponível
(no caso sem padding, o primeiro final possível é t=32). Lacunas reiniciam a
contagem. `persistent_anomaly(t)` é causal quando calculado por motor/origem e
somente com a contagem passada; não é automaticamente um latch eterno de alerta.

### Colunas que não podem virar features diretamente

Em [persistent_alerts](../src/anomaly_detection/evaluation/detect_anomalies.py#L62),
o primeiro ciclo de alerta é calculado após percorrer a trajetória e gravado
em **todas** as linhas daquele motor. Exemplo observado: FD001, SourceSplit=train,
motor 1, janela terminando no ciclo 30 já contém
`first_persistent_anomaly_cycle=186`. Não se conhecia esse evento no ciclo 30.
Foram encontradas linhas com esse ciclo no futuro em FD001/002/003/004:
16.351 / 45.553 / 18.916 / 52.226, respectivamente.

Em [annotate_alerts](../src/anomaly_detection/evaluation/evaluate_anomaly_detection.py#L79),
`first_persistent_anomaly_cycle`, `above_episode_length`, `transient_window`,
`sustained_episode_window` e `contiguous_windows` usam mínimo/tamanho do
episódio ou segmento completo. São diagnósticos retrospectivos, não features
disponíveis no momento inicial do episódio. Não há defeito em usá-los no relatório;
haveria vazamento se fossem anexados indiscriminadamente à regressão online.

Também devem ficar fora das features: `RUL`, `RUL_original`, `RUL_capped`,
duração total do motor, ciclo de falha derivado e fração da vida completa.
Uma futura interface deve aceitar uma **lista positiva de campos causais**,
inicialmente só o reconstruction error da janela e eventual indicador de
disponibilidade, não copiar todo o CSV AE.

### Os primeiros 30%: treinamento retrospectivo versus operação

A regra usa a duração completa dos motores **de treinamento** para construir
um conjunto nominal hipotético offline. Isso é uma definição retrospectiva de
amostragem, não vazamento temporal automático de uma feature no teste; os labels
RUL não entram no cálculo de reconstrução. O detector atual não exige saber os
30% da vida do motor de teste: aplica o AE congelado a todas as janelas disponíveis.

O limite é operacional e científico: não se conhece antecipadamente a duração
final de um motor novo, e os primeiros 30% não são certificados de saúde real.
Portanto, não prometer essa regra como mecanismo online para ajustar o AE ou
recalibrar scalers/thresholds. A nominalidade assumida deve ser validada em
desenvolvimento e declarada como hipótese.

## 9. Qual score integrar e como tratar in-sample versus out-of-sample

### Informação inicial recomendada

| Alternativa | Vantagens | Limitações / cuidados |
| --- | --- | --- |
| A — Reconstruction error contínuo | Preserva magnitude; não depende de threshold; simples de auditar e ablar | Escala depende de features/normalização/regimes; pode ter cauda longa; precisa congelar todos os transformadores |
| B — Score normalizado | Facilita condicionamento numérico; permite transformação como log1p e escala pré-fixada | Transformação/escala são ajustes adicionais; aprender só no grupo de desenvolvimento apropriado e registrar parâmetros; não usar estatísticas do teste |
| C — Indicador binário + persistência | Interpretação operacional direta e causal se usa apenas passado | Perde magnitude; depende de calibração independente, hipóteses nominais e política temporal; não usar o threshold atual na comparação principal |

Recomenda-se começar por **A**, eventualmente com uma escala B escolhida e
ajustada só em desenvolvimento interno de RUL train. Isso não requer threshold,
percentil ou persistência para produzir a primeira feature. Normalização por
regime é uma hipótese adicional, não obrigatória; labels AE/RUL de clusters não
são equivalentes. Não escolher A/B/C pelo desempenho do teste já observado.

### Scores em motores que ajustaram o AE

É permitido que um extrator não supervisionado seja aprendido em RUL train e
usado como parte do preditor ajustado nesse mesmo grupo, sem rótulos de
validação/teste. Isso não constitui automaticamente vazamento de alvo.
Entretanto, scores sobre regiões memorizadas podem ter distribuição diferente
da de novos motores. O fato de o score ser não supervisionado não elimina essa
questão, nem a necessidade de manter RUL validation independente.

Os scores nominais já salvos mostram:

| Dataset | MSE nominal médio: AE train / validation | Acima de P99: AE train / validation |
| --- | --- | --- |
| FD001 | 0,012728 / 0,013663 | 0,38% / 1,11% |
| FD002 | 0,034794 / 0,034677 | 0,61% / 1,03% |
| FD003 | 0,009563 / 0,009102 | 5,97% / 1,10% |
| FD004 | 0,017798 / 0,019033 | 0,02% / 1,04% |

Os cerca de 1% na validação são esperados pelo percentil que nela foi definido;
não demonstram taxa independente de falsos alarmes, muito menos 1% de motores.
FD003 é um contraponto importante: os scores nominais de AE train não são
uniformemente menores; a fração acima do threshold é maior no treino. Diferenças
podem refletir composição dos motores, nominalidade e ajuste. Não foi isolado
sobreajuste como causa, e não se deve escolher cross-fitting com base nesses
contrastes ou nos resultados de teste da futura fusão.

### Estratégias possíveis, sem implementação

1. **AE único, treinado/selecionado dentro de RUL train, congelado.** Gerar score
   de todas as janelas de RUL train, mantendo indicador de papel AE fit/inner-val;
   validar a fusão somente em RUL validation. É o menor custo e uma baseline de
   representação não supervisionada válida, desde que se assuma e documente o
   contraste in-sample/out-of-sample e se façam diagnósticos em desenvolvimento.
2. **Cross-fitting por motor dentro de RUL train.** Para cada fold, ajustar todos
   os componentes AE sem os motores retidos e gerar scores só para esses motores.
   Reunir os scores OOF por chave. Útil quando a hipótese exige features realmente
   fora do ajuste, quando a seleção/regularização do AE tende a memorizar ou
   quando se quer controlar explicitamente esse contraste. Não é obrigatório
   para todo encoder não supervisionado.
3. **Subconjunto dedicado para aprender representação.** Separar grupos dentro
   de RUL train para AE e para ajuste do preditor. Evita scores in-sample no
   segundo grupo, mas reduz seus dados; uma comparação justa exigiria controle
   sem AE usando o mesmo subconjunto e possivelmente novos controles RUL.
   Não deslocar a reserva de 15% para ajustar pesos sem redefinir seu papel.

Cross-fitting custa K preparações/ajustes AE por FD, mais o desenho de inferência
final. Scalers/KMeans, seleção/ES e qualquer escala de score também devem respeitar
cada fold. Modelos de folds diferentes podem produzir scores com escalas distintas.
É preciso fixar se validação/teste usarão um AE final ou um ensemble e como a
distribuição desses scores será alinhada à dos scores OOF; cross-fitting não
resolve essa mudança de distribuição por mágica.

Com K folds nos quatro FDs, são 4K ajustes AE, além de eventuais 4 AEs finais e
suas inferências; novas buscas aumentariam ainda mais o custo. Os pesos atuais
não podem representar um fold que os exclua, pois já viram esses motores; podem
servir somente de referência exploratória. **Nenhum OOF foi implementado.**

## 10. Proposta de protocolo coordenado para a próxima etapa

Esta é uma proposta para revisão, não uma autorização para começar a executá-la.

1. **Congelar RUL v2.** Manter splits, manifestos, features, scalers, janelas,
   baselines e previsões atuais. O teste oficial fica fora das escolhas de
   arquitetura, score, nominalidade, missingness, threshold e seeds.
2. **Criar preparação AE coordenada, independente da antiga.** Receber
   `splits.json` como entrada obrigatória; declarar todos os IDs usados. Dividir
   apenas RUL train em AE fit e validação interna AE, por motor. Nenhum motor
   RUL validation/calibration/test entra em ajuste de pesos, seleção AE,
   EarlyStopping AE, normalização, clusters ou escolha de features.
3. **Manter a hipótese nominal explicitamente versionada.** Os 30% iniciais
   podem ser ponto de partida offline, sem tratá-los como verdade-terreno.
   Ajustar transformadores somente nas linhas nominais AE fit. Gerar tabelas
   internas AE de janelas/IDs/hash; não modificar os NPZ RUL v2.
4. **Definir a seleção AE antes da execução.** Uma configuração predefinida ou
   seleção controlada somente na validação interna do RUL train. Não é necessário
   iniciar 30 novos trials por padrão. Não usar o threshold, a reserva ou a
   validação RUL para selecionar o AE.
5. **Escolher o regime de scoring do treino.** Começar com o AE único congelado
   e papéis registrados, ou aprovar explicitamente OOF antes do experimento.
   No treino da fusão, usar todas as janelas do grupo previsto no desenho.
6. **Congelar a transformação do score.** Se houver log/escala, ajustar apenas
   em dados apropriados de RUL train e aplicar sem reajuste aos demais grupos.
   Não normalizar usando a concatenação train/validation/test.
7. **Gerar e armazenar scores por chave física.** RUL train usa a política do
   item 5; RUL validation usa só transformações/modelos congelados; teste idem,
   apenas na etapa final planejada. Validar cobertura/ausências/duplicatas e
   aceitar somente campos causais. A máscara para motores curtos requer decisão.
8. **Reservar a calibração independente se necessária.** Para score contínuo,
   não usar automaticamente os 15%. Se forem usados flags/thresholds, só após
   fixar modelo/procedimento, usar a região nominal definida dos motores RUL
   calibration, nunca empregados no desenvolvimento AE/RUL integrado. Fixar
   percentil e persistência sem otimização no teste. RUL calibration não é AE
   validation interna nem conjunto extra para escolher hiperparâmetros.
9. **Ajustar o preditor integrado só em RUL train.** RUL validation pode avaliar
   hiperparâmetros e EarlyStopping do preditor integrado como no protocolo RUL,
   mas não recebe gradientes nem reajuste dos transformadores. Distinguir esse
   papel de seleção permitido do ajuste/seleção **AE**, restrito ao treino.
10. **Comparar uma vez o plano fixado.** Mesmos motores/seeds/targets/métricas,
    com resultado terminal principal e janelas/faixas complementares. Documentar
    que este teste já é um benchmark conhecido, não uma primeira avaliação cega.

Com o AE novo, a reserva 15% passa a ser útil para uma calibração independente
**do sistema adaptativo**, desde que não seja consumida antes na seleção. Com o
AE atual, ela não tem essa independência. Sua utilidade para score contínuo não
implica necessidade de threshold ou obrigação de consumi-la agora.

## 11. Alternativas de integração futuras

| Estratégia | Alteração / custo relativo | Interpretabilidade | Cuidados para comparar |
| --- | --- | --- | --- |
| 1 — Fusão tardia escalar | Concatenar score ao vetor antes da regressão; simples, requer treinar a cabeça/modelo integrado | Ablar presença/ausência de um escalar é direto | Política de normalização, disponibilidade e mesma cabeça de controle |
| 2 — Sequência de scores passados | Entrada temporal adicional e gestão de histórico; maior complexidade | Permite estudar evolução do erro, mas adiciona escolhas de lookback | Score da janela em t só está disponível em t; não usar janelas futuras para preencher histórico |
| 3 — Latente do encoder | Extrair vetor de 8/32/8/64 dimensões nos AEs atuais, ou outra configuração futura; novas conexões/escala | Mais rica, porém menos simples de atribuir a erro de reconstrução | Maior dimensionalidade/capacidade, escala e independência dos encoders; não garante ganho |

Recomendação inicial: **estratégia 1 com score contínuo e AE congelado**, sem
threshold como condição de entrada. A posição exata da concatenação e a opção de
congelar/ajustar backbone RUL devem ser fixadas antes do experimento.

Na estratégia 2, uma sequência de 30 scores de janela pode exigir até 59 ciclos
observados para ter todos os scores sem preenchimento (primeiro score natural
em 30, trigésimo em 59). Não é a mesma disponibilidade de um score escalar em
t=30. Isso amplia a questão de padding/missingness; não anexar a reconstrução de
uma janela posterior aos seus ciclos anteriores.

Usar pesos RUL atuais como inicialização é possível tecnicamente, mantendo suas
features. Mas treinar/fine-tunar apenas a versão +AE por épocas extras criaria
um orçamento adicional não controlado. Se essa rota for adotada, incluir controle
sem informação AE com o mesmo procedimento de adaptação. Uma alternativa
inicial mais direta é treinar os modelos integrados do zero sob o mesmo
orçamento/hiperparâmetros de referência, preservando todos os originais.

## 12. Plano de comparação científica e ablação

Comparar, separadamente em cada FD:

- CNN-LSTM v2 original versus CNN-LSTM v2 + score.
- Transformer v2 original versus Transformer v2 + score.

As 40 baselines finais salvas permanecem congeladas. Usar seeds 1–5, mesmas
janelas/coortes, mesma perda de treino capped125, validação por motor, alvo de
avaliação original sem cap e RMSE/MAE/MedianAE/P95AE/MaxAE. Reportar todas as
seeds, viés/faixas de RUL e dispersão; não selecionar a menor métrica de teste.

### Comparação inicial de menor custo

Hiperparâmetros das baselines **RUL v2** podem ser congelados como referência
para um estudo inicial, mantendo batch/LR/épocas/patience e o mesmo protocolo.
Eles foram selecionados legitimamente em RUL validation; isto é distinto dos
hiperparâmetros históricos do AE, selecionados com outro particionamento.
A interpretação será “adicionar score sob este procedimento fixado”, não
“melhor modelo integrado possível”.

Ablação recomendada:

- A0: baseline RUL original, já disponível.
- A1: mesma arquitetura da fusão e treinamento, mas score constante/sem
  informação; a máscara de disponibilidade, se usada, deve ser controlada.
- A2: mesma arquitetura/procedimento de A1 com score AE real.

A1 ajuda a distinguir informação do score de capacidade, inicialização extra,
missingness ou alteração da cabeça. Não é preciso refazer A0 para preservá-la;
A1 é um controle novo, se o estudo pretender isolar o efeito informacional.
Se o caminho sem score reproduzir exatamente A0 em arquitetura e procedimento,
documentar essa equivalência antes de decidir se A1 adicional é necessário.

### Comparação principal e eventual otimização

Um desenho predefinido com hiperparâmetros fixos pode ser uma comparação
científica válida para essa hipótese restrita. Para uma alegação de desempenho
de arquiteturas **otimizadas**, a fusão pode exigir busca própria e orçamento
controlado comparável ao de suas baselines: mesmo critério, divisão, limite de
trials/épocas e política de seeds. A dimensão extra e a escala do score podem
mudar a configuração adequada; os hiperparâmetros originais não garantem
otimalidade após a concatenação.

Não usar o teste para decidir entre essas rotas. Ajustar/selecionar apenas em
desenvolvimento, fixar a família de comparações e congelar decisões antes de
abrir os novos resultados finais. Não tirar uma média de RMSE entre os FDs como
se fossem uma população homogênea. Para incerteza futura, diferenças pareadas
por motor e reamostragem de motores são mais adequadas que tratar janelas
sobrepostas como independentes. Mesma seed entre arquiteturas não significa
mesma realização aleatória.

A Etapa A mostrou grande erro contra RUL original >125. Não mudar o target para
favorecer a integração. Score AE não recupera automaticamente distinções de
alvos colapsados pelo cap nem demonstra que os erros nessa faixa são anomalias.

## 13. Artefatos, custos e rastreabilidade da próxima implementação

### Classificação por dataset

| Dataset | Existentes reutilizáveis | Existentes incompatíveis com a integração principal | Nova inferência futura | Novo pré-processamento AE | Novos treinos |
| --- | --- | --- | --- | --- | --- |
| FD001 | Raw, splits/metadados RUL, 10 baselines RUL e suas previsões; código AE | Pesos/scalers/threshold/scores AE atuais, apenas para uso independente principal | Scores do AE coordenado nas janelas aprovadas | Sim, dentro de RUL train; não refazer RUL v2 | 1 AE final + 10 modelos RUL integrados, antes de controles/buscas |
| FD002 | Mesmos tipos de artefatos | Mesma exposição, mais 6 motores teste sem score atual | Idem, com política para curtos | Sim | 1 AE final + 10 integrados |
| FD003 | Mesmos tipos de artefatos | Mesma exposição à validação/reserva | Scores coordenados | Sim | 1 AE final + 10 integrados |
| FD004 | Mesmos tipos de artefatos | Mesma exposição, mais 11 motores teste sem score atual | Idem, com política para curtos | Sim | 1 AE final + 10 integrados |

“1 AE final” pressupõe desenho simples e configuração AE fixa; uma busca ou
OOF acrescenta modelos. No total mínimo dessa rota: **4 novos AEs finais e
40 treinos finais RUL integrados** (2 arquiteturas × 5 seeds × 4 FDs). Um controle
A1 nas mesmas combinações acrescentaria até 40 treinos. Uma nova busca de 30
trials nas 8 combinações integradas significaria até 240 tentativas adicionais;
não se presume que todas completem 50 épocas. Não se estimam horas, GPU/RAM ou
consumo de hardware sem medição.

Não há ganho em gerar novos scores com os AEs antigos para cobrir as partições
principais antes de resolver o escopo: cobertura de todos os grupos já é alta
e não elimina exposição. O que muda exige nova preparação **do ramo AE** e
inferência do AE coordenado, não reprocessamento RUL nem retreino de A0.

### Arquivos posteriores sugeridos — nenhum criado aqui

- `src/anomaly_detection/v2/preprocessing.py`: aceitar splits RUL, separar
  desenvolvimento interno, preservar política de janelas e grupos; destino
  independente como `data/processed_anomaly_rul_v2/FD00X/`.
- `src/anomaly_detection/v2/train.py`: treino/seleção nominal coordenada e
  manifesto completo; `models/autoencoder_rul_v2/`, sem tocar nos antigos.
- `src/anomaly_detection/v2/score.py`: inferência futura por chave, coverage,
  proveniência, sem campos retrospectivos como features; armazenamento separado
  em `results/anomaly_detection/rul_v2_scores/`.
- Configuração versionada do plano de integração (grupos herdados, política
  de curtos, A/B/C, OOF ou AE único, orçamento, seeds, alvo e critérios).
- Builders/treino/avaliação **novos** para fusão tardia e ablação, por exemplo
  `src/rul_prediction/integration/`, com modelos/resultados próprios e sem
  modificar baselines congeladas.
- Testes sintéticos de disjunção, namespaces, causalidade por prefixos, join
  um-para-um, ausências/padding, identidade e não sobrescrita; teste de scores
  invariantes ao acrescentar ciclos posteriores quando o score se refere a t.

### Identidades observadas

Os valores abaixo estão abreviados somente para leitura. Os valores completos
constam nos manifestos/thresholds e foram comparados integralmente:

| Dataset | Fingerprint AE (prefixo 16 hex) | SHA256 do modelo AE (prefixo 16 hex) |
| --- | --- | --- |
| FD001 | `b50836b870daabf8` | `5af3d46043e92d66` |
| FD002 | `7f7fd961f70faa8a` | `c5d2d473f0f63d44` |
| FD003 | `0b83a94d770f49c8` | `f426f743462fc2ba` |
| FD004 | `54927ba18648684d` | `7e1859122520e1ab` |

Os CSVs de detecção são ligados aos JSONs de avaliação pelos SHA256 de
`input_sha256` de cada FD. A configuração global antiga da avaliação contém
caminhos anteriores à refatoração; foram usados os **caminhos atuais** e os
hashes por FD para evitar confundir renomeação com alteração dos dados.
O script de detecção antigo não registra hashes de todos os arquivos de teste
brutos nem versões de código como o RUL v2: a igualdade de janelas/RUL e a cadeia
de hashes disponível foram verificadas, mas não equivalem a uma reconstrução
independente de cada entrada normalizada histórica. A próxima versão deve
registrar manifesto completo por artefato, feature order e transformadores.

## 14. Próximos passos, em ordem de prioridade

1. **Aprovar o contrato de desenvolvimento AE coordenado.** Preservar os splits
   RUL; definir grupos internos apenas dentro de RUL train, escopo dos scalers,
   seleção/ES e papel não automático da reserva. Sem retreinar nada ainda.
2. **Decidir previamente política para motores curtos e informação causal.**
   Escalar contínuo é o ponto de partida; campos retrospectivos ficam proibidos.
   Comparação terminal precisa manter a mesma coorte ou declarar complemento.
3. **Implementar somente a preparação AE coordenada e seus testes sintéticos,**
   em novos caminhos, começando por FD001. Acrescentar fixtures de FD002 para
   seis regimes e motor curto. Não reaproveitar o fluxo 80%/20% sobre todo o
   desenvolvimento original.
4. Após revisão, autorizar explicitamente treino/scoring AE e depois a fusão,
   iniciando pela rota simples de AE único. Decidir OOF apenas se a hipótese
   exigir ou o diagnóstico em desenvolvimento justificar seu custo.
5. Fixar a ablação e orçamentos. Só então executar futuros modelos integrados,
   comparar todas as seeds e preservar as baselines da Etapa A.

**Próxima tarefa sugerida ao Codex:** “Implementar a preparação AE coordenada ao
RUL v2, sem executar treinamento/inferência, herdando splits.json, com seleção de
motores AE fit/inner-validation somente em RUL train, novos manifestos e testes
sintéticos de disjunção, namespaces e tratamento explícito de motores curtos.”

### Respostas objetivas

1. **Aproveitar os quatro AEs atuais?** Sim para exploração/experimentos AE
   originais; não como extratores independentes da comparação integrada principal.
2. **Quais motores e papéis?** Matrizes e IDs da seção 4; todos os motores
   RUL validation/calibration já participaram de AE fit ou AE validation.
3. **Vazamento confirmado ou risco?** Sobreposição e não independência dos
   thresholds são confirmadas; vazamento em uma futura integração depende de
   reutilizar esses componentes. Não há integração executada para acusar.
4. **Retreinar algum AE?** Quatro, se mantida a independência principal pedida,
   junto de seus transformadores. Não nesta etapa; não retreinar RUL original.
5. **Reserva de 15% útil?** Sim para futura calibração de um AE novo sem exposição
   a ela; não é independente para os AEs atuais. Score contínuo não a exige.
6. **Scores sem vazamento?** AE ajustado/selecionado só em RUL train, escala
   apropriada congelada, validação/teste fora de todos esses ajustes, score
   causal por janela; OOF é uma opção planejada, não um reparo dos pesos antigos.
7. **Estratégia inicial?** Fusão tardia de score contínuo, sem threshold como
   requisito, com controle sem informação e política de disponibilidade explícita.
8. **Arquivos posteriores?** Preparação/treino/scoring AE coordenados, builders
   e rotinas de ablação/fusão separados, manifestos/configuração e testes.
9. **Próxima tarefa?** Preparação AE coordenada e contratos/testes apenas,
   começando FD001, antes de autorizar qualquer experimento novo.

**Encerramento do escopo:** a única alteração desta etapa é este relatório.
Nenhuma rede foi treinada ou executada, nenhum score novo foi inferido, nenhum
estudo Optuna foi executado, nenhum dataset foi reprocessado e nenhum artefato
histórico foi sobrescrito. Não foram alterados src/anomaly_detection/,
src/rul_prediction/v2/, src/xai/, modelos, dados, estudos, resultados ou documentação
histórica. A auditoria usou apenas leitura, cruzamentos de metadados e estatísticas
dos scores já salvos. Os 100 arquivos de entrada foram reconferidos sem alteração.

