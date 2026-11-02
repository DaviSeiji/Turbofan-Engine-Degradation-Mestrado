# Auditoria metodológica do pipeline de previsão de RUL — NASA C-MAPSS

Data: 02/11/2026. Versão inspecionada: `635b225` (`att requirements`), posterior à
refatoração `f172b59`. Esta auditoria cria somente este relatório. Não executa
treinamento, Optuna, KerasTuner, inferência das redes nem o Autoencoder.

## 1. Resumo executivo

O pipeline calcula corretamente os rótulos verificados e permite uma comparação
exploratória entre CNN-LSTM e Transformer sob as mesmas entradas. Entretanto,
não oferece ainda uma validação completamente independente por motor.

Cinco achados prioritários:

1. **Sobreposição confirmada no corte de validação:** um motor aparece no ajuste
   e na validação em cada FD. Os dois tensores imediatamente ao redor do corte
   compartilham exatamente 29 ciclos. Não significa que todos os motores de
   validação estejam sobrepostos.
2. **Contaminação confirmada da validação no procedimento de pré-processamento:**
   seleção de features, KMeans e scalers são ajustados antes de reservar a
   validação. Esse problema é mais abrangente que a fronteira de um único motor.
3. **Detecção inadequada de regimes em FD001/FD003:** o detector retorna múltiplos
   regimes para esses datasets de condição única, acionando seis clusters a partir
   de pequenas diferenças operacionais. Os NPZ também não equivalem a um MinMax
   global, embora faltem os objetos históricos para autenticar toda a transformação.
4. **O teste atual é por janela, com cap 125:** é uma tarefa legítima, mas não
   equivale ao benchmark terminal por motor. Motores longos recebem mais peso.
5. **Seleção pelo teste no notebook de baselines:** o campeão entre os modelos
   clássicos é escolhido pelo RMSE de teste. Esse achado não foi encontrado na
   seleção automática de hiperparâmetros da CNN-LSTM e do Transformer.

Os resultados históricos devem ser preservados como experimentos exploratórios
por janela. Uma avaliação terminal dos campeões pode usar as redes existentes.
Para estabelecer uma baseline com desenvolvimento independente por motor, será
necessário um novo conjunto versionado de pré-processamento e treinamento dos
modelos que participarão dessa comparação. Não é necessário apagar resultados,
repetir o Autoencoder agora ou refazer indiscriminadamente todos os experimentos.

### Evidências e limites da auditoria

Foram lidos código, células dos notebooks, dados brutos, os 16 NPZ de RUL,
oito bancos Optuna, CSVs de resultados e configurações ZIP dos modelos salvos.
Os bancos foram abertos com SQLite `mode=ro`, sem usar a inicialização de storage
do Optuna. Metadados do Autoencoder foram apenas consultados para a integração futura.

A reconstrução em memória usou a ordem de `groupby('unit_nr')` e os ciclos locais.
As quantidades de janelas e **todos os rótulos** coincidiram com os NPZ; a igualdade
dos 29 ciclos na fronteira foi conferida diretamente nos tensores. Os IDs de motor
atribuídos às amostras são metadados reconstruídos, não colunas armazenadas no NPZ.
Essa combinação sustenta o diagnóstico do corte atual, mas não substitui um
manifesto histórico que autentique a origem dos dados e dos transformadores.

Não foram reajustados KMeans ou scalers, nem reconstruídas previsões. Não foram
calculadas novas métricas preditivas. Estatísticas apresentadas abaixo descrevem
os dados existentes ou resumem os CSVs já salvos.

## 2. Descrição do pipeline atual

1. Ler trajetórias completas de treino, trajetórias truncadas de teste e vetor RUL.
2. Calcular RUL por ciclo e aplicar teto 125.
3. Detectar múltiplos regimes, remover features de desvio padrão exatamente zero
   e ajustar normalização no conjunto completo denominado `train`.
4. Gerar janelas de 30 ciclos, sem cruzar motores, com alvo no último ciclo.
5. Reservar a cauda de 20% das janelas em cada `fit`.
6. Selecionar hiperparâmetros pela validação, reconstruir o campeão e treiná-lo
   novamente usando a mesma regra de validação.
7. Calcular MAE/RMSE sobre todas as janelas de teste.
8. No benchmark, repetir o treino com os melhores parâmetros e seeds 1–5.

Entradas são features normalizadas; `unit_nr`, ciclo e RUL são excluídos de `X`.
As contagens de features incluem ajustes operacionais, não apenas sensores.
CNN-LSTM e Transformer carregam os mesmos NPZ em cada FD.

| Dataset | Condições / modos de falha | Motores treino / teste observados | Features | Janelas treino | Janelas teste |
| --- | --- | ---: | ---: | ---: | ---: |
| FD001 | 1 / 1 | 100 / 100 | 19 | 17.731 | 10.196 |
| FD002 | 6 / 1 | 260 / 259 | 24 | 46.219 | 26.511 |
| FD003 | 1 / 2 | 100 / 100 | 20 | 21.820 | 13.696 |
| FD004 | 6 / 2 | 249 / 248 | 24 | 54.028 | 34.092 |

As condições e os modos de falha são descritos pela [NASA](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
As quantidades da tabela foram contadas nos arquivos locais. O `readme.txt` local,
linhas 20–21, informa 248/249 para FD004, invertendo esses contadores em relação
aos arquivos. Isso é uma divergência documental; não demonstra troca dos arquivos.
O vetor `RUL_FD004.txt` possui 248 entradas e alinha com o teste local.

## 3. Auditoria do pré-processamento

### 3.1. RUL, cap e alinhamento: verificações aprovadas

Em [preproc_data_nasa.py:37](../src/common/preprocessing/preproc_data_nasa.py#L37),
o RUL de treino é `último_ciclo_do_motor − ciclo_atual`. No teste,
[linha 51](../src/common/preprocessing/preproc_data_nasa.py#L51), é
`último_ciclo_observado + RUL_final_fornecido − ciclo_atual`.
O teste não é tratado como se falhasse no último ciclo observado.

O cap é aplicado aos alvos em
[linha 69](../src/common/preprocessing/preproc_data_nasa.py#L69), para treino e
teste. Não é utilizado como feature. A checagem refez os alvos finais de cada
janela pela ordem dos dados brutos e encontrou **zero divergências nos oito
vetores y**, incluindo os motores curtos. Não se encontrou erro de deslocamento
entre fim da janela e alvo, sob a ordem reconstruída.

Os motores locais começam no ciclo 1, têm ordem temporal crescente, sem ciclos
duplicados ou linhas com NaN nos arquivos inspecionados. A geração preserva a
ordem dentro de cada motor e a auditoria não encontrou falhas na sobreposição
esperada entre janelas consecutivas.

**Limite:** a função não ordena explicitamente por ciclo nem rejeita lacunas antes
de janelar, e `dropna()` pode comprimir uma trajetória incompleta. É um risco
potencial para novas entradas, não uma falha observada nos arquivos atuais.
Fonte: [linhas 88](../src/common/preprocessing/preproc_data_nasa.py#L88) e
[151](../src/common/preprocessing/preproc_data_nasa.py#L151).

### 3.2. Regimes: erro confirmado na identificação de condição única

[detect_multi_regime:77](../src/common/preprocessing/preproc_data_nasa.py#L77)
arredonda os três ajustes a duas casas decimais e retorna verdadeiro se houver
mais de uma combinação. O resultado nos dados de treino é:

| Dataset | Combinações arredondadas | Retorno do detector | Interpretação |
| --- | ---: | --- | --- |
| FD001 | 3 | `True` | Não corresponde à única condição do protocolo |
| FD002 | 11 | `True` | Múltiplas condições esperadas; 11 combinações não são 11 regimes físicos |
| FD003 | 3 | `True` | Não corresponde à única condição do protocolo |
| FD004 | 11 | `True` | Múltiplas condições esperadas |

Em FD001/FD003, apenas `setting_1` gera três valores após arredondamento; os outros
dois ajustes têm um valor arredondado cada. Logo, ruído operacional é suficiente
para acionar o ramo que ajusta **seis** clusters nos valores contínuos originais
([linhas 98–106](../src/common/preprocessing/preproc_data_nasa.py#L98)).

Consequência: a rotina implementada normaliza FD001/FD003 em subgrupos de pequenas
variações operacionais, em vez da normalização global descrita no comentário.
Isso pode alterar escalas e a relação entre trajetórias; não foi medida a direção
nem o tamanho do efeito nas métricas. Não há evidência suficiente para atribuir
o desempenho dos modelos a esse comportamento.

Uma comparação analítica, sem ajustar estimadores, também verificou que os NPZ
de FD001/FD003 **não coincidem** com MinMax global calculado no treino completo:
diferenças máximas absolutas de aproximadamente 0,684 e 0,692. A divergência não
se limita aos ajustes: em FD001, `s_8` difere até 0,333; em FD003, `s_14` até 0,252.
Isso corrobora uma transformação diferente da global, mas não autentica o KMeans
histórico: seus centros/scalers de RUL não foram encontrados.

**Recomendação:** no próximo protocolo, declarar explicitamente o tratamento por
dataset: global em FD001/FD003 e seis condições operacionais em FD002/FD004,
com transformadores ajustados somente no grupo de ajuste. Corrigir esse caminho
muda as entradas: exige novos NPZ e treino dos modelos comparados sob essa versão.

Para FD002/FD004, o uso de seis clusters é coerente com o desenho dos datasets.
Sem os centros históricos, não se certifica a qualidade das atribuições individuais.
Transições de regime dentro de uma janela são legítimas; não demonstram falha.

### 3.3. Features constantes e consistência treino/teste

[Linhas 108–119](../src/common/preprocessing/preproc_data_nasa.py#L108): a lista
é obtida no treino e reutilizada no teste, preservando a ordem. A contagem resultante
coincidiu com as features armazenadas nos quatro `X_test`. O treino não contém
essa lista no NPZ, embora seus shapes sejam compatíveis.

Há um problema numérico de baixa gravidade: `std == 0` não remove todas as colunas
realmente constantes. Em FD001 e FD003, `s_5` e `s_16` têm um único valor bruto,
mas o desvio padrão calculado fica entre aproximadamente `1e-17` e `4e-15`.
Ambas permanecem na lista de features. Isso aumenta a dimensionalidade sem
informação; não é evidência de vazamento ou erro do RUL. Uma política futura de
constância deve ser explícita e tolerante à precisão numérica.

### 3.4. MinMax, escopo de ajuste e ausência de vazamento de teste no código

[Linhas 123–145](../src/common/preprocessing/preproc_data_nasa.py#L123): o ramo
de treino ajusta scalers; o ramo de teste só transforma com os objetos recebidos.
O `main` passa `features`, `multi_regime`, KMeans e scalers do treino ao teste
([linha 211](../src/common/preprocessing/preproc_data_nasa.py#L211)).

**Não foi encontrado ajuste dos transformadores no teste no pipeline modular.**
Isso não certifica toda execução histórica sem seus manifestos, mas é diferente
do problema confirmado da validação: o ajuste usa o `train` completo, incluindo
as observações depois reservadas pelo Keras.

A recomendação de aprender as transformações apenas no grupo de ajuste também
se aplica aos grupos internos de validação, conforme a orientação de
[pré-processamento e vazamento do scikit-learn](https://scikit-learn.org/1.7/common_pitfalls.html).
O código demonstra esse escopo inadequado; não demonstra que cada parâmetro de
escala mudaria ao remover a validação, nem permite quantificar um viés no MAE.

Valores de teste fora de `[0,1]` são esperados com MinMax de treino sem clipping.
Foram observados mínimos negativos nos quatro FDs, chegando a aproximadamente
−0,6 em FD004. Isso não prova vazamento nem demanda clipping automático.

### 3.5. Persistência insuficiente para reproduzir entradas novas

[Linhas 219–224](../src/common/preprocessing/preproc_data_nasa.py#L219) salvam os
arrays e a lista de features do teste, mas não os transformadores de RUL,
metadados por janela, hashes ou a partição dos motores. Não foram encontrados
scalers/KMeans de RUL nos artefatos locais. Os objetos do Autoencoder pertencem
a outro pipeline e não resolvem essa ausência.

Os NPZ preservados permitem reutilizar as entradas existentes, inclusive suas
últimas janelas. Aplicar um campeão a uma trajetória nova ou reproduzir a
normalização completa exige recuperar os objetos originais ou demonstrar uma
reconstrução equivalente. Não se deve aplicar silenciosamente uma normalização nova
aos pesos antigos.

## 4. Auditoria das divisões treino/validação/teste

### 4.1. O que o Keras efetivamente faz

A [documentação de `Model.fit`](https://keras.io/api/models/model_training_apis/)
define a validação pela cauda dos arrays, antes do embaralhamento. Isso foi
conferido também na implementação instalada de Keras 3.15.0,
`array_slicing.py:461–520`: `split_at = floor(N * (1 − validation_split))`.
O `shuffle` padrão reorganiza o grupo de ajuste; não transforma esse corte em
uma divisão aleatória por motor.

As janelas são concatenadas em ordem crescente de motor. Não há uma reserva
prévia por `unit_nr`. O mesmo corte é utilizado nos trials, no treinamento final
e no benchmark multiseed.

### 4.2. Quantificação nos NPZ atuais

| FD | Janelas de ajuste | Janelas de validação | Motores no ajuste / validação | Motor compartilhado | Janelas desse motor: ajuste / validação | Último ciclo no ajuste / primeiro na validação |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | 14.184 | 3.547 | 82 / 19 | 82 | 155 / 30 | 184 / 185 |
| FD002 | 36.975 | 9.244 | 208 / 53 | 208 | 25 / 96 | 54 / 55 |
| FD003 | 17.456 | 4.364 | 80 / 21 | 80 | 115 / 3 | 144 / 145 |
| FD004 | 43.222 | 10.806 | 199 / 51 | 199 | 68 / 79 | 97 / 98 |

Em todos os FDs, `X[split_at−1,1:,:]` e `X[split_at,:−1,:]` são exatamente iguais:
**29 dos 30 ciclos são compartilhados na fronteira**. Não foi necessário executar
um modelo para demonstrar essa dependência.

Contando todos os pares de janelas entre as duas partes que compartilham ao menos
um ciclo do motor de fronteira:

| FD | Pares sobrepostos entre ajuste/validação | Janelas de validação afetadas | Fração das janelas de validação |
| --- | ---: | ---: | ---: |
| FD001 | 435 | 29 | 0,82% |
| FD002 | 425 | 29 | 0,31% |
| FD003 | 84 | 3 | 0,069% |
| FD004 | 435 | 29 | 0,27% |

Não se deve extrapolar esses percentuais para dizer que toda a validação é uma
cópia do ajuste. A maior parte de seus motores não entra no ajuste dos pesos da
rede. Já o pré-processamento usa todas as observações de treino antes do corte.

### 4.3. Viés de composição e consequências

A validação compreende IDs 82–100, 208–260, 80–100 e 199–249, respectivamente;
inclui somente a cauda da trajetória do motor de fronteira e trajetórias completas
dos motores seguintes. Por exemplo, as três janelas de validação do motor 80 de
FD003 têm alvos 2, 1 e 0. A divisão não é aleatória nem garante representação dos
modos de falha. O ID, por si só, não prova uma ordenação por dificuldade: esse viés
de representatividade permanece **risco potencial**, não fato demonstrado.

Optuna, EarlyStopping e a seleção final consultam reiteradamente essa mesma
validação. Não há seleção por teste nesses três passos dos preditores neurais,
mas a validação não estima de maneira limpa desempenho em motores nunca vistos
pelo desenvolvimento inteiro.

Para uma tarefa de generalização a motores novos, reservar grupos completos
antes da seleção de features e da normalização é metodologicamente mais adequado.
Para uma tarefa diferente, de continuação de um motor já conhecido, uma divisão
temporal pode ser legítima, mas precisa ser declarada e tratar a interseção de
ciclos conforme a pergunta científica. Esse não é o protocolo terminal de novos
motores que se pretende comparar aqui.

### 4.4. Distinções que não devem ser confundidas

| Situação | Diagnóstico desta auditoria |
| --- | --- |
| Treino e teste entram juntos no `fit` dos scalers ou na busca neural | Não encontrado no código modular inspecionado |
| Validação participa do pré-processamento | Confirmado pelo escopo do `fit` antes do corte |
| Janelas do mesmo motor cruzam ajuste/validação | Confirmado em um motor por FD e nos tensores da fronteira |
| Sobreposição entre janelas do mesmo grupo | Dependência estatística esperada; não é automaticamente vazamento entre conjuntos |
| Mesmos números de `unit_nr` em treino e teste | Não significam o mesmo motor físico: são namespaces distintos |
| RUL futuro usado para construir o alvo de teste | Correto para avaliação supervisionada; não é uma feature de entrada |

## 5. Auditoria do treinamento e Optuna

### 5.1. Arquiteturas e espaços de busca

| Pipeline | Arquitetura / espaço | Busca e critério |
| --- | --- | --- |
| CNN | 1–3 Conv1D; filtros iniciais 32/64/96/128; kernel 3/5; filtros adicionais 32–128 e 32–64; Dense 32/64/128; dropout 0,1–0,5; LR 1e−2/1e−3/1e−4 | KerasTuner Hyperband, `val_loss`, `max_epochs=40`, fator 3; não são 30 trials Optuna |
| CNN-LSTM | filtros 32/64/128; kernel inteiro 3–5; LSTM 32/64/128; dropout 0,1–0,4; LR log 1e−4–1e−2; batch 32/64/128 | Optuna, menor MSE de validação, 30 tentativas adicionais por chamada, até 40 épocas |
| Transformer | `d_model` 32/64/128; heads 2/4/8; FF 64/128/256; 1–3 blocos; dropout 0,1–0,4; LR log 1e−4–1e−2; batch 32/64/128 | Mesmo objetivo, quantidade solicitada e limite de épocas da CNN-LSTM |

Fontes: [tune_cnn:23](../src/rul_prediction/training/tune_cnn.py#L23),
[tune_hybrid:19](../src/rul_prediction/training/tune_hybrid.py#L19) e
[tune_transformer:75](../src/rul_prediction/training/tune_transformer.py#L75).

O Transformer projeta as features antes das conexões residuais e utiliza
`key_dim=d_model//num_heads`. Todas as combinações atuais são divisíveis.
Possui embedding temporal treinável, atenção, normalização, feed-forward,
pooling global e saída linear. Não foi encontrado erro dimensional nessa construção.
Os builders do benchmark reproduzem essas arquiteturas e a CNN-LSTM, conforme
[model_builders.py:6](../src/rul_prediction/models/model_builders.py#L6) e
[46](../src/rul_prediction/models/model_builders.py#L46).

### 5.2. Trials, treino final e restauração de pesos

CNN-LSTM/Transformer usam Adam, MSE e MAE. Nos trials, EarlyStopping monitora
`val_loss`, tem patience 5 e restaura os melhores pesos; o objetivo retorna o mínimo
da curva de validação. O MedianPruner tem `n_warmup_steps=5`.
No treino final, ambos reconstruem um modelo novo, usam até 50 épocas, patience 7,
restauração dos melhores pesos e o batch selecionado. A reserva de validação
continua existindo: o treino final não usa 100% das janelas no ajuste dos pesos.

A CNN usa patience 5 também no treino final e batch padrão de `fit` (32),
sem buscá-lo como hiperparâmetro. É uma diferença de protocolo em relação aos
outros modelos, não um erro automático. Os orçamentos da CNN via Hyperband e das
outras arquiteturas via Optuna não são equivalentes. Fonte:
[tune_cnn:91](../src/rul_prediction/training/tune_cnn.py#L91) e
[133](../src/rul_prediction/training/tune_cnn.py#L133).

Restaurar pesos é coerente com selecionar pela melhor validação,
conforme [EarlyStopping](https://keras.io/api/callbacks/early_stopping/).
Não torna independente uma validação contaminada pelo pré-processamento.

### 5.3. Estudos existentes: orçamento real

Contagem feita por consultas SQL somente leitura:

| Dataset | CNN-LSTM: total / completos / podados | Transformer: total / completos / podados | Melhor trial CNN-LSTM / Transformer, índice começando em zero |
| --- | ---: | ---: | ---: |
| FD001 | 60 / 21 / 39 | 30 / 12 / 18 | 25 / 6 |
| FD002 | 30 / 11 / 19 | 30 / 14 / 16 | 2 / 6 |
| FD003 | 30 / 16 / 14 | 30 / 11 / 19 | 25 / 2 |
| FD004 | 30 / 6 / 24 | 30 / 18 / 12 | 4 / 9 |

Não foram observados trials com falha ou em andamento nesses oito bancos.
Podas contam no orçamento; 30 tentativas não significam 30 modelos treinados
até o limite de épocas.

`load_if_exists=True` seguido de `optimize(..., n_trials=30)` solicita mais
tentativas, não completa um total global de 30. Fontes:
[tune_hybrid:80](../src/rul_prediction/training/tune_hybrid.py#L80),
[tune_transformer:140](../src/rul_prediction/training/tune_transformer.py#L140) e
[API Optuna 4.9](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.study.Study.html).
O comentário do Transformer explicita essa intenção. Na CNN-LSTM, a existência
de 60 tentativas em FD001 é compatível com continuação do estudo; o histórico
não prova quantas chamadas foram feitas.

**Nuance importante:** o melhor trial da CNN-LSTM FD001 é o 25, dentro das primeiras
30 tentativas. As tentativas adicionais não diminuíram o melhor objetivo atual.
O orçamento desigual precisa ser documentado, mas não justifica sozinho refazer
esse campeão nem atribuir sua qualidade ao dobro do orçamento.

### 5.4. Persistência, retomada e rastreabilidade

- A CNN-LSTM salva o campeão e o CSV nos mesmos caminhos sem bloquear arquivos
  existentes ([linha 139](../src/rul_prediction/training/tune_hybrid.py#L139)).
  Uma nova execução pode acrescentar trials e substituir pesos/CSV.
- O Transformer verifica a existência de todos os campeões solicitados antes
  da busca ([linha 126](../src/rul_prediction/training/tune_transformer.py#L126)).
  Hoje, com esses arquivos presentes, encerraria antes de acrescentar trials.
  Se a guarda for superada por arquivamento prévio, o estudo existente recebe
  mais 30 tentativas. Seu CSV preserva linhas de outros datasets.
- A CNN usa `overwrite=False` no tuner, mas pode substituir seu modelo final.
  Nenhum `oracle.json` de KerasTuner foi encontrado localmente: não é possível
  confirmar o número de configurações históricas ou o estado de sua retomada.
- Arquiteturas e learning rates dos oito campeões CNN-LSTM/Transformer foram
  comparados, via `config.json` dentro dos `.keras`, aos melhores parâmetros dos
  bancos. São compatíveis, respeitando a precisão de ponto flutuante do LR.
  Não foi encontrada uma divergência concreta de hiperparâmetros nesses campeões.
- Os bancos não contêm atributos de estudo identificando hashes de NPZ/split.
  Os CSVs do benchmark também não registram versão de dados ou de parâmetros.
  Isso permite, em uma retomada futura, misturar experimentos sob a mesma chave
  `Model, Dataset, Seed` se os arquivos de entrada forem substituídos.

O último ponto é uma limitação confirmada de rastreabilidade e um risco potencial
de mistura; **não há evidência de que essa mistura já tenha ocorrido**.

### 5.5. Uso do teste na seleção: separar os escopos

Nos scripts CNN, CNN-LSTM e Transformer, a busca monitora validação e o teste é
avaliado depois do treino final. Carregar `X_test` antes da busca, como faz a
CNN-LSTM, não significa usá-lo na função objetivo.

Há uma exceção confirmada no notebook
[01_baseline.ipynb:544](../notebooks/rul_prediction/01_baseline.ipynb#L544),
célula 2: cada modelo clássico é avaliado em `X_test`; em
[linha 558](../notebooks/rul_prediction/01_baseline.ipynb#L558),
`if rmse < melhor_rmse` escolhe o campeão pelo teste. Os resultados individuais
podem ser preservados como exploração, mas o vencedor não tem avaliação de seleção
independente. A escolha deve passar para uma validação por motor, seguida de um
teste final fixado. Isso não exige apagar os baselines existentes.

A consulta repetida ao teste por decisões humanas sobre arquiteturas, tratamento
de dados ou seeds também pode contaminá-lo. O código não permite reconstituir
essas decisões: é **risco potencial**, não vazamento adicional comprovado.

## 6. Auditoria das métricas e do protocolo de teste

### 6.1. Protocolo A atual versus protocolo B terminal

[preproc_data_nasa.py:216](../src/common/preprocessing/preproc_data_nasa.py#L216)
usa `last_window_only=False`. Logo, todos os scripts que carregam os NPZ fazem
**avaliação por janela**. A tarefa tradicional descrita pela
[NASA](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) tem como alvo
os ciclos restantes após o último ciclo observado de cada trajetória de teste.

| Aspecto | A: todas as janelas, atual | B: última janela de cada motor |
| --- | --- | --- |
| Unidade de observação | Janela temporal | Motor no fim da observação |
| Quantidade de previsões por motor | Varia com a duração | Uma |
| Alvo | RUL no fim de cada janela, limitado a 125 | RUL final fornecido, com política de cap declarada |
| Peso no agregado | Proporcional ao número de janelas | Igual por motor |
| Uso científico | Monitoramento ao longo da trajetória | Comparação terminal entre motores |

Se `n_m` é o número de janelas do motor e `N=Σn_m`:

`MAE_micro = Σ (n_m/N) MAE_m`

`RMSE_micro = sqrt[Σ (n_m/N) MSE_m]`

RMSE micro não é a média ponderada dos RMSEs. A média de MAEs por motor é uma
terceira métrica, útil e diferente de avaliar somente a última janela.

### 6.2. Peso desigual e distribuição dos alvos

| FD | Menor / maior número de janelas por motor de teste | Razão dos pesos extremos | Alvos de janela iguais a 125 | Motores com RUL final bruto >125 |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 2 / 274 | 137,0 | 51,10% | 11 / 100 |
| FD002 | 1 / 338 | 338,0 | 50,83% | 57 / 259 |
| FD003 | 9 / 446 | 49,56 | 62,98% | 15 / 100 |
| FD004 | 1 / 457 | 457,0 | 66,21% | 67 / 248 |

O MAE/RMSE atual atribui muito peso a regiões anteriores da trajetória, frequentemente
no platô 125. Não é possível inferir, sem novas previsões, se o resultado terminal
ficará maior ou menor. Não se deve converter numericamente o MAE atual em MAE por
motor nem compará-lo diretamente com publicações de protocolo diferente.

O cap 125 é consistente nos alvos atuais e pode ser uma escolha legítima de
pesquisa. Não é uma regra universal imposta pela NASA. Uma comparação externa
precisa alinhar cap, instante avaliado, features, padding e métrica. Recomenda-se
uma tabela terminal com cap 125 para manter a tarefa de treino; uma avaliação
contra RUL bruto, se desejada, deve ser identificada separadamente.

### 6.3. Motores curtos e padding

O código repete o primeiro ciclo à esquerda para produzir uma única janela quando
há menos de 30 observações ([linha 166](../src/common/preprocessing/preproc_data_nasa.py#L166)).
Não utiliza ciclos de outro motor nem dados posteriores ao fim observado.

- FD001/FD003: nenhum motor de teste curto.
- FD002: seis — IDs 28, 59, 83, 122, 201 e 223; entre 21 e 29 ciclos.
- FD004: onze — IDs 10, 19, 28, 125, 141, 156, 164, 204, 229, 239 e 246;
  entre 19 e 29 ciclos.
- No treino, nenhum motor tem menos de 30 ciclos.

O padding preserva esses motores na avaliação, mas cria entradas de um tipo ausente
do treino. É uma escolha de imputação a documentar, não um vazamento. Excluir esses
motores ou usar outra política altera a coorte de teste; não fazer essa mudança
silenciosamente ao comparar modelos.

### 6.4. Novas métricas possíveis sem novo treinamento

Os campeões salvos permitem, em uma tarefa futura, selecionar as últimas janelas
dos NPZ e executar somente inferência para obter métricas terminais. Também
permitem MAE macro por motor, estratos de RUL, análise com/sem padding, curvas de
erro e score assimétrico com fórmula/protocolo explicitados.

Os agregados MAE/RMSE atuais não contêm informação suficiente para recuperar essas
métricas. Nenhuma delas foi produzida nesta auditoria.

## 7. Reprodutibilidade e seeds

### 7.1. Busca e treinamento original

- CNN-LSTM: não há seed explícita do sampler ou do treinamento; tampouco limpeza
  de sessão entre trials. A ausência de seeds limita reprodução exata. O risco
  de recursos acumulados é operacional; não prova viés nas métricas.
- Transformer: TPESampler com seed 42, seed de treino `42 + trial.number` e seed 42
  para o campeão. Limpa sessões e libera modelos. Isso melhora controle, mas não
  garante determinismo de todos os kernels/hardware ou retomada idêntica do sampler.
- CNN: não há seed explícita da busca/treino nesse script.
- Nenhum dos preditores força operações determinísticas nem grava um manifesto
  completo de ambiente e dados por execução.

### 7.2. Benchmark multiseed: o que é comparável

[testar_modelos.py:68](../src/rul_prediction/evaluation/testar_modelos.py#L68)
limpa a sessão e fixa a seed antes de construir cada modelo. Ambos usam os mesmos
NPZ, corte de validação, Adam/MSE, até 50 épocas, patience 7, restauração dos melhores
pesos e o batch escolhido em seus estudos. As arquiteturas e os batches podem
diferir legitimamente. Mesma seed não significa inicialização ou permutações
idênticas entre arquiteturas.

[experiment_utils.py:29](../src/rul_prediction/experiment_utils.py#L29) carrega os
melhores parâmetros dos estudos existentes. Faz uma cópia temporária do banco
antes de inicializar Optuna; a auditoria não executou essa rotina, usando SQL
somente leitura diretamente nos originais.

O CSV detalhado tem **40 linhas**, cinco seeds distintas (1–5) em cada um dos oito
grupos modelo/dataset, sem chaves duplicadas. As médias e desvios amostrais do CSV
resumo coincidiram com a recomputação em memória (`ddof=1`, tolerância 1e−8).

As métricas implementadas em
[testar_modelos.py:50](../src/rul_prediction/evaluation/testar_modelos.py#L50) são
MAE, raiz do MSE, mediana, percentil 95 e máximo dos erros absolutos, depois de
alinhar os vetores escalares. Não foi encontrado erro nessas fórmulas.
O tempo registrado inclui o `fit`, sem incluir a construção nem a inferência.

| FD | CNN-LSTM MAE / RMSE, média ± desvio | Transformer MAE / RMSE, média ± desvio |
| --- | --- | --- |
| FD001 | 10,92 ± 0,85 / 14,66 ± 0,71 | 10,12 ± 0,52 / 14,74 ± 0,69 |
| FD002 | 12,06 ± 0,31 / 16,28 ± 0,26 | 11,06 ± 0,26 / 16,15 ± 0,31 |
| FD003 | 7,88 ± 1,65 / 12,21 ± 2,01 | 7,95 ± 0,21 / 12,55 ± 0,10 |
| FD004 | 22,74 ± 12,86 / 26,15 ± 11,09 | 8,66 ± 0,21 / 14,25 ± 0,21 |

Valores transcritos/resumidos dos resultados existentes, todos por janela.
No FD004, as seeds CNN-LSTM 1, 2 e 5 têm MAEs próximos de 30–33; as seeds 3 e 4,
próximos de 8–9. A variabilidade está confirmada. Não foi demonstrado se decorre
de convergência, inicialização, LR ou outro fator. Não eliminar seeds por terem
resultado ruim no teste. O desempenho não pode ser atribuído aos regimes ou aos
dois modos de falha apenas com essa tabela.

O benchmark é útil para comparar a variabilidade do treinamento no protocolo
atual. Não estima, por cinco seeds, a incerteza da amostragem de motores nem
corrige os problemas de validação compartilhados.

### 7.3. Limite para reaproveitar o benchmark

O benchmark retorna métricas e descarta cada rede
([linhas 84–93](../src/rul_prediction/evaluation/testar_modelos.py#L84)); não salva
pesos por seed nem previsões individuais. Não foram encontrados esses artefatos
locais. Logo, uma avaliação terminal **das mesmas cinco instâncias históricas**
não pode ser reconstruída apenas desses CSVs. Seria necessário localizar os
pesos/previsões fora do inventário ou repetir esses treinamentos em uma nova versão.
Os campeões das buscas originais não substituem as cinco instâncias do benchmark.

Indicadores adicionais recomendados, sem implementação:

- Diferenças pareadas entre modelos nas mesmas seeds, com todos os valores individuais.
- Métricas terminais e MAE/MSE com peso igual por motor.
- Intervalos por reamostragem de motores, sem tratar janelas sobrepostas como independentes.
- Erro por faixa de RUL, por duração e por presença de padding.
- Curvas de treino/validação, época de melhor validação e taxa de execuções instáveis.
- Custo de inferência, parâmetros e condições de hardware, separados do tempo de `fit`.

## 8. Requisitos para integração com Autoencoder

### 8.1. Compatibilidade não significa compartilhar a normalização

Ambos usam 30 ciclos, mas os transformadores têm objetivos distintos:

| Aspecto | RUL atual | Autoencoder existente, consultado apenas em código/metadados |
| --- | --- | --- |
| Features FD001/002/003/004 | 19 / 24 / 20 / 24 | 20 / 24 / 20 / 24 |
| Dados usados para ajustar normalização | Trajetórias completas do `train` antes da reserva Keras | Primeiros 30% de motores de ajuste nominal |
| Partição | Corte na cauda das janelas | Divisão aleatória por motor, seed 42 |
| Condições únicas | Detector atual aciona múltiplos regimes | Global para FD001/FD003 |
| Motores com <30 ciclos | Padding e uma janela | Sem janela artificial |
| Metadados por janela | Ausentes no NPZ | Motor, início e fim armazenados |

Fontes AE: [preprocessing.py:82](../src/anomaly_detection/preprocessing.py#L82),
[121](../src/anomaly_detection/preprocessing.py#L121) e
[138](../src/anomaly_detection/preprocessing.py#L138). Não foram executadas.

Não passar os tensores normalizados de RUL ao AE como se fossem suas entradas,
nem normalizar RUL com o scaler nominal do AE. Para fusão tardia, cada ramo mantém
seu processamento e associa a representação do preditor ao score correspondente
à mesma janela física.

### 8.2. Alinhamento e janelas ausentes

A chave segura é `(Dataset, Split, unit_nr, start_cycle, end_cycle)`, com identidade
de motor diferente entre treino e teste. Não concatenar arrays por índice.

| FD | Janelas teste RUL | Scores teste AE existentes | Janelas RUL sem equivalente completo no AE |
| --- | ---: | ---: | ---: |
| FD001 | 10.196 | 10.196 | 0 |
| FD002 | 26.511 | 26.505 | 6 |
| FD003 | 13.696 | 13.696 | 0 |
| FD004 | 34.092 | 34.081 | 11 |

As diferenças coincidem com os motores curtos. É necessário escolher previamente
uma política para eles e aplicá-la igualmente aos modelos comparados. Não preencher
score ausente com zero, pois zero implicaria reconstrução perfeita. As opções
incluem um estado explícito de ausência ou uma estratégia de janela curta validada;
excluir motores também é uma mudança de coorte que precisa ser declarada.

### 8.3. Os Autoencoders atuais não respeitam a validação atual de RUL

O cruzamento entre `data/processed_anomaly/FD00X/nominal_units.csv` e os motores
reconstruídos da validação de RUL mostrou:

| FD | Motores de validação RUL | Destes, usados no ajuste nominal do AE existente |
| --- | ---: | ---: |
| FD001 | 19 | 17 |
| FD002 | 53 | 43 |
| FD003 | 21 | 18 |
| FD004 | 51 | 44 |

A incompatibilidade das partições é confirmada. **Ainda não é vazamento de uma
integração executada**, pois essa integração não existe. Se esses AEs fossem
reutilizados diretamente para selecionar o preditor integrado, sua validação
incluiria motores que influenciaram o ramo de features do AE.

Antes da integração, definir os grupos por motor para o sistema inteiro. Dentro
de cada divisão de desenvolvimento, ajustar AE, scalers, KMeans e qualquer
transformação de scores sem os motores da validação do preditor. O teste deve
passar somente por transformações congeladas.

Scores calculados em motores de ajuste do próprio AE não são, por definição,
vazamento de rótulo RUL: o AE não usa esse alvo. Porém, o contraste entre scores
no ajuste e em motores novos pode refletir sobreajuste de reconstrução. Avaliar
esse contraste; se relevante, gerar scores fora da amostra por grupos de motores
(cross-fitting) ou usar um AE congelado em um subconjunto de desenvolvimento.
Cross-fitting não é uma exigência automática para toda representação não supervisionada.

Para fusão com **MSE contínuo**, um threshold binário não é necessário. Se forem
usados flags, scores padronizados ou regras calibradas, sua calibração precisa
de escopo documentado, fora do teste. Uma calibração independente é recomendável
para alegações de detecção; não confundir isso com o uso do score como feature.

Os AEs atuais continuam válidos para seus experimentos de anomalia. Só precisam
de novos ajustes se a futura divisão conjunta os tornar incompatíveis com os
grupos de desenvolvimento definidos para a integração.

### 8.4. Avaliação final independente

Fixar divisão, cap, padding, métricas e seleção antes da comparação. A baseline
CNN-LSTM/Transformer e suas versões integradas devem usar as mesmas janelas,
coortes, seeds e regras de seleção. O teste C-MAPSS já foi consultado diversas
vezes; não é possível torná-lo retrospectivamente inédito. Novas decisões devem
ser tomadas em desenvolvimento e as comparações futuras nesse teste devem declarar
seu caráter de benchmark conhecido, sem alegar uma primeira avaliação cega.

## 9. Tabela de problemas e prioridades

**Confirmado** descreve o que código/dados demonstram; **risco** exige condições
ou verificação adicional; **melhoria** não implica erro do experimento atual.
A gravidade refere-se à interpretação científica pretendida, não a uma estimativa
numérica de quanto MAE/RMSE mudaria.

| ID | Classe / gravidade | Problema e evidência localizada | Consequência / impacto potencial | Correção recomendada e necessidade de repetição |
| --- | --- | --- | --- | --- |
| A1 | Confirmado / alta | Motor e 29 ciclos cruzam o corte; `preproc_data_nasa.py:151–175`, `tune_hybrid.py:50`, `tune_transformer.py:99`, `testar_modelos.py:80`; tabelas da seção 4 | Validação inclui parte de uma série vista no ajuste; possível favorecimento local de seleção e parada | Dividir por motor antes de janelar; para medir o efeito limpo, treinar uma nova baseline. Não corrigível apenas recalculando o teste |
| A2 | Confirmado / alta | Pré-processamento antes da reserva; `preproc_data_nasa.py:98–145` | Validação influencia transformações; estimativa de seleção sem independência completa | Ajustar todos os transformadores somente no grupo de ajuste; novos NPZ e novos treinos para o protocolo corrigido |
| A3 | Confirmado no código/dados / alta | Detector retorna `True` para FD001/FD003; `preproc_data_nasa.py:77–103` | Subdivisão artificial de condição única e escalas diferentes do protocolo global descrito; efeito preditivo desconhecido | Política por dataset; recuperar a proveniência antiga e gerar versão corrigida. Reprocessar/reteinar FD001/FD003 para essa comparação |
| A4 | Melhoria; limitação confirmada / média | Todas as janelas no teste; `preproc_data_nasa.py:216`, `testar_modelos.py:84` | Peso desigual e tarefa diferente da terminal; não é erro intrínseco do MAE/RMSE | Adicionar avaliação terminal e macro; campeões existentes podem ser reavaliados sem treino |
| A5 | Confirmado / alta, escopo baselines clássicos | Vencedor escolhido pelo teste; notebook `01_baseline`, célula 2, linha 558 do JSON | Seleção usa o conjunto que depois representa sua avaliação final | Escolher modelos por validação de motores; repetir a seleção/ajuste apenas dos baselines incluídos na nova comparação |
| A6 | Orçamento confirmado; risco comparativo / média | FD001 CNN-LSTM com 60 trials versus 30; `tune_hybrid.py:93`, bancos locais | Esforço desigual e possibilidade de trocar o campeão ao retomar; neste caso, o melhor trial continua sendo 25 | Fixar teto de tentativas e versões; não exige repetição automática, pois o mínimo já estava nas primeiras 30 |
| A7 | Limitação confirmada / média | Sem seeds explícitas em CNN/CNN-LSTM; `tune_cnn.py`, `tune_hybrid.py`; Transformer e benchmark explicitam seeds | Reprodução exata limitada; incerteza de busca distinta entre modelos | Gravar ambiente/seeds/determinismo e definir orçamento; repetir somente os novos experimentos que precisarem dessas garantias |
| A8 | Limitação confirmada; mistura apenas potencial / média | Sem scalers/metadados de RUL persistidos; `preproc_data_nasa.py:219–224`; chave de benchmark sem versão, `testar_modelos.py:26` | Difícil aplicar redes a novos dados e autenticar retomadas; não há prova de mistura atual | Manifestos, hashes, IDs por janela, transformadores e estudos separados por versão; sem treino para preservar/inventariar, novo treino se entradas mudarem |
| A9 | Limitação confirmada / média | Benchmark descarta redes e retorna agregados; `testar_modelos.py:84–93` | Não permite avaliar retrospectivamente as mesmas cinco redes em outro protocolo | Salvar previsões/identidades e pesos nas próximas execuções; novas métricas históricas por seed exigem recuperar artefatos ou repetir aquelas execuções |
| A10 | Confirmado / baixa | `std == 0` mantém `s_5`/`s_16` constantes em FD001/FD003; `preproc_data_nasa.py:112–114` | Features sem informação e sensibilidade numérica; não é vazamento | Critério explícito de constância na próxima versão; remoção altera shape e exige novos treinos, sem urgência isolada |
| A11 | Risco potencial / baixa nos dados atuais | Sem sort/gap check após `dropna`; `preproc_data_nasa.py:89,151` | Entradas futuras desordenadas poderiam produzir janelas inválidas; não ocorreu nos dados auditados | Validar ordem e continuidade; apenas checagens, sem repetir os dados atuais se continuarem aprovados |
| A12 | Risco potencial / alta | Decisões manuais baseadas em teste repetidamente consultado não são rastreáveis | Sobreajuste ao benchmark por decisões do pesquisador | Diário/versionamento de decisões e seleção em desenvolvimento; não declarar teste já visto como avaliação cega |
| A13 | Partições incompatíveis confirmadas; vazamento futuro potencial / alta | `anomaly_detection/preprocessing.py:141–153`, metadados locais versus corte RUL | Reutilizar AE pode expor motores da validação ao extrator de features | Coordenar grupos antes da integração; treinar AE apenas se necessário para essa nova divisão, preservando a etapa de anomalias existente |
| A14 | Variabilidade confirmada; causa potencial / média | CNN-LSTM FD004, CSV multiseed, seção 7 | Média sensível a execuções instáveis; causa não pode ser imputada aos regimes | Manter todas as seeds, estudar curvas/validação e reportar dispersão; não selecionar seeds pelo teste |
| A15 | Melhoria / média | Falta de métricas terminais, macro e incerteza por motor | Evidência limitada sobre generalização e sobre ganho da futura integração | Adicionar métricas e intervalos por motor; usar modelos/previsões existentes quando disponíveis |

As linhas são da versão inspecionada. Os IDs A1/A2 têm correções complementares:
só mover o corte para outra posição dos NPZ antigos não remove a participação
da validação no ajuste dos transformadores.

## 10. Plano de correções recomendado

Nenhuma etapa abaixo foi implementada nesta auditoria.

1. **Congelar a evidência atual.** Preservar NPZ, pesos, bancos, CSVs e suas
   interpretações como protocolo histórico por janela. Registrar hashes, versão
   e limitações; não continuar estudos antigos para preencher lacunas de documentação.
2. **Definir a pergunta e as métricas.** Adotar como principal a avaliação terminal
   por motor; manter a avaliação por janela como complementar. Declarar cap 125,
   política para motores curtos e coorte idêntica entre os preditores comparados.
3. **Reavaliar os campeões sem retreinar.** Em uma tarefa específica de inferência,
   obter métricas terminais/macro usando os NPZ existentes e os pesos preservados.
   Essa análise esclarece a mudança de protocolo, mas não corrige A1/A2.
4. **Criar uma versão independente por motor.** Dividir os motores brutos antes
   de qualquer ajuste; ajustar features, regimes e normalização só no grupo de
   ajuste; salvar metadados e objetos. Corrigir o detector de condição única.
   Não substituir os NPZ históricos.
5. **Validar primeiro um experimento pequeno.** FD001 verifica condição única e
   ausência de compartilhamento; FD002 verifica o ramo de seis regimes e padding.
   Conferir rótulos e hashes antes de expandir para FD003/FD004.
6. **Estabelecer a baseline conjunta.** Comparar CNN-LSTM e Transformer com a mesma
   versão de dados, critérios de parada, seeds, orçamento declarado e avaliação
   terminal. Hiperparâmetros históricos podem ser congelados para uma comparação
   inicial de baixo custo; se houver nova otimização, usar estudos novos e a
   validação corrigida. Não chamar o ótimo antigo de ótimo do novo protocolo.
7. **Preservar a granularidade dos novos resultados.** Salvar previsões por motor,
   identidades das janelas, parâmetros, curvas, melhor época e pesos por seed,
   evitando a necessidade de retreinar para acrescentar uma métrica.
8. **Só então integrar o score AE.** Compartilhar a divisão por motor, manter os
   dois pré-processadores, associar scores por chave e definir a política de
   ausência. Comparar com/sem score sob o mesmo protocolo congelado. Não escolher
   a integração pela inspeção repetida do teste.

## Quais experimentos anteriores continuam válidos e quais precisam ser repetidos?

| Experimento / objetivo | O que pode ser preservado | Ação efetivamente necessária |
| --- | --- | --- |
| MAE/RMSE históricos por janela dos campeões | Integralmente como descrição do protocolo atual, com ressalvas de seleção/validação | Nenhuma repetição para mantê-los; corrigir a interpretação de comparabilidade |
| Comparação multiseed atual CNN-LSTM/Transformer | CSVs e dispersões das 40 execuções; comparação exploratória sob o mesmo pipeline | Novas sínteses pareadas podem usar o CSV; não produzem avaliação terminal nem independência por motor |
| Avaliação terminal dos campeões salvos | Mesmos pesos, NPZ e alvo final reconstruído | Apenas inferência e novas métricas em arquivos novos; não exige novo treinamento |
| Avaliação terminal das mesmas cinco instâncias do benchmark | Agregados atuais não bastam; pesos/previsões por seed não localizados | Localizar esses artefatos ou repetir o benchmark se essa análise for necessária; não usar o campeão como substituto |
| Baseline neural com validação realmente independente | Arquiteturas, código conceitual e parâmetros anteriores como ponto de partida | Nova divisão, novos transformadores e treino dos modelos incluídos. Para alegar essa propriedade nos quatro FDs, a nova avaliação precisa cobrir os quatro |
| Teste do tratamento global correto em FD001/FD003 | Preservar o ramo antigo como ablação histórica | Nova versão de dados e treino para medir a mudança; não aplicar scaler global aos pesos antigos |
| CNN e baselines clássicos históricos | Resultados individuais exploratórios e pesos | Não precisam ser repetidos se não forem usados como controles da nova baseline; se incluídos, seguir o mesmo protocolo e corrigir a seleção pelo teste |
| Novas métricas macro, por RUL e por motor | Campeões podem ser reutilizados; CSVs agregados não contêm previsões | Inferência adicional ou reutilização de previsões, quando disponíveis; não é necessariamente novo treino |
| XAI dos modelos existentes | Explicações continuam descrevendo aqueles modelos e dados | Não há razão para refazer tudo agora. Novos preditores/entradas exigirão novas explicações para suas próprias alegações |
| Experimentos atuais do Autoencoder | Preservação integral como etapa separada de anomalias | Nenhuma repetição nesta auditoria; novos ajustes apenas se a futura partição conjunta não for compatível |

**Baseline recomendada antes da integração:** uma comparação CNN-LSTM versus
Transformer com divisão por motor antes do pré-processamento, tratamento explícito
de condições operacionais, métricas terminais como resultado principal e avaliação
por janela como complemento. Começar com parâmetros históricos fixados e poucas
execuções planejadas é suficiente para validar o protocolo; uma nova busca completa
só se justifica depois de essa versão estar auditada e congelada.

## Referências e rastreabilidade

- [NASA — descrição e objetivo do C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data):
  condição/modos de falha e alvo após o último ciclo observado.
- [Keras — Model.fit](https://keras.io/api/models/model_training_apis/): seleção
  da validação antes do embaralhamento; conferida com o código instalado 3.15.0.
- [scikit-learn 1.7 — common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html):
  escopo de ajuste de transformações e separação dos dados.
- [Optuna 4.9 — Study](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.study.Study.html):
  continuação de estudos e novas tentativas por chamada.
- [Keras — EarlyStopping](https://keras.io/api/callbacks/early_stopping/).
- [KerasTuner — Hyperband](https://keras.io/keras_tuner/api/tuners/hyperband/).

Evidência local: `data/raw_data/{train,test,RUL}_FD00X.txt`,
`data/processed_data/{X,y}_{train,test}_FD00X.npz`,
`models/{optuna_logs,optuna_logs_transformer}/optuna_study_FD00X.db`,
`models/benchmark_seeds_{detalhado,resumo}.csv`, CSVs finais dos campeões,
configurações internas de seus `.keras`, metadados nominais do AE e seus CSVs
de detecção existentes. Os artefatos locais não foram modificados.
