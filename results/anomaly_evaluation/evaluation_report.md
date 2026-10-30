# Avaliação quantitativa da detecção de anomalias

Avaliação descritiva dos erros existentes. Nenhum modelo foi treinado, nenhum estudo Optuna foi executado e nenhuma configuração foi selecionada pelos resultados de teste. O baseline continua sendo P99 e 3 janelas consecutivas.

## Cobertura e antecedência do alerta

Cobertura não é acurácia, recall ou sensibilidade real de falhas: não há anotação do início da anomalia. O denominador é o número de motores com pelo menos k janelas consecutivas disponíveis. Motores insuficientes são contabilizados separadamente; a ausência de alerta é NaN, nunca ciclo zero. Quartis e mediana incluem apenas motores detectados.

| Dataset | Split | eligible_motors | insufficient_motors | detected_motors | coverage_pct | alert_rul_median | alert_rul_q25 | alert_rul_q75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | test | 99 | 1 | 13 | 13.13 | 31.00 | 21.00 | 49.00 |
| FD001 | train | 100 | 0 | 100 | 100.00 | 25.00 | 11.00 | 36.00 |
| FD002 | test | 252 | 7 | 79 | 31.35 | 40.00 | 28.00 | 83.00 |
| FD002 | train | 260 | 0 | 260 | 100.00 | 29.00 | 18.00 | 46.00 |
| FD003 | test | 100 | 0 | 31 | 31.00 | 86.00 | 56.00 | 175.50 |
| FD003 | train | 100 | 0 | 93 | 93.00 | 39.00 | 15.00 | 69.00 |
| FD004 | test | 237 | 11 | 55 | 23.21 | 34.00 | 19.00 | 64.50 |
| FD004 | train | 249 | 0 | 233 | 93.57 | 25.00 | 17.00 | 38.00 |

No treino completo, o último ciclo observado é a falha (RUL final zero). No teste truncado, RUL no alerta = RUL final fornecido + último ciclo observado − ciclo do alerta. Isso mede distância até a falha, não distância até o início verdadeiro da degradação.

### Motores de teste próximos da falha

Os limites de RUL são cumulativos. No treino, todas as faixas coincidem porque o RUL final é zero.

| Dataset | RULGroup | eligible_motors | detected_motors | coverage_pct | alert_rul_median |
| --- | --- | --- | --- | --- | --- |
| FD001 | RUL <= 30 | 25 | 11 | 44.00 | 29.00 |
| FD002 | RUL <= 30 | 61 | 56 | 91.80 | 32.00 |
| FD003 | RUL <= 30 | 20 | 15 | 75.00 | 61.00 |
| FD004 | RUL <= 30 | 53 | 39 | 73.58 | 23.00 |

### Leitura dos resultados por dataset

- **FD001:** alerta em 100/100 motores completos de treino (100.00%), com RUL mediano de 25.0 ciclos. No teste truncado, 13/99 motores elegíveis (13.13%), com RUL mediano de 31.0 ciclos; 1 motores sem janelas suficientes. Entre os motores com RUL final ≤30, a cobertura é 44.00% (11/25). Há 2 motores com alerta na região nominal assumida (2.00%).

- **FD002:** alerta em 260/260 motores completos de treino (100.00%), com RUL mediano de 29.0 ciclos. No teste truncado, 79/252 motores elegíveis (31.35%), com RUL mediano de 40.0 ciclos; 7 motores sem janelas suficientes. Entre os motores com RUL final ≤30, a cobertura é 91.80% (56/61). Há 4 motores com alerta na região nominal assumida (1.54%).

- **FD003:** alerta em 93/100 motores completos de treino (93.00%), com RUL mediano de 39.0 ciclos. No teste truncado, 31/100 motores elegíveis (31.00%), com RUL mediano de 86.0 ciclos; 0 motores sem janelas suficientes. Entre os motores com RUL final ≤30, a cobertura é 75.00% (15/20). Há 7 motores com alerta na região nominal assumida (7.00%).

- **FD004:** alerta em 233/249 motores completos de treino (93.57%), com RUL mediano de 25.0 ciclos. No teste truncado, 55/237 motores elegíveis (23.21%), com RUL mediano de 34.0 ciclos; 11 motores sem janelas suficientes. Entre os motores com RUL final ≤30, a cobertura é 73.58% (39/53). Há 3 motores com alerta na região nominal assumida (1.20%).

A cobertura maior no grupo próximo da falha, quando comparada ao teste completo, é compatível com um score sensível à degradação, mas a comparação entre grupos não demonstra crescimento monotônico dentro de cada motor. A cobertura global menor no teste também reflete o truncamento das trajetórias e não deve ser interpretada isoladamente como erro do detector. Alertas muito antecipados podem refletir heterogeneidade entre motores ou desvios transitórios, não necessariamente maior qualidade. Os gráficos individuais e as taxas nominais precisam ser considerados em conjunto.

## Alertas na região nominal assumida

Somente janelas inteiramente dentro dos primeiros 30% da trajetória completa de treino. O teste não recebe essa regra. A elegibilidade nominal exige ao menos uma janela completa; as tabelas também informam motores sem ciclos/janelas suficientes. A divisão fit_train/validation é recuperada dos artefatos, não inferida a partir dos erros.

| Dataset | DevelopmentRole | eligible_motors | insufficient_nominal_cycles | motors_with_nominal_alert | nominal_motor_alert_pct | nominal_above_windows_pct | nominal_persistent_windows_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | all_train | 100 | 0 | 2 | 2.00 | 0.52 | 0.25 |
| FD001 | fit_train | 80 | 0 | 1 | 1.25 | 0.38 | 0.23 |
| FD001 | validation | 20 | 0 | 1 | 5.00 | 1.11 | 0.32 |
| FD002 | all_train | 260 | 0 | 4 | 1.54 | 0.68 | 0.41 |
| FD002 | fit_train | 208 | 0 | 3 | 1.44 | 0.61 | 0.32 |
| FD002 | validation | 52 | 0 | 1 | 1.92 | 1.03 | 0.83 |
| FD003 | all_train | 100 | 0 | 7 | 7.00 | 5.08 | 4.32 |
| FD003 | fit_train | 80 | 0 | 6 | 7.50 | 5.97 | 5.10 |
| FD003 | validation | 20 | 0 | 1 | 5.00 | 1.10 | 0.85 |
| FD004 | all_train | 249 | 0 | 3 | 1.20 | 0.22 | 0.12 |
| FD004 | fit_train | 199 | 0 | 0 | 0.00 | 0.02 | 0.00 |
| FD004 | validation | 50 | 0 | 3 | 6.00 | 1.04 | 0.61 |

Alertas nessa região são candidatos a alertas prematuros sob a hipótese operacional de normalidade, não falsos positivos comprovados. fit_train é diagnóstico in-sample; validation também já participou da seleção de hiperparâmetros, EarlyStopping e definição do threshold. Nenhuma dessas taxas é uma estimativa de generalização em calibração independente.

### Motores com alerta nominal no baseline

- FD001: motores de treino 31, 33.

- FD002: motores de treino 44, 109, 124, 174.

- FD003: motores de treino 10, 17, 18, 24, 62, 71, 89.

- FD004: motores de treino 19, 34, 247.

No FD003, 5.97% das janelas nominais de ajuste ultrapassam o threshold, contra 1.10% na validação usada para calibrá-lo. Isso merece investigar representatividade dos motores e adequação da hipótese nominal. Não é, por si só, evidência de erro no cálculo ou vazamento; o P99 foi reproduzido e a divisão por motor conferida.

## Persistência: 3 versus 5 janelas

| Dataset | Split | detected_3 | detected_5 | alerts_disappearing | alerts_later | median_delay_cycles_among_later |
| --- | --- | --- | --- | --- | --- | --- |
| FD001 | test | 13 | 11 | 2 | 11 | 2.00 |
| FD001 | train | 100 | 100 | 0 | 100 | 2.00 |
| FD002 | test | 79 | 75 | 4 | 75 | 2.00 |
| FD002 | train | 260 | 260 | 0 | 260 | 2.00 |
| FD003 | test | 31 | 30 | 1 | 30 | 2.00 |
| FD003 | train | 93 | 86 | 7 | 86 | 2.00 |
| FD004 | test | 55 | 43 | 12 | 43 | 2.00 |
| FD004 | train | 233 | 225 | 8 | 225 | 2.00 |

Aumentar a persistência pode suprimir episódios curtos ou adiar o primeiro alerta. As janelas de 30 ciclos se sobrepõem em 29 ciclos; não são observações independentes. A tabela evaluation_summary compara cobertura e RUL mediano, e nominal_alerts compara as taxas nominais para ambas as regras. O denominador elegível pode mudar com k. Não há escolha automática de uma regra vencedora.

## Sensibilidade ao threshold

Os percentis adicionais usam exclusivamente os erros das janelas nominais de validação identificadas pelos metadados salvos. O cruzamento com o CSV é validado por hashes, separação de motores e reprodução do P99 original (tolerância relativa 1e-5). O valor original de P99 é preservado exatamente. Não se calculam percentis no teste nem no conjunto completo de treino.

- FD001: origem `existing_train_CSV_joined_to_validation_nominal_metadata`; percentis disponíveis: 95.0, 97.0, 99.0, 99.5.

- FD002: origem `existing_train_CSV_joined_to_validation_nominal_metadata`; percentis disponíveis: 95.0, 97.0, 99.0, 99.5.

- FD003: origem `existing_train_CSV_joined_to_validation_nominal_metadata`; percentis disponíveis: 95.0, 97.0, 99.0, 99.5.

- FD004: origem `existing_train_CSV_joined_to_validation_nominal_metadata`; percentis disponíveis: 95.0, 97.0, 99.0, 99.5.

A sensibilidade é descritiva. Reutilizar a validação do desenvolvimento não a torna uma calibração independente.

| Dataset | Percentile | Persistence | detected_motors | eligible_motors | coverage_pct | nominal_motor_alert_pct |
| --- | --- | --- | --- | --- | --- | --- |
| FD001 | 95.00 | 3 | 14 | 99 | 14.14 | 4.00 |
| FD001 | 95.00 | 5 | 13 | 99 | 13.13 | 3.00 |
| FD001 | 97.00 | 3 | 14 | 99 | 14.14 | 4.00 |
| FD001 | 97.00 | 5 | 13 | 99 | 13.13 | 2.00 |
| FD001 | 99.00 | 3 | 13 | 99 | 13.13 | 2.00 |
| FD001 | 99.00 | 5 | 11 | 99 | 11.11 | 1.00 |
| FD001 | 99.50 | 3 | 13 | 99 | 13.13 | 2.00 |
| FD001 | 99.50 | 5 | 11 | 99 | 11.11 | 1.00 |
| FD002 | 95.00 | 3 | 113 | 252 | 44.84 | 13.85 |
| FD002 | 95.00 | 5 | 106 | 251 | 42.23 | 11.92 |
| FD002 | 97.00 | 3 | 102 | 252 | 40.48 | 10.38 |
| FD002 | 97.00 | 5 | 95 | 251 | 37.85 | 6.92 |
| FD002 | 99.00 | 3 | 79 | 252 | 31.35 | 1.54 |
| FD002 | 99.00 | 5 | 75 | 251 | 29.88 | 1.15 |
| FD002 | 99.50 | 3 | 72 | 252 | 28.57 | 0.77 |
| FD002 | 99.50 | 5 | 65 | 251 | 25.90 | 0.38 |
| FD003 | 95.00 | 3 | 32 | 100 | 32.00 | 8.00 |
| FD003 | 95.00 | 5 | 31 | 100 | 31.00 | 8.00 |
| FD003 | 97.00 | 3 | 31 | 100 | 31.00 | 8.00 |
| FD003 | 97.00 | 5 | 31 | 100 | 31.00 | 7.00 |
| FD003 | 99.00 | 3 | 31 | 100 | 31.00 | 7.00 |
| FD003 | 99.00 | 5 | 30 | 100 | 30.00 | 7.00 |
| FD003 | 99.50 | 3 | 30 | 100 | 30.00 | 7.00 |
| FD003 | 99.50 | 5 | 29 | 100 | 29.00 | 7.00 |
| FD004 | 95.00 | 3 | 82 | 237 | 34.60 | 6.02 |
| FD004 | 95.00 | 5 | 70 | 237 | 29.54 | 3.61 |
| FD004 | 97.00 | 3 | 71 | 237 | 29.96 | 3.21 |
| FD004 | 97.00 | 5 | 59 | 237 | 24.89 | 1.20 |
| FD004 | 99.00 | 3 | 55 | 237 | 23.21 | 1.20 |
| FD004 | 99.00 | 5 | 43 | 237 | 18.14 | 0.80 |
| FD004 | 99.50 | 3 | 40 | 237 | 16.88 | 0.80 |
| FD004 | 99.50 | 5 | 27 | 237 | 11.39 | 0.40 |

A última coluna acima sempre se refere à região nominal do treino completo, não ao teste.

## Regimes operacionais e FD004, motor 1 de teste

### FD002

| Split | Group | windows | mean_error | median_error | above_threshold_pct | persistent_window_pct | mean_error_change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | False | 152 | 0.035199 | 0.034650 | 3.947368 | 3.947368 | -0.000268 |
| test | True | 26353 | 0.036410 | 0.035335 | 7.983911 | 6.830342 | 0.000090 |
| train | False | 319 | 0.039327 | 0.035193 | 18.808777 | 16.927900 | -0.000102 |
| train | True | 45900 | 0.041735 | 0.036224 | 20.799564 | 19.021786 | 0.000403 |

test: 26505/26505 janelas contêm transição interna (100.00%).

train: 46219/46219 janelas contêm transição interna (100.00%).

### FD004

| Split | Group | windows | mean_error | median_error | above_threshold_pct | persistent_window_pct | mean_error_change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | False | 216 | 0.018748 | 0.018033 | 2.314815 | 1.851852 | -0.000217 |
| test | True | 33865 | 0.019325 | 0.018381 | 2.846597 | 2.158571 | 0.000038 |
| train | False | 344 | 0.020852 | 0.018313 | 10.465116 | 9.302326 | -0.000112 |
| train | True | 53684 | 0.023076 | 0.019130 | 13.477014 | 11.742791 | 0.000237 |

test: 34081/34081 janelas contêm transição interna (100.00%).

train: 54028/54028 janelas contêm transição interna (100.00%).

FD004, motor 1 do teste, mantendo P99:

| Persistence | detection_cycle | alert_RUL | last_observed_cycle | last_RUL |
| --- | --- | --- | --- | --- |
| 3 | 101.00 | 151.00 | 230 | 22.00 |
| 5 | N/A | N/A | 230 | 22.00 |

No ciclo 101, o erro é 0.028780, acima do threshold: True; alerta persistente: True; comprimento total do episódio acima do threshold: 3 janelas. O recorte de ciclos 85–120 está salvo em CSV para inspeção.

O episódio termina no ciclo 101 e tem exatamente o mínimo exigido pela regra de 3. Portanto, o alerta é persistente segundo a regra implementada, mas o episódio é curto e não constitui evidência de degradação sustentada. Sua supressão por 5 janelas ilustra a sensibilidade à regra, sem provar que 5 seja a escolha ideal.

O regime é estimado pelo KMeans já salvo, sem ajuste novo. Regime da janela significa regime do último ciclo. Transição interna exige mudança após o primeiro ciclo e até o último; transição próxima ao final ocorre nos três ciclos finais. mean_error_change é a diferença de MSE em relação à janela anterior consecutiva. O perfil −3 a +3 ciclos pode contar janelas em vários eventos sobrepostos. Associações são descritivas, sem controle por motor, RUL ou dependência temporal: não comprovam causalidade. Quando quase todas as janelas contêm transições, o contraste transição versus ausência é limitado ou inexistente.

## Auditoria, limitações e próxima etapa

- FD001: 0 divergências na reprodução dos indicadores originais.

- CSV foi ordenado temporalmente para avaliação; arquivo original preservado.

- Análise de regimes não aplicável: dataset de condição operacional única.

- FD002: 0 divergências na reprodução dos indicadores originais.

- CSV foi ordenado temporalmente para avaliação; arquivo original preservado.

- test: transition_inside sem grupo de contraste; associação não comparável.

- train: transition_inside sem grupo de contraste; associação não comparável.

- FD003: 0 divergências na reprodução dos indicadores originais.

- CSV foi ordenado temporalmente para avaliação; arquivo original preservado.

- Análise de regimes não aplicável: dataset de condição operacional única.

- FD004: 0 divergências na reprodução dos indicadores originais.

- CSV foi ordenado temporalmente para avaliação; arquivo original preservado.

- test: transition_inside sem grupo de contraste; associação não comparável.

- train: transition_inside sem grupo de contraste; associação não comparável.

Não houve modificação dos resultados originais, dos modelos ou do pré-processamento. Não há retreinamento nem integração ao Transformer nesta etapa. A investigação de regimes não se aplica a FD001/FD003.

A hipótese nominal de 30% não é verdade-terreno. O erro agrega features normalizadas, inclusive condições operacionais quando presentes, e não corresponde apenas ao erro físico de sensores. Comparar magnitude bruta do MSE entre datasets não é uma escala universal de severidade. Os motores representativos dos gráficos são 1, 2 e 3 em cada split; as métricas usam todos os motores disponíveis.

**Recomendação:** os resultados permitem investigar o score como candidato a variável adicional, mas não comprovam ainda detecção do início da degradação ou valor preditivo incremental para RUL. Antes de afirmar validação independente, fazer uma segunda rodada com divisão por motor em ajuste, validação de desenvolvimento e calibração separada. O grupo de calibração deve ficar fora tanto do ajuste quanto da escolha de hiperparâmetros/EarlyStopping. Fixar threshold e persistência antes de abrir uma avaliação final independente; como este teste já foi examinado, tratar futuras escolhas informadas por ele como exploratórias. Se os motores disponíveis já influenciaram a seleção atual, apenas renomear a validação não resolve: a separação exige novo protocolo de desenvolvimento.

Em uma etapa futura, a utilidade para o Transformer exigirá uma comparação controlada com/sem o score, preservando a separação por motor e o mesmo protocolo experimental. Esta comparação não foi executada.

## Reprodutibilidade

`python -m src.ml.evaluate_anomaly_detection --datasets FD001 FD002 FD003 FD004`

Para atualizar somente esta avaliação, acrescentar `--overwrite`. Todas as figuras são PNG a 300 dpi; os CSVs e evaluation_config.json documentam configurações, denominadores, hashes e omissões. O arquivo comparison/evaluation_summary.csv reúne o baseline e a persistência 5.
