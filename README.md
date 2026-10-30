# Turbofan Engine Degradation — Mestrado

## Etapa 2: avaliação da detecção de anomalias

A avaliação usa os resultados existentes do LSTM Autoencoder. Não treina modelos,
não executa Optuna e não modifica os resultados em `results/anomaly_detection/`.
Execute os comandos na raiz do repositório, no ambiente Python do projeto.

```powershell
# Todos os datasets.
python -m src.ml.evaluate_anomaly_detection --datasets FD001 FD002 FD003 FD004

# Somente um dataset.
python -m src.ml.evaluate_anomaly_detection --datasets FD001

# Repetir uma avaliação já existente (atualiza somente a nova pasta de avaliação).
python -m src.ml.evaluate_anomaly_detection --datasets FD001 FD002 FD003 FD004 --overwrite

# Testes sintéticos, sem carregar ou treinar redes.
python -m unittest discover -s tests -p "test_evaluate_anomaly_detection.py" -v
```

Requer os CSVs e `detection_config.json` em `results/anomaly_detection/FD00X/`.
Utiliza os dados brutos e os artefatos em `data/processed_anomaly/FD00X/` quando
disponíveis. Ausências de artefatos opcionais são registradas no relatório.
Sem dados brutos, o inventário fica limitado aos motores/períodos presentes nos CSVs.
Dados essenciais inconsistentes interrompem a execução com uma mensagem de erro.

### Saídas

Tudo é salvo em `results/anomaly_evaluation/`, separado dos resultados originais:

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
python -m src.ml.evaluate_anomaly_detection --datasets FD001 --recover-calibration-only
python -m src.ml.evaluate_anomaly_detection --datasets FD001 --overwrite
```

Essa opção apenas carrega o modelo salvo e reconstrói `X_val` nominal já salvo,
verificando hashes e o P99 original. Não ajusta modelo, normalizador ou KMeans,
não salva o modelo novamente e não inicia a avaliação na mesma execução.
Se a origem da calibração não puder ser confirmada, a análise principal mantém
somente o P99 original e registra a omissão dos percentis adicionais.
