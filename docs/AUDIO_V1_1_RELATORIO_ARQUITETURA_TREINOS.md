# Relatório da Arquitetura de Áudio V1.1

Este documento resume a situação atual da arquitetura neural de áudio, as estratégias de treino utilizadas, os conjuntos de dados empregados, o modo como o áudio é recebido pelo modelo e os próximos passos recomendados.

## 1. Arquitetura atual

A arquitetura neural atual é baseada em uma CNN leve treinada sobre representações log-Mel. O modelo não recebe o arquivo de áudio bruto como entrada direta para a rede neural. O arquivo pode chegar em formatos como `.ogg`, `.mp3`, `.wav`, `.m4a` ou similares, mas antes da inferência ele é decodificado, reamostrado e transformado em um tensor de características acústicas.

O fluxo atual é:

1. O arquivo de áudio é carregado pelo pipeline.
2. O áudio é convertido para mono quando necessário.
3. O áudio é reamostrado para 16 kHz.
4. O áudio é dividido em janelas de 4 segundos.
5. Cada janela é convertida em espectrograma Mel.
6. O espectrograma Mel é convertido para escala logarítmica.
7. Os valores são limitados entre -80 dB e 0 dB.
8. A representação final é normalizada para a faixa de 0 a 1.
9. O tensor log-Mel é enviado para a CNN.

Parâmetros principais da representação:

| Parâmetro            | Valor atual |
| --------------------- | ----------: |
| Taxa de amostragem    |      16 kHz |
| Janela neural         |  4 segundos |
| Número de bandas Mel |          80 |
| `n_fft`             |        1024 |
| `hop_length`        |         256 |
| `win_length`        |        1024 |
| Escala                |     log-Mel |

Na inferência, o áudio é analisado por janelas ao longo do tempo. Isso permite identificar se a suspeita está distribuída pelo áudio inteiro ou concentrada em trechos específicos.

## 2. Modelo neural

A arquitetura principal em uso é a opção `cnn`.

Ela possui:

- blocos convolucionais 2D;
- normalização por lote;
- ativações ReLU;
- camadas de `MaxPool2d`;
- `AdaptiveAvgPool2d`;
- classificador linear final com saída binária.

A saída do modelo é um logit que, após função sigmoide, vira uma probabilidade estimada de áudio fake.

Também foi criada uma opção `resnet_small`, mais robusta, com blocos residuais. Ela foi testada, mas até o momento a CNN leve segue mais adequada para os experimentos de V1.1, principalmente por ser mais simples de calibrar e interpretar nos testes atuais.

## 3. Estratégias de treino usadas

### Treino supervisionado por janelas

O treinamento supervisionado usa janelas de áudio de 4 segundos. Para áudios manipulados com marcação temporal, a janela pode ser alinhada ao trecho manipulado por meio dos campos `start` e `end` do manifesto.

Essa abordagem é importante porque nem todo áudio fake é fake por inteiro. Em muitos testes, a manipulação é localizada em uma faixa temporal específica.

### Agregação e thresholds

Foram calibrados dois tipos de limiar:

- limiar de decisão, usado para classificar o áudio como real, fake ou inconclusivo;
- limiar de atenção, usado para destacar janelas suspeitas mantendo uma revocação mínima para fake.

Nos treinos mais recentes foi usado:

```text
--attention-min-recall 0.90
```

Isso força o limiar de atenção a manter pelo menos 90% de recuperação dos exemplos fake, aceitando algum custo em falso positivo.

### Balanceamento e amostras de coleta

Foram feitos experimentos com:

- mistura de bases;
- manipulações controladas nos áudios coletados;
- repetição de amostras reais;
- aumento de peso para `coleta_real`;
- fatiamento de áudios reais longos em janelas;
- treino multitarefa com cabeçote auxiliar de domínio;
- treino adversarial de domínio com reversão de gradiente.

O objetivo desses testes foi reduzir o viés de domínio. Esse viés ocorre quando o modelo aprende características da base de dados, como limpeza do áudio, formalidade da fala, duração, compressão ou origem do áudio, em vez de aprender sinais reais de manipulação.

### Treino adversarial de domínio

O melhor avanço recente veio do treino com domínio acústico adversarial.

Nesse modo, o modelo recebe uma tarefa auxiliar de domínio, mas com reversão de gradiente. A ideia é dificultar que o backbone aprenda atalhos como:

```text
áudio de coleta = fake
áudio formal limpo = real ou fake conforme a base
áudio curto = padrão conhecido
```

O comando usado seguiu esta estrutura:

```bash
.venv-training/bin/python scripts/train_audio_deepfake.py \
  --manifest audio/data/manifests/audio_v1_1_mixed_coraa_v2_coleta77_manifest.csv \
  --output-dir audio/models/audio_deepfake_v1_1_mixed_coraa_v2_coleta77_domain_adv025_acoustic_cnn \
  --epochs 20 \
  --batch-size 32 \
  --architecture cnn \
  --device cuda \
  --amp \
  --attention-min-recall 0.90 \
  --domain-loss-weight 0.25 \
  --domain-label-mode acoustic \
  --domain-adversarial
```

Esse treino reduziu a capacidade do modelo de identificar o domínio, o que é desejável nesse caso. A acurácia do domínio caiu para uma faixa próxima de 35% a 44% durante a validação, indicando que o backbone ficou menos dependente de características específicas da base.

## 4. Conjuntos de dados usados

### 4.0. Manifestos

Cada conjunto de dados é representado por um arquivo CSV chamado manifesto. O manifesto é a única fonte de verdade que o pipeline de treino consome: ele não varre diretórios de áudio diretamente, apenas lê as linhas do CSV.

Isso foi implementado assim por três razões práticas:

1. **Mistura controlada de bases heterogêneas.** BRSpeech, FakeBrAccent, CORAA e coleta própria têm estruturas diferentes. O manifesto normaliza tudo em um esquema único antes do treino.
2. **Rastreabilidade por janela.** Como o modelo treina em janelas de 4 segundos, cada linha do manifesto pode representar um trecho de um arquivo maior, não o arquivo inteiro. Os campos `start` e `end` apontam o intervalo exato dentro do áudio original.
3. **Separação de splits por falante.** O campo `speaker_id` é usado para garantir que o mesmo falante não apareça ao mesmo tempo em `train` e `test`. Sem isso, o modelo poderia memorizar a voz em vez de aprender a detectar manipulação.

A estrutura base de cada linha do manifesto é:

| Campo | Descrição |
|---|---|
| `audio_path` | Caminho absoluto para o arquivo de áudio |
| `split` | Partição: `train`, `val` ou `test` |
| `label` | Rótulo da janela: `real` ou `fake` |
| `source` | Base de origem (ex.: `coleta_real`, `brspeech_df`) |
| `domain` | Agrupamento acústico usado no treino adversarial (ex.: `fala_espontanea`, `tts_formal_limpo`) |
| `speaker_id` | Identificador do falante para estratificação de splits |
| `start` / `end` | Início e fim em segundos da janela dentro do arquivo; vazio significa o arquivo inteiro |
| `file_label` | Rótulo do arquivo completo, pode diferir de `label` quando a manipulação é localizada |
| `manipulation_type` | Tipo de manipulação aplicada (`splice_other`, `silence`, `pitch`, etc.) ou `none` |
| `ground_truth_path` | Caminho para o áudio original antes da manipulação, quando disponível |

Os manifestos mistos são gerados pelo script `scripts/build_audio_v11_mixed_manifest.py`, que amostra cada base com cotas configuráveis, aplica a estratificação por falante e grava o CSV final.

### BRSpeech-DF

Base brasileira com exemplos bonafide e spoof. Foi usada para ampliar a quantidade de exemplos supervisionados de áudio real e fake.

Observação atual: o modelo tem bom desempenho dentro desse domínio, mas a base apresenta muitos áudios limpos e com fala formal. Isso pode favorecer bons resultados no próprio domínio, mas não representa totalmente o áudio espontâneo de celular.

### FakeBrAccent-D com equalização de codec

Base utilizada nos primeiros ciclos da V1, com foco em sotaques e variações de fala em português brasileiro. Também foi utilizada uma versão com tratamento de codec para reduzir atalhos simples de compressão.

A equalização é feita pelo script `scripts/equalize_audio_codec.py`. Para cada áudio do manifesto, o processo ocorre em duas etapas via ffmpeg:

1. O arquivo original é recodificado para MP3 mono 16 kHz a 96 kbps (`libmp3lame`).
2. O MP3 intermediário é decodificado de volta para WAV PCM 16-bit mono 16 kHz.

O objetivo não é melhorar a qualidade, mas impor os mesmos artefatos de compressão MP3 em todos os exemplos. Sem isso, o modelo pode aprender que áudio fake tende a ser mais limpo (gerado diretamente como WAV) enquanto áudio real tende a carregar ruído de compressão, usando esse atalho em vez de aprender sinais de manipulação.

Observação atual: segue útil como parte da mistura, mas não é suficiente para resolver generalização em áudio espontâneo.

### CORAA

Base de fala espontânea do português brasileiro. Foi usada como fonte de áudios reais mais próximos de fala natural e informal.

Observação atual: o CORAA evidenciou a sensibilidade de domínio dos modelos. Áudios reais espontâneos podem elevar a suspeita neural quando o modelo foi treinado majoritariamente em bases mais limpas ou formais.

### Coleta própria

Conjunto de áudios reais coletados com consentimento, em contexto mais próximo de celular e mensagens de voz. Esses áudios têm durações variadas, chegando a mais de 2 minutos em alguns casos.

Também foram geradas versões manipuladas controladas, incluindo:

- `splice_other`;
- silêncio;
- aumento de volume;
- redução de volume;
- alteração de tempo;
- alteração de pitch.

Observação atual: a coleta é o ponto mais importante para aproximar o modelo do uso real. Ao mesmo tempo, ela revelou um problema forte: o modelo ainda tende a marcar parte dos áudios reais de coleta como fake.

## 5. Resultados recentes

### 5.0. Métricas utilizadas

Dois indicadores principais aparecem nas tabelas de pontos críticos por fonte:

**FPR False Positive Rate (Taxa de Falso Positivo)**

Proporção de áudios reais classificados incorretamente como fake.

```
FPR = FP / (FP + TN)
```

No projeto, um FPR alto em uma fonte como `coleta_real` significa que o modelo está marcando áudios genuínos como suspeitos. Isso é o problema mais crítico atual, pois gera alertas sem fundamento em áudios legítimos.

**FNR  False Negative Rate (Taxa de Falso Negativo)**

Proporção de áudios fake que passaram sem ser detectados, ou seja, foram classificados como reais.

```
FNR = FN / (FN + TP)
```

No projeto, um FNR alto em uma fonte como `coleta_manipulated_strong` significa que manipulações reais escaparam da detecção.

**FPR de atenção**

O pipeline usa dois limiares distintos: um limiar de decisão, para a classificação final do áudio como real, fake ou inconclusivo, e um limiar de atenção, mais baixo, para marcar janelas suspeitas sem exigir certeza suficiente para a decisão final.

O limiar de atenção é calibrado pela função `find_attention_threshold` no script de treino. Ela varre todos os valores de probabilidade possíveis e escolhe o menor limiar que ainda garante pelo menos 90% de recuperação dos fakes (`--attention-min-recall 0.90`). Entre todos os limiares que atingem essa cobertura mínima, o algoritmo escolhe o que produz o menor FPR.

O **FPR de atenção** é o FPR resultante nesse limiar mais sensível. Como o limiar de atenção é intencionalmente mais baixo do que o de decisão, ele flageia mais janelas e, por consequência, tende a ter um FPR mais alto do que o limiar de decisão. Esse número indica o custo em falsos positivos de manter a cobertura mínima de 90% sobre fakes.

**Test F1 calibrado e Test Acc calibrado**

As métricas sem sufixo (Test F1, Test Acc) são calculadas com limiar fixo em 0,5. As versões calibradas usam um limiar ajustado, encontrado pela função `find_best_threshold` aplicada ao conjunto de validação: ela varre todos os valores de probabilidade gerados pelo modelo na validação e escolhe o limiar que maximiza o F1 nesse conjunto. Esse limiar é então aplicado ao conjunto de teste para gerar as métricas calibradas.

A diferença prática é que 0,5 é arbitrário. Se o modelo foi treinado com classes desbalanceadas ou produz probabilidades concentradas acima ou abaixo de 0,5, o limiar ótimo pode ser bem diferente. As métricas calibradas refletem o desempenho real quando o limiar é ajustado para o melhor ponto de operação encontrado nos dados de validação.

O objetivo ideal é manter FNR baixo (não deixar passar fakes) sem elevar demais o FPR (não acusar áudios reais). As tabelas abaixo mostram por fonte onde cada modelo ainda desequilibra esse tradeoff.

---

### Modelo misto com coleta 77

Checkpoint:

```text
audio_deepfake_v1_1_mixed_coraa_v2_coleta77_cnn
```

Resultado agregado:

| Métrica           |  Valor |
| ------------------ | -----: |
| Test AUC           | 0.9319 |
| Test F1            | 0.8323 |
| Test Acc           | 0.8330 |
| Test F1 calibrado  | 0.9070 |
| Test Acc calibrado | 0.8964 |
| FPR de atenção   | 0.1186 |

Ponto crítico:

| Fonte                         |    FPR |    FNR |
| ----------------------------- | -----: | -----: |
| `coleta_real`               | 0.9091 | 0.0000 |
| `coleta_manipulated_strong` | 0.0000 | 0.0455 |

Leitura: o modelo detecta bem a coleta manipulada, mas ainda marca muitos áudios reais de coleta como fake.

### Modelo adversarial com domínio acústico

Checkpoint:

```text
audio_deepfake_v1_1_mixed_coraa_v2_coleta77_domain_adv025_acoustic_cnn
```

Resultado agregado:

| Métrica           |  Valor |
| ------------------ | -----: |
| Test AUC           | 0.9586 |
| Test F1            | 0.8911 |
| Test Acc           | 0.8858 |
| Test F1 calibrado  | 0.9087 |
| Test Acc calibrado | 0.8985 |
| FPR de atenção   | 0.0722 |

Ponto crítico:

| Fonte                               |    FPR |    FNR |
| ----------------------------------- | -----: | -----: |
| `coleta_real`                     | 0.8182 | 0.0000 |
| `coleta_manipulated_strong`       | 0.0000 | 0.0341 |
| `coraa`                           | 0.1556 | 0.0000 |
| `fake_braccent_d_codec_equalized` | 0.0800 | 0.0938 |

Leitura: este é o melhor resultado geral até o momento. O treino adversarial melhorou a robustez agregada, mas ainda não resolveu o falso positivo em `coleta_real`.

### Modelo com janelas reais de coleta

Checkpoint:

```text
audio_deepfake_v1_1_mixed_coraa_v2_coleta77_real_windows_domain_adv025_cnn
```

Resultado agregado:

| Métrica           |  Valor |
| ------------------ | -----: |
| Test AUC           | 0.9373 |
| Test F1            | 0.8651 |
| Test Acc           | 0.8562 |
| Test F1 calibrado  | 0.8541 |
| Test Acc calibrado | 0.8288 |
| FPR de atenção   | 0.1649 |

Ponto crítico:

| Fonte                         |    FPR |    FNR |
| ----------------------------- | -----: | -----: |
| `coleta_real`               | 0.7273 | 0.0000 |
| `coleta_manipulated_strong` | 0.0000 | 0.1591 |

Leitura: fatiar a coleta real em janelas ajudou levemente a reduzir falso positivo em coleta real, mas piorou a detecção das versões manipuladas e aumentou a taxa geral de falso positivo. Isso indica que a duração era parte do problema, mas não a solução completa.

## 6. Situação atual

A arquitetura atual já é funcional e apresenta bons resultados agregados em cenário misto. O melhor checkpoint experimental recente é o adversarial com domínio acústico:

```text
audio_deepfake_v1_1_mixed_coraa_v2_coleta77_domain_adv025_acoustic_cnn
```

Ele oferece o melhor equilíbrio geral entre AUC, acurácia calibrada, F1 e falso positivo de atenção.

Porém, ainda existe uma limitação importante: a sensibilidade a domínio. O modelo ainda pode interpretar características de áudio real de celular, fala espontânea, duração longa ou compressão como sinais de suspeita.

O problema mais claro no momento é:

```text
coleta_real ainda apresenta falso positivo elevado.
```

Isso precisa ser corrigido antes de tratar o modelo neural como uma versão robusta para uso amplo.

## 7. Próximos passos

### 7.1. Criar hard negatives pareados

O próximo passo mais forte é criar hard negatives pareados.

Em vez de fatiar a coleta real inteira, para cada áudio fake manipulado devemos criar uma janela real correspondente no áudio original, com o mesmo `start` e `end` da manipulação.

Exemplo:

```text
fake: Bahia interior manipulado, 22s-26s
real: Bahia interior original,   22s-26s
```

Isso força o modelo a comparar:

```text
mesmo áudio;
mesmo trecho;
mesma duração;
mesmo contexto acústico;
com e sem manipulação.
```

Esse tipo de par é mais forte do que simplesmente adicionar janelas reais aleatórias, porque reduz atalhos de duração, domínio e conteúdo.

### 7.2. Avaliar por domínio antes de escolher o checkpoint final

Cada novo checkpoint deve ser avaliado por fonte:

- BRSpeech-DF;
- FakeBrAccent-D;
- CORAA;
- coleta real;
- coleta manipulada.

A métrica agregada não é suficiente. Um modelo pode ter AUC alta e ainda falhar gravemente em um domínio real de uso.

### 7.3. Ajustar calibração por modo de uso

Os limiares atuais são globais. No futuro, pode ser necessário separar:

- limiar de decisão;
- limiar de atenção;
- limiar para áudio curto;
- limiar para áudio longo;
- limiar para domínio fora do treino.

### 7.4. Manter a camada simbólica como contestadora

A camada neuro-simbólica não deve apenas acompanhar o score neural. Ela deve continuar atuando como revisão independente, especialmente em casos nos quais o modelo neural eleva a suspeita por domínio.

O papel ideal da camada simbólica é:

```text
confirmar ruptura quando houver evidência temporal e linguística;
contestar falso positivo quando a suspeita neural não tiver suporte independente;
informar baixa cobertura quando o áudio for curto ou pouco analisável.
```

## 8. Conclusão

A arquitetura de áudio V1.1 evoluiu de um detector neural simples para um pipeline mais robusto, com treino supervisionado por janelas, mistura de domínios, manipulações controladas, calibração de thresholds e treino adversarial de domínio.

O melhor resultado atual é promissor, mas ainda não resolve completamente a generalização para áudios reais de coleta. A próxima melhoria deve focar em hard negatives pareados, pois essa estratégia ataca diretamente o problema observado: distinguir manipulação localizada de variação natural dentro do mesmo áudio e no mesmo intervalo temporal.
