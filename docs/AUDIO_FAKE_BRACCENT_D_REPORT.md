# Relatório Técnico: Camada Simbólica Regional e Treinamento com FakeBrAccent-D

## Resumo Executivo

Esta etapa iniciou a frente de detecção de deepfake de áudio em português
brasileiro no DeepTrustShield. O objetivo não foi apenas treinar um classificador
real/fake, mas criar uma arquitetura mais cautelosa, capaz de reconhecer que um
modelo neural pode confundir sotaque, regionalismo, compressão ou domínio fora do
treino com sinais de áudio sintético.

Foram implementados:

- camada simbólica regional para revisão de decisões do modelo neural;
- inferência por janelas em áudios longos;
- manifesto da base FakeBrAccent-D com preservação de sotaque e falante;
- treino com métricas por sotaque;
- validação cruzada estratificada por `label + accent` e agrupada por
  `speaker_id`;
- normalização de áudio;
- equalização de codec;
- controle negativo com labels embaralhados.

O modelo considerado mais confiável nesta etapa é o treinado com a versão
`codec_equalized` da FakeBrAccent-D, pois reduz o principal vazamento encontrado:
a separação quase perfeita entre arquivos reais em `.wav` e falsos em `.mp3`.

## Motivação

A proposta da camada simbólica nasceu da preocupação com regionalismo. Em áudio,
especialmente em português brasileiro, sotaques, ritmo, entonação, pausas,
compressão de aplicativos e condições de gravação podem gerar falsos positivos.

Por isso, a camada simbólica foi reformulada para atuar como uma camada de
hipóteses concorrentes:

```text
modelo neural suspeita de fake
  -> camada simbólica pergunta:
     isso é fake?
     ou pode ser regionalismo?
     ou domínio fora do treino?
     ou compressão/canal?
     ou evidência insuficiente?
```

## Camada Simbólica Regional

Arquivo principal:

```text
worker/app/pipeline/audio_symbolic.py
```

A primeira versão tratava qualidade e incerteza:

- áudio curto;
- muito silêncio;
- volume baixo;
- score perto do threshold;
- análise parcial de áudio longo.

Depois, a camada foi refatorada para refletir o objetivo original: regionalismo
como hipótese alternativa ao fake.

### Flags simbólicas principais

| Flag | Função |
|------|--------|
| `strong_fake_score` | Modelo neural marcou fake com margem alta |
| `near_threshold` | Score próximo do limite calibrado |
| `partial_audio_analysis` | Áudio longo analisado parcialmente |
| `segment_score_disagreement` | Janelas divergentes entre si |
| `isolated_suspicious_segment` | Suspeita concentrada em poucos trechos |
| `consistent_suspicious_segments` | Maioria das janelas acima do threshold |
| `regional_bias_review` | Hipótese de falso positivo por regionalismo/domínio |
| `domain_mismatch_warning` | Formato/canal/duração parecem fora do domínio de treino |

### Inferência por janelas

Foi implementada análise por janelas no arquivo:

```text
worker/app/pipeline/audio_inference.py
```

O áudio é dividido em janelas de `clip_seconds`, com sobreposição de 50%. O
resultado inclui:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.0,
      "p_fake": 0.93
    }
  ],
  "segment_summary": {
    "count": 10,
    "coverage_fraction": 1.0
  }
}
```

Isso permitiu diferenciar:

- suspeita localizada;
- suspeita consistente;
- análise parcial;
- score global enganoso.

## Primeiro Problema Encontrado: Falso Positivo em Áudio Real

Um áudio real `.ogg` foi analisado pelo modelo anterior baseado em ASVspoof. O
modelo marcou praticamente todas as janelas como fake:

```text
10/10 janelas suspeitas
score médio por segmento: 0.9415
threshold ASVspoof: 0.2501
```

Sem a camada simbólica, a decisão seria `fake/high`. Após a revisão simbólica, a
decisão passou a sinalizar:

```text
regional_bias_review = true
domain_mismatch_warning = true
risk_level = medium
confidence = moderate
```

Isso mostrou que a camada simbólica não corrige um modelo enviesado sozinha, mas
impede que o sistema entregue uma conclusão excessivamente confiante quando há
evidência de domínio fora do treino.

## Base FakeBrAccent-D

Base utilizada:

```text
https://www.kaggle.com/datasets/katarinaveljovic/fake-braccent-d
```

Ela foi baixada manualmente e extraída em:

```text
audio/data/raw/fake_braccent_d
```

O manifesto inicial foi gerado com:

```text
scripts/build_fake_braccent_manifest.py
```

O manifesto preserva:

- `label`;
- `accent`;
- `speaker_id`;
- `source`;
- `split`.

Contagem geral:

```text
total: 1535
real: 1172
fake: 363
falantes: 107
```

Distribuição por sotaque:

```text
baiano: 262
carioca: 147
fluminense: 194
nordestino: 237
sulista: 695
```

## Problemas Encontrados na Base

### 1. Vazamento por extensão e codec

Foi encontrada uma separação forte entre classe e formato original:

```text
real: 1172 arquivos .wav
fake: 361 arquivos .mp3 e 2 arquivos .wav
```

Isso é um problema grave. Um modelo pode aprender:

```text
.wav = real
.mp3 = fake
```

em vez de aprender características de síntese de voz.

### 2. Artefato herdado de compressão

Converter todos os arquivos para `.wav` não basta. Se os fakes eram `.mp3`, eles
podem continuar carregando assinatura de compressão mesmo depois de decodificados
para WAV.

Por isso foram criadas duas versões:

```text
fake_braccent_d_normalized
fake_braccent_d_codec_equalized
```

### 3. Diferença de duração

Foi encontrada diferença média de duração:

```text
fake: média 7.94s
real: média 12.51s
```

O treino usa janelas de 4 segundos, o que reduz esse risco, mas a diferença ainda
pode afetar pausas, ritmo, conteúdo e distribuição temporal.

### 4. Desbalanceamento

A base é desbalanceada:

```text
real: 1172
fake: 363
```

Para compensar, o treino usa `pos_weight` automático:

```text
pos_weight = real_count / fake_count
```

No treino final:

```text
pos_weight = 3.4675
```

### 5. Poucos exemplos em alguns grupos

Alguns sotaques têm poucos exemplos, especialmente quando separados por
treino/validação/teste. Isso torna métricas por sotaque instáveis em alguns
casos, como `carioca` no teste.

## Medidas de Correção

### Manifesto com falante e sotaque

Script:

```text
scripts/build_fake_braccent_manifest.py
```

Inclui:

```csv
audio_path,relative_path,filename,extension,split,label,source,accent,speaker_id
```

### Normalização simples

Script:

```text
scripts/normalize_audio_dataset.py
```

Converte todos os áudios para:

```text
WAV
mono
16 kHz
```

Manifesto gerado:

```text
audio/data/manifests/fake_braccent_d_normalized_manifest.csv
```

### Equalização de codec

Script:

```text
scripts/equalize_audio_codec.py
```

Todos os arquivos passam pelo mesmo funil:

```text
áudio original
  -> MP3 96k mono 16 kHz
  -> WAV PCM 16 kHz
```

Manifesto gerado:

```text
audio/data/manifests/fake_braccent_d_codec_equalized_manifest.csv
```

Essa versão é considerada mais confiável porque reduz o vazamento de extensão e
também reduz a diferença de tratamento de codec entre reais e falsos.

## Validação Cruzada

Foi criado:

```text
scripts/train_audio_deepfake_cv.py
```

Estratégia:

```text
StratifiedGroupKFold
estrato = label + accent
grupo = speaker_id
```

Isso evita que o mesmo falante apareça em treino e validação no mesmo fold.

## Resultados Suspeitos

### Base bruta

A base bruta gerou métricas quase perfeitas:

```text
mean_best_val_auc: 0.9996
mean_val_f1_cal: 0.9934
mean_val_acc_cal: 0.9968
```

Esse resultado foi considerado suspeito por causa da associação entre classe e
extensão:

```text
real = .wav
fake = .mp3
```

### Base normalizada para WAV

Mesmo após converter tudo para WAV, as métricas continuaram muito altas:

```text
mean_best_val_auc: 0.9994
mean_val_f1_cal: 0.9851
mean_val_acc_cal: 0.9929
```

Isso indicou que a compressão MP3 original dos fakes poderia permanecer como
artefato no sinal.

### Base com codec equalizado

Após equalização de codec, o resultado continuou alto, mas é metodologicamente
mais confiável:

```text
mean_best_val_auc: 0.9987
mean_val_f1_cal: 0.9838
mean_val_acc_cal: 0.9922
```

Interpretação: o resultado ainda pode refletir artefatos do pipeline de geração,
mas não parece ser causado apenas por extensão ou por um bug simples no pipeline
de validação.

## Controle Negativo: Labels Embaralhados

Foi implementado controle negativo com:

```text
--shuffle-labels
```

Com labels embaralhados, o esperado é AUC próximo de 0.5.

Resultado:

```text
mean_best_val_auc: 0.4924
```

Por fold:

```text
fold 0: 0.4965
fold 1: 0.4661
fold 2: 0.4975
fold 3: 0.5499
fold 4: 0.4519
```

Conclusão: não há indício de vazamento estrutural óbvio no código de validação
cruzada. Se houvesse vazamento grave por caminho, split, grupo ou leitura de
label, o AUC continuaria alto mesmo com labels embaralhados.

## Modelo Mais Confiável

O modelo mais confiável desta etapa é:

```text
audio/models/audio_deepfake_v1_fake_braccent_d_codec_equalized
```

Manifesto usado:

```text
audio/data/manifests/fake_braccent_d_codec_equalized_manifest.csv
```

Motivos:

1. usa a versão com codec equalizado;
2. reduz o vazamento de extensão `.wav` versus `.mp3`;
3. usa áudio mono 16 kHz;
4. usa peso de classe para compensar desbalanceamento;
5. foi treinado em GPU com o mesmo pipeline usado no CV;
6. possui threshold calibrado em validação;
7. foi interpretado à luz do controle negativo com labels embaralhados.

### Resultado do modelo escolhido

Configuração:

```text
device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
AMP: false
epochs: 20
batch_size: 32
clip_seconds: 4.0
n_mels: 80
pos_weight: 3.4675
```

Resultado geral:

```text
best_epoch: 17
best_val_auc: 0.9997
threshold calibrado: 0.9744
test_auc: 0.9976
test_f1 bruto: 0.7529
test_acc bruto: 0.8091
test_f1 calibrado: 0.9516
test_acc calibrado: 0.9727
```

O threshold calibrado alto:

```text
0.9744
```

torna o modelo mais conservador. Isso é desejável para reduzir falsos positivos,
principalmente em áudios reais fora do domínio da base.

### Métricas por sotaque no teste

| Sotaque | AUC | F1 | Acurácia | Falso positivo | Falso negativo |
|---------|-----|----|----------|----------------|----------------|
| Baiano | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0% |
| Carioca | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0% |
| Fluminense | 1.0000 | 0.8649 | 0.8387 | 0.0% | 23.8% |
| Nordestino | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0% |
| Sulista | 1.0000 | 0.9474 | 0.9900 | 1.1% | 0.0% |

Observação: o grupo `carioca` no teste é pequeno, então o resultado perfeito não
deve ser interpretado como prova robusta.

## Como Usar o Modelo no Worker

Copiar pesos e resumo:

```bash
mkdir -p model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized

cp audio/models/audio_deepfake_v1_fake_braccent_d_codec_equalized/audio_deepfake_v1_best.pth \
  model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/audio_deepfake_v1_best.pth

cp audio/models/audio_deepfake_v1_fake_braccent_d_codec_equalized/summary.json \
  model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/summary.json
```

Configurar ambiente:

```bash
AUDIO_MODEL_PATH=model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/audio_deepfake_v1_best.pth
AUDIO_MODEL_SUMMARY_PATH=model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/summary.json
```

## Recomendações para Futuras Bases

### 1. Não separar classe por formato

Evitar:

```text
real em WAV
fake em MP3
```

Todos os arquivos devem passar pelo mesmo pipeline de exportação.

### 2. Padronizar codec antes da publicação

Publicar uma versão canônica:

```text
WAV PCM
mono
16 kHz ou 24 kHz
mesmo nível de normalização
```

Se houver versão comprimida, comprimir reais e falsos da mesma forma.

### 3. Controlar falante

O mesmo falante não deve aparecer em treino e teste. Separar por `speaker_id`.

### 4. Controlar frase

Quando possível, evitar que a mesma frase apareça em treino e teste em versões
real/fake muito parecidas. Isso reduz memorização textual/prosódica.

### 5. Registrar metadados

Uma boa base deve trazer:

- falante anonimizado;
- sotaque/região;
- gênero, se aplicável;
- frase;
- origem;
- ferramenta de geração;
- codec original;
- sample rate;
- duração;
- split oficial.

### 6. Publicar baselines e controles

Além do dataset, publicar:

- baseline neural;
- baseline simples com features superficiais;
- validação cruzada por falante;
- controle com labels embaralhados;
- métricas por sotaque;
- matriz de confusão por grupo.

### 7. Testar contra áudio externo

Mesmo bons resultados internos não bastam. É necessário testar em:

- áudios reais gravados por celular;
- áudios de WhatsApp/OGG;
- ambientes ruidosos;
- diferentes sotaques não vistos;
- fala espontânea, não apenas frases lidas.

## Conclusão

A etapa mostrou que a FakeBrAccent-D é útil, mas não deve ser usada crua. A base
tem riscos claros de artefato por codec, duração e pipeline de geração. Após
normalização, equalização de codec, validação cruzada por falante e controle com
labels embaralhados, o resultado continua forte.

O modelo final mais confiável é o treinado na versão `codec_equalized`, com
threshold calibrado de `0.9744`. Ele deve ser tratado como um baseline PT-BR
promissor, não como detector robusto definitivo.

O próximo passo científico é avaliar generalização fora da FakeBrAccent-D,
especialmente em áudios reais brasileiros gravados em celular e aplicativos de
mensagem. É nesse ponto que a camada simbólica regional deve ganhar mais peso,
rebaixando decisões quando o áudio parecer fora do domínio conhecido.
