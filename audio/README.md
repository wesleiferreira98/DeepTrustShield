# Audio Workspace

Esta pasta agora eh a base da frente de deteccao de audio real/fake.

## Estrutura esperada

```text
audio/
  data/
    raw/
      asvspoof_df/
        train/
          real/
          fake/
        val/
          real/
          fake/
        test/
          real/
          fake/
    manifests/
  models/
  notebooks/
```

Os notebooks de audio devem ficar sempre em `audio/notebooks/`. A pasta
`notebooks/` na raiz fica reservada para experimentos de imagem/face.

## FakeBrAccent-D

Base Kaggle:

```text
https://www.kaggle.com/datasets/katarinaveljovic/fake-braccent-d
```

Esta e a base prioritaria para a frente regional em portugues brasileiro. Ela
deve ser usada para treinar o baseline BR e medir vies por sotaque.

### Download

O download via Kaggle requer o `kaggle.json` da sua conta.

```bash
pip install kaggle
mkdir -p ~/.kaggle
cp /caminho/para/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download \
  -d katarinaveljovic/fake-braccent-d \
  -p audio/data/raw/fake_braccent_d_download

unzip audio/data/raw/fake_braccent_d_download/fake-braccent-d.zip \
  -d audio/data/raw/fake_braccent_d
```

Se o arquivo ZIP for baixado manualmente pelo navegador, coloque-o em:

```text
audio/data/raw/fake_braccent_d_download/fake-braccent-d.zip
```

e rode apenas o `unzip`.

### Manifesto com sotaque

```bash
python scripts/build_fake_braccent_manifest.py \
  --dataset-root audio/data/raw/fake_braccent_d \
  --output audio/data/manifests/fake_braccent_d_manifest.csv
```

O manifesto inclui:

- `audio_path`
- `relative_path`
- `filename`
- `extension`
- `split`
- `label`
- `source`
- `accent`

### Treino

```bash
python scripts/train_audio_deepfake.py \
  --manifest audio/data/manifests/fake_braccent_d_manifest.csv \
  --output-dir audio/models/audio_deepfake_v1_fake_braccent_d \
  --epochs 20 \
  --batch-size 32
```

O `summary.json` do treino inclui metricas por sotaque:

- `val_metrics_by_accent`
- `test_metrics_by_accent`
- `false_positive_rate`
- `false_negative_rate`

## Sidecars de transcricao para a camada fonetico-fonologica

A camada neuro-simbolica fonetico-fonologica le arquivos auxiliares ao lado do
audio para extrair candidatos de eventos linguistico-foneticos. O caminho mais
seguro para pesquisa e gerar esses arquivos offline com `faster-whisper`.

Instale as dependencias de pesquisa:

```bash
.venv/bin/pip install -r requirements-training.txt
```

Para transcrever um unico audio:

```bash
.venv/bin/python scripts/build_audio_transcript_sidecars.py \
  --input caminho/para/audio.ogg \
  --model medium \
  --device cuda \
  --compute-type float16 \
  --language pt
```

Para transcrever uma pasta:

```bash
.venv/bin/python scripts/build_audio_transcript_sidecars.py \
  --input audio/data/eval_external \
  --recursive \
  --model medium \
  --device cuda \
  --compute-type float16 \
  --language pt
```

Para transcrever os audios de um manifesto:

```bash
.venv/bin/python scripts/build_audio_transcript_sidecars.py \
  --manifest audio/data/manifests/fake_braccent_d_codec_equalized_manifest.csv \
  --model medium \
  --device cuda \
  --compute-type float16 \
  --language pt
```

Por padrao, o script aplica um pre-tratamento leve somente para o ASR: mono,
16 kHz, remocao de DC, filtro passa-altas suave, reducao conservadora de ruido
estacionario, corte apenas de silencio nas bordas e normalizacao leve de volume.
Quando esse modo esta ativo e `--output-suffix` nao e informado, o sidecar e
salvo como `.preprocessed.transcript.json`. Para comparar com o audio bruto,
use `--no-preprocess-asr-audio`; nesse caso o sidecar padrao continua sendo
`.transcript.json`.

Cada audio gera um sidecar:

```text
audio.transcript.json
```

Esse arquivo inclui texto, palavras com tempo aproximado, segmentos do ASR e
metadados do modelo. O worker ja consegue ler esse sidecar e gerar candidatos
para a camada simbolica. Sem sidecar, a analise segue normalmente, mas a parte
fonetico-fonologica retorna cobertura insuficiente.

### ASR automático no fluxo web

O worker também pode gerar esse sidecar automaticamente durante a análise web
quando nenhum sidecar for encontrado.

Variáveis principais:

```text
AUDIO_ASR_ENABLED=true
AUDIO_ASR_MODEL=medium
AUDIO_ASR_LANGUAGE=pt
AUDIO_ASR_DEVICE=auto
AUDIO_ASR_COMPUTE_TYPE=default
AUDIO_ASR_PREPROCESS_ENABLED=true
```

Quando ativado, o resultado de áudio inclui uma seção `transcript` e o front
mostra a transcrição automática de forma formatada antes da revisão simbólica.
O pré-tratamento de ASR é aplicado apenas à entrada temporária do Whisper; o
detector neural e as heurísticas fonéticas continuam usando o áudio original.

Se `faster-whisper` não estiver instalado ou o ASR falhar, a análise neural
continua funcionando; apenas a camada fonético-fonológica fica sem cobertura.

## Caminho recomendado para ASVspoof 2019 LA sem lotar a maquina

Se voce tiver a base oficial ja extraida, monte primeiro uma amostra pequena:

```bash
python scripts/build_asvspoof2019_la_sample.py \
  --source-root /caminho/para/ASVspoof2019_LA \
  --output-root audio/data/raw/asvspoof2019_la_sample \
  --max-per-label 300 \
  --mode hardlink \
  --clean
```

Isso gera:

```text
audio/data/raw/asvspoof2019_la_sample/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
  manifest.csv
```

O modo `hardlink` eh o melhor para nao duplicar os audios no disco.

## 1. Gerar o manifesto

```bash
python scripts/build_audio_manifest.py \
  --dataset-root audio/data/raw/asvspoof2019_la_sample \
  --output audio/data/manifests/asvspoof2019_la_sample_manifest.csv
```

O manifesto gerado inclui:

- `audio_path`
- `relative_path`
- `filename`
- `extension`
- `split`
- `label`
- `source`

## 2. Treinar o baseline fora do notebook

```bash
python scripts/train_audio_deepfake.py \
  --manifest audio/data/manifests/asvspoof2019_la_sample_manifest.csv \
  --output-dir audio/models/audio_deepfake_v1_asvspoof2019_sample \
  --epochs 10 \
  --batch-size 32
```

Saidas principais:

- `audio/models/audio_deepfake_v1_asvspoof2019_sample/audio_deepfake_v1_best.pth`
- `audio/models/audio_deepfake_v1_asvspoof2019_sample/history.csv`
- `audio/models/audio_deepfake_v1_asvspoof2019_sample/summary.json`

## 3. Baseline atual

O baseline desta trilha foi desenhado para:

- audio mono
- 16 kHz
- janelas fixas de 4 segundos
- log-Mel spectrogram
- CNN leve em PyTorch

## 4. Notebook

Notebook inicial:

- `audio/notebooks/train_audio_deepfake_v1.ipynb`

Ele continua util para exploracao, mas o fluxo principal agora deve sair pelo script:

- `scripts/train_audio_deepfake.py`

## 5. Proximo passo recomendado

Depois que voce tiver um primeiro dataset montado e o baseline rodando, o proximo passo natural eh:

- medir AUC/F1 por fonte
- testar augmentations de audio
- avaliar modelos melhores que a CNN leve
- preparar a integracao com backend so depois que o baseline estiver minimamente calibrado
