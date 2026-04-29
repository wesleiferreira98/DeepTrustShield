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
