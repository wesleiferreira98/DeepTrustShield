# Notebooks de Audio

Esta pasta e reservada para experimentos de deteccao de deepfake de audio.

Use esta pasta para:

- treino exploratorio com FakeBrAccent-D;
- analise de falso positivo por sotaque;
- experimentos com log-Mel + CNN;
- comparacao futura com embeddings de wav2vec2/WavLM.

A pasta `notebooks/` na raiz do projeto fica reservada para imagem/face.

Fluxo recomendado para FakeBrAccent-D:

```bash
python scripts/build_fake_braccent_manifest.py \
  --dataset-root audio/data/raw/fake_braccent_d \
  --output audio/data/manifests/fake_braccent_d_manifest.csv

python scripts/train_audio_deepfake.py \
  --manifest audio/data/manifests/fake_braccent_d_manifest.csv \
  --output-dir audio/models/audio_deepfake_v1_fake_braccent_d \
  --epochs 20 \
  --batch-size 32
```
