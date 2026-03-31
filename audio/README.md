# Audio Workspace

This directory contains the base structure for the audio fraud detection track.

## Layout

```text
audio/
  data/
    raw/
    manifests/
  models/
  notebooks/
  research/
```

## Expected dataset shape

The first baseline assumes a directory like:

```text
audio/data/raw/asvspoof_df/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

## Manifest generation

```bash
python scripts/build_audio_manifest.py \
  --dataset-root audio/data/raw/asvspoof_df \
  --output audio/data/manifests/asvspoof_df_manifest.csv
```

## First notebook

The initial baseline notebook is:

- `audio/notebooks/train_audio_deepfake_v1.ipynb`

It is designed for:

- mono audio
- 16 kHz
- fixed-length clips
- log-Mel spectrogram features
- lightweight CNN classifier
