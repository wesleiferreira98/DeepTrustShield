# Audio Fraud Detection Plan

## Objective

Build an MVP to detect whether a short voice clip is `real` or `synthetic/cloned`.

The initial goal is not to solve all audio fraud scenarios. The first milestone is a
practical binary detector that can later be exposed by the existing FastAPI backend.

## Scope

### In scope for MVP

- Short speech clips
- Single-speaker audio
- Binary classification: `real` vs `fake`
- Audio upload and backend inference
- Model evaluation with `AUC`, `EER`, `F1`, and calibrated threshold

### Out of scope for MVP

- Replay detection from speakerphone/second device
- Partial fake / splicing detection
- Speaker verification / identity matching
- Multi-speaker conversations
- Long-form audio analysis

## Problem Breakdown

There are three related but distinct problems:

1. `Speech deepfake / voice clone detection`
- TTS
- voice conversion
- cloned speech

2. `Replay detection`
- playing back a generated or recorded clip through another device

3. `Partial fake / splicing`
- only part of the clip is synthetic

The MVP focuses only on the first problem.

## Recommended Datasets

### Primary starting point

1. `ASVspoof 2021 DF`
- Best initial dataset for standalone speech deepfake detection
- Standard benchmark in the area

### Secondary / later-stage datasets

2. `ASVspoof 5`
- More recent and more adversarial
- Better suited for later robustness work

3. `ADD 2022`
- Useful for future expansion to more realistic or partially fake scenarios

## Audio Standardization

The training and inference pipeline should normalize all clips to:

- mono
- `16 kHz`
- fixed duration window, initially `4s`

Suggested preprocessing:

1. load audio
2. resample to `16 kHz`
3. convert to mono
4. trim silence conservatively
5. pad or crop to fixed length
6. normalize amplitude if needed

## Model Strategy

### v1 baseline

- input: `log-Mel spectrogram`
- model: lightweight `CNN` or `ResNet18`

Why:

- simple
- fast to train
- easy to debug
- good enough to validate the end-to-end pipeline

### v2 upgrade

- input: SSL embeddings from `wav2vec2` or `WavLM`
- head: `MLP` or light attention classifier

Why:

- usually better generalization
- more robust to dataset-specific artifacts

## Metrics

Use:

- `AUC`
- `EER`
- `F1`
- `Accuracy`
- calibrated threshold from validation set

`EER` is especially relevant for spoofing-style tasks.

## Proposed Repository Structure

```text
audio/
  data/
    raw/
    manifests/
  models/
  notebooks/
  research/
```

Suggested purpose:

- `audio/data/raw`: downloaded or prepared audio datasets
- `audio/data/manifests`: CSV files for train/val/test splits
- `audio/models`: trained checkpoints and metadata
- `audio/notebooks`: training notebooks
- `audio/research`: papers, notes, experiments

## MVP Backend Shape

Future endpoint:

`POST /v1/audio/analyze`

Expected response:

```json
{
  "label": "real",
  "p_fake_audio": 0.12,
  "threshold": 0.47,
  "model_version": "audio_v1",
  "duration_seconds": 3.98
}
```

## Execution Plan

1. Create `audio/` structure
2. Prepare dataset manifests for `ASVspoof 2021 DF`
3. Build `train_audio_deepfake_v1.ipynb`
4. Train `log-Mel + CNN` baseline
5. Evaluate with `AUC`, `EER`, and threshold calibration
6. Integrate backend endpoint
7. Expand later to replay detection

## Risks

- Overfitting to a specific vocoder or benchmark
- Real compressed internet audio looking fake
- Dataset mismatch between academic corpora and mobile recordings
- Good synthetic audio generalizing better than the baseline

## Immediate Next Step

Start with `ASVspoof 2021 DF` and build the first dataset manifest and training notebook.
