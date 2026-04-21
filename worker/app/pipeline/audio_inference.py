import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

_audio_model = None
_audio_device = None
_audio_model_version = None
_audio_threshold = 0.5
_audio_args = {}


class AudioCNN(nn.Module):
    def __init__(self, n_mels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 1)
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(-1)


def get_audio_device():
    global _audio_device
    if _audio_device is None:
        _audio_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using %s for audio inference", _audio_device)
    return _audio_device


def _load_summary(summary_path: str) -> dict:
    path = Path(summary_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load audio model summary %s: %s", summary_path, exc)
        return {}


def _summary_threshold(summary: dict) -> float | None:
    for key in ("threshold_calibration", "test_metrics_calibrated"):
        value = summary.get(key)
        if isinstance(value, dict) and isinstance(value.get("threshold"), (int, float)):
            return float(value["threshold"])
    return None


def load_audio_model(model_path: str, summary_path: str | None = None):
    global _audio_model, _audio_args, _audio_model_version, _audio_threshold

    if _audio_model is not None:
        return _audio_model

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modelo de audio nao encontrado: {model_file}")

    device = get_audio_device()
    checkpoint = torch.load(str(model_file), map_location=device, weights_only=False)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    summary = _load_summary(summary_path or str(model_file.with_name("summary.json")))
    n_mels = int(args.get("n_mels") or summary.get("n_mels") or 80)
    model = AudioCNN(n_mels=n_mels)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    threshold = _summary_threshold(summary)
    if threshold is not None:
        _audio_threshold = threshold

    _audio_args = {
        "sample_rate": int(args.get("sample_rate") or summary.get("sample_rate") or 16000),
        "clip_seconds": float(args.get("clip_seconds") or summary.get("clip_seconds") or 4.0),
        "n_mels": n_mels,
    }
    _audio_model_version = summary.get("model_version") or "audio_deepfake_v1"

    model = model.to(device)
    model.eval()
    _audio_model = model

    logger.info(
        "Loaded audio model %s from %s with threshold %.4f",
        _audio_model_version,
        model_file,
        _audio_threshold,
    )
    return model


def get_audio_threshold() -> float:
    return float(_audio_threshold)


def get_loaded_audio_model_version() -> str:
    return _audio_model_version or "audio_deepfake_v1"


def get_loaded_audio_args() -> dict:
    return dict(_audio_args)


def _pad_or_center_crop(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    length = waveform.size(-1)
    if length > num_samples:
        start = max((length - num_samples) // 2, 0)
        return waveform[:, start:start + num_samples]
    if length < num_samples:
        return torch.nn.functional.pad(waveform, (0, num_samples - length))
    return waveform


def preprocess_audio_file(audio_path: str, sample_rate: int, clip_seconds: float, n_mels: int) -> tuple[torch.Tensor, dict]:
    waveform, source_rate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    original_samples = int(waveform.size(-1))
    if source_rate != sample_rate:
        waveform = T.Resample(orig_freq=source_rate, new_freq=sample_rate)(waveform)

    num_samples = int(sample_rate * clip_seconds)
    waveform = _pad_or_center_crop(waveform, num_samples)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
    )
    amplitude_to_db = T.AmplitudeToDB()
    features = amplitude_to_db(mel_transform(waveform)).clamp(min=-80.0, max=0.0)
    features = (features + 80.0) / 80.0

    info = {
        "source_sample_rate": int(source_rate),
        "target_sample_rate": int(sample_rate),
        "original_samples": original_samples,
        "duration_seconds": original_samples / float(source_rate) if source_rate else None,
        "clip_seconds": float(clip_seconds),
        "n_mels": int(n_mels),
    }
    return features.unsqueeze(0), info


def run_audio_inference(audio_path: str, model_path: str, summary_path: str | None = None) -> dict:
    model = load_audio_model(model_path, summary_path)
    device = get_audio_device()
    args = get_loaded_audio_args()

    features, audio_info = preprocess_audio_file(
        audio_path,
        sample_rate=args["sample_rate"],
        clip_seconds=args["clip_seconds"],
        n_mels=args["n_mels"],
    )

    with torch.no_grad():
        logits = model(features.to(device))
        p_fake = float(torch.sigmoid(logits).item())

    threshold = get_audio_threshold()
    logger.info(
        "Audio inference result: p_fake = %.4f (threshold: %.4f, model: %s, file: %s)",
        p_fake,
        threshold,
        get_loaded_audio_model_version(),
        os.path.basename(audio_path),
    )

    return {
        "p_fake": p_fake,
        "threshold": threshold,
        "model_version": get_loaded_audio_model_version(),
        "audio_info": audio_info,
    }
