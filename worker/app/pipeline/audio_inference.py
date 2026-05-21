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
_audio_model_cache = {}
MAX_SEGMENTS = 32


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.shortcut(x))


class AudioResNetSmall(nn.Module):
    def __init__(self, n_mels: int, *, dropout: float = 0.20) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32, dropout=dropout * 0.5),
            ResidualBlock(32, 64, stride=2, dropout=dropout * 0.5),
            ResidualBlock(64, 64, dropout=dropout * 0.5),
            ResidualBlock(64, 128, stride=2, dropout=dropout),
            ResidualBlock(128, 128, dropout=dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x).squeeze(-1)


def build_audio_model(architecture: str, n_mels: int, *, dropout: float) -> nn.Module:
    if architecture == "resnet_small":
        return AudioResNetSmall(n_mels=n_mels, dropout=dropout)
    return AudioCNN(n_mels=n_mels)


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


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def load_audio_model(model_path: str, summary_path: str | None = None):
    global _audio_model, _audio_args, _audio_model_version, _audio_threshold

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modelo de audio nao encontrado: {model_file}")

    device = get_audio_device()
    cache_key = (str(model_file.resolve()), str(Path(summary_path).resolve()) if summary_path else "")
    if cache_key in _audio_model_cache:
        cached = _audio_model_cache[cache_key]
        _audio_model = cached["model"]
        _audio_args = cached["args"]
        _audio_model_version = cached["version"]
        _audio_threshold = cached["threshold"]
        return _audio_model

    checkpoint = torch.load(str(model_file), map_location=device, weights_only=False)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    summary = _load_summary(summary_path or str(model_file.with_name("summary.json")))
    n_mels = int(_first_present(args.get("n_mels"), summary.get("n_mels"), 80))
    architecture = _first_present(args.get("architecture"), summary.get("architecture"), "cnn")
    dropout = float(_first_present(args.get("dropout"), summary.get("dropout"), 0.20))
    model = build_audio_model(str(architecture), n_mels=n_mels, dropout=dropout)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    threshold = _summary_threshold(summary)
    if threshold is not None:
        _audio_threshold = threshold

    _audio_args = {
        "sample_rate": int(_first_present(args.get("sample_rate"), summary.get("sample_rate"), 16000)),
        "clip_seconds": float(_first_present(args.get("clip_seconds"), summary.get("clip_seconds"), 4.0)),
        "n_mels": n_mels,
        "architecture": str(architecture),
    }
    _audio_model_version = summary.get("model_version") or "audio_deepfake_v1"

    model = model.to(device)
    model.eval()
    _audio_model = model
    _audio_model_cache[cache_key] = {
        "model": model,
        "args": dict(_audio_args),
        "version": _audio_model_version,
        "threshold": _audio_threshold,
    }

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


def _pad_or_crop_from_start(waveform: torch.Tensor, start: int, num_samples: int) -> torch.Tensor:
    end = start + num_samples
    segment = waveform[:, start:min(end, waveform.size(-1))]
    if segment.size(-1) < num_samples:
        segment = torch.nn.functional.pad(segment, (0, num_samples - segment.size(-1)))
    return segment


def _features_from_waveform(waveform: torch.Tensor, sample_rate: int, n_mels: int) -> torch.Tensor:
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
    )
    amplitude_to_db = T.AmplitudeToDB()
    features = amplitude_to_db(mel_transform(waveform)).clamp(min=-80.0, max=0.0)
    return (features + 80.0) / 80.0


def _load_standard_waveform(audio_path: str, sample_rate: int) -> tuple[torch.Tensor, dict]:
    waveform, source_rate = torchaudio.load(audio_path)
    original_channels = int(waveform.size(0))
    original_samples = int(waveform.size(-1))

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = T.Resample(orig_freq=source_rate, new_freq=sample_rate)(waveform)

    info = {
        "source_sample_rate": int(source_rate),
        "target_sample_rate": int(sample_rate),
        "channels": original_channels,
        "original_samples": original_samples,
        "duration_seconds": original_samples / float(source_rate) if source_rate else None,
    }
    return waveform, info


def preprocess_audio_file(audio_path: str, sample_rate: int, clip_seconds: float, n_mels: int) -> tuple[torch.Tensor, dict]:
    waveform, info = _load_standard_waveform(audio_path, sample_rate)
    num_samples = int(sample_rate * clip_seconds)
    waveform = _pad_or_center_crop(waveform, num_samples)
    features = _features_from_waveform(waveform, sample_rate, n_mels)
    info = {**info, "clip_seconds": float(clip_seconds), "n_mels": int(n_mels)}
    return features.unsqueeze(0), info


def _segment_starts(total_samples: int, window_samples: int, stride_samples: int) -> list[int]:
    if total_samples <= window_samples:
        return [0]

    starts = list(range(0, total_samples - window_samples + 1, stride_samples))
    final_start = total_samples - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)

    if len(starts) <= MAX_SEGMENTS:
        return starts

    step = (len(starts) - 1) / float(MAX_SEGMENTS - 1)
    sampled = [starts[round(index * step)] for index in range(MAX_SEGMENTS)]
    return sorted(set(sampled))


def run_audio_segment_inference(
    model: nn.Module,
    audio_path: str,
    *,
    sample_rate: int,
    clip_seconds: float,
    n_mels: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    waveform, _ = _load_standard_waveform(audio_path, sample_rate)
    total_samples = int(waveform.size(-1))
    window_samples = int(sample_rate * clip_seconds)
    stride_seconds = max(clip_seconds / 2.0, 1.0)
    stride_samples = int(sample_rate * stride_seconds)
    starts = _segment_starts(total_samples, window_samples, stride_samples)

    features_batch = []
    segment_meta = []
    for index, start in enumerate(starts):
        segment_waveform = _pad_or_crop_from_start(waveform, start, window_samples)
        features_batch.append(_features_from_waveform(segment_waveform, sample_rate, n_mels))
        segment_meta.append(
            {
                "index": index,
                "start": start / float(sample_rate),
                "end": min(start + window_samples, total_samples) / float(sample_rate),
                "duration_seconds": min(window_samples, max(total_samples - start, 0)) / float(sample_rate),
            }
        )

    if not features_batch:
        return [], {"count": 0, "coverage_fraction": 0.0, "stride_seconds": stride_seconds}

    batch = torch.stack(features_batch, dim=0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).detach().cpu().numpy().tolist()

    segments = [
        {
            **meta,
            "p_fake": float(prob),
        }
        for meta, prob in zip(segment_meta, probs)
    ]
    covered_until = max((segment["end"] for segment in segments), default=0.0)
    duration_seconds = total_samples / float(sample_rate) if sample_rate else 0.0
    summary = {
        "count": len(segments),
        "window_seconds": float(clip_seconds),
        "stride_seconds": float(stride_seconds),
        "coverage_fraction": min(1.0, covered_until / max(duration_seconds, 0.001)),
        "max_segments": MAX_SEGMENTS,
    }
    return segments, summary


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

    segments, segment_summary = run_audio_segment_inference(
        model,
        audio_path,
        sample_rate=args["sample_rate"],
        clip_seconds=args["clip_seconds"],
        n_mels=args["n_mels"],
        device=device,
    )

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
        "segments": segments,
        "segment_summary": segment_summary,
    }
