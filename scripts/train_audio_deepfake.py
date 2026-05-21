#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "audio" / "data" / "manifests" / "asvspoof_df_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "audio" / "models" / "audio_deepfake_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina um baseline de deteccao de audio real/fake com log-Mel + CNN leve."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="CSV com colunas audio_path, split, label.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Pasta para checkpoint, history e summary.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--architecture", choices=("cnn", "resnet_small"), default="cnn")
    parser.add_argument(
        "--pos-weight",
        default="auto",
        help="Peso da classe fake no BCE. Use 'auto' para real/fake no treino ou um float.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Usa mixed precision quando o device for CUDA.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--attention-min-recall", type=float, default=0.90)
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device CUDA solicitado, mas torch.cuda.is_available() retornou False.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_text = (row.get("label") or "").strip().lower()
            if label_text not in {"real", "fake"}:
                continue
            audio_path = Path(row["audio_path"]).expanduser()
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            rows.append(
                {
                    "audio_path": str(audio_path),
                    "split": (row.get("split") or "").strip(),
                    "label": 0 if label_text == "real" else 1,
                    "label_text": label_text,
                    "source": row.get("source", ""),
                    "accent": row.get("accent", "unknown") or "unknown",
                    "speaker_id": row.get("speaker_id", "unknown") or "unknown",
                    "start": row.get("start", ""),
                    "end": row.get("end", ""),
                    "file_label": row.get("file_label", label_text) or label_text,
                    "manipulation_type": row.get("manipulation_type", "none") or "none",
                    "overlap_ratio": row.get("overlap_ratio", ""),
                    "ground_truth_path": row.get("ground_truth_path", ""),
                }
            )
    return rows


class AudioDeepfakeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        *,
        sample_rate: int,
        clip_seconds: float,
        mel_transform: T.MelSpectrogram,
        amplitude_to_db: T.AmplitudeToDB,
        random_crop: bool,
    ) -> None:
        self.rows = rows
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * clip_seconds)
        self.mel_transform = mel_transform
        self.amplitude_to_db = amplitude_to_db
        self.random_crop = random_crop
        self.resamplers: dict[int, T.Resample] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_waveform(self, audio_path: str) -> torch.Tensor:
        waveform, source_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != self.sample_rate:
            if source_rate not in self.resamplers:
                self.resamplers[source_rate] = T.Resample(orig_freq=source_rate, new_freq=self.sample_rate)
            waveform = self.resamplers[source_rate](waveform)
        return waveform

    def _window_from_row(self, waveform: torch.Tensor, row: dict) -> torch.Tensor:
        try:
            start_seconds = float(row.get("start"))
            end_seconds = float(row.get("end"))
        except (TypeError, ValueError):
            return self._pad_or_crop(waveform)

        if end_seconds <= start_seconds:
            return self._pad_or_crop(waveform)

        start_sample = max(0, int(round(start_seconds * self.sample_rate)))
        end_sample = min(waveform.size(-1), int(round(end_seconds * self.sample_rate)))
        window = waveform[:, start_sample:end_sample]
        return self._pad_or_crop(window, random_crop=False)

    def _pad_or_crop(self, waveform: torch.Tensor, random_crop: bool | None = None) -> torch.Tensor:
        random_crop = self.random_crop if random_crop is None else random_crop
        length = waveform.size(-1)
        if length > self.num_samples:
            if random_crop:
                start = random.randint(0, length - self.num_samples)
            else:
                start = max((length - self.num_samples) // 2, 0)
            waveform = waveform[:, start : start + self.num_samples]
        elif length < self.num_samples:
            pad_amount = self.num_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        waveform = self._load_waveform(row["audio_path"])
        waveform = self._window_from_row(waveform, row)
        features = self.amplitude_to_db(self.mel_transform(waveform)).clamp(min=-80.0, max=0.0)
        features = (features + 80.0) / 80.0
        return features, torch.tensor(float(row["label"]), dtype=torch.float32)


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


def build_model(architecture: str, n_mels: int, *, dropout: float) -> nn.Module:
    if architecture == "resnet_small":
        return AudioResNetSmall(n_mels=n_mels, dropout=dropout)
    return AudioCNN(n_mels=n_mels)


def count_labels(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["label_text"] for row in rows)
    return {"real": counts.get("real", 0), "fake": counts.get("fake", 0)}


def count_by_field(rows: list[dict], field: str) -> dict[str, int]:
    counts = Counter((row.get(field) or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def resolve_pos_weight(value: str, train_rows: list[dict]) -> float:
    if value.strip().lower() != "auto":
        return float(value)
    labels = count_labels(train_rows)
    real_count = labels["real"]
    fake_count = labels["fake"]
    if fake_count == 0:
        return 1.0
    return real_count / fake_count


def metrics_from_probs(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    if not labels:
        return {"auc": 0.0, "f1": 0.0, "acc": 0.0}

    binary_preds = [1 if prob >= threshold else 0 for prob in probs]
    auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.0
    return {
        "auc": auc,
        "f1": float(f1_score(labels, binary_preds, zero_division=0)),
        "acc": float(accuracy_score(labels, binary_preds)),
    }


def find_best_threshold(
    labels: list[int],
    probs: list[float],
    *,
    metric: str = "f1",
) -> dict[str, float]:
    if not labels or not probs:
        return {"threshold": 0.5, "f1": 0.0, "acc": 0.0, "auc": 0.0}

    candidates = sorted(set(float(prob) for prob in probs))
    candidates = [0.0, *candidates, 1.0]
    best = {"threshold": 0.5, **metrics_from_probs(labels, probs, threshold=0.5)}
    for threshold in candidates:
        metrics = metrics_from_probs(labels, probs, threshold=threshold)
        is_better = (
            metrics[metric] > best[metric]
            or (
                metrics[metric] == best[metric]
                and metrics["acc"] > best["acc"]
            )
        )
        if is_better:
            best = {"threshold": float(threshold), **metrics}
    return best


def find_attention_threshold(labels: list[int], probs: list[float], *, min_recall: float = 0.90) -> dict[str, float]:
    if not labels or not probs:
        return {"threshold": 0.5, "recall_fake": 0.0, "false_positive_rate": 0.0}

    candidates = sorted(set(float(prob) for prob in probs))
    candidates = [0.0, *candidates, 1.0]
    fake_count = max(sum(1 for label in labels if label == 1), 1)
    real_count = max(sum(1 for label in labels if label == 0), 1)
    feasible = []
    for threshold in candidates:
        true_positive = sum(1 for label, prob in zip(labels, probs) if label == 1 and prob >= threshold)
        false_positive = sum(1 for label, prob in zip(labels, probs) if label == 0 and prob >= threshold)
        recall_fake = true_positive / fake_count
        false_positive_rate = false_positive / real_count
        item = {
            "threshold": float(threshold),
            "recall_fake": float(recall_fake),
            "false_positive_rate": float(false_positive_rate),
        }
        if recall_fake >= min_recall:
            feasible.append(item)

    if feasible:
        return sorted(feasible, key=lambda item: (item["false_positive_rate"], -item["threshold"]))[0]

    return max(
        (
            {
                "threshold": float(threshold),
                "recall_fake": sum(1 for label, prob in zip(labels, probs) if label == 1 and prob >= threshold) / fake_count,
                "false_positive_rate": sum(1 for label, prob in zip(labels, probs) if label == 0 and prob >= threshold) / real_count,
            }
            for threshold in candidates
        ),
        key=lambda item: (item["recall_fake"], -item["false_positive_rate"]),
    )


def aggregate_file_predictions(rows: list[dict], labels: list[int], probs: list[float]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for row, label, prob in zip(rows, labels, probs):
        key = row["audio_path"]
        bucket = grouped.setdefault(
            key,
            {
                "label": 1 if row.get("file_label") == "fake" else int(label),
                "probs": [],
                "source": row.get("source", ""),
                "manipulation_type": row.get("manipulation_type", "none"),
            },
        )
        bucket["label"] = max(bucket["label"], int(label), 1 if row.get("file_label") == "fake" else 0)
        bucket["probs"].append(float(prob))

    outputs = {
        "mean": {"labels": [], "probs": []},
        "max": {"labels": [], "probs": []},
        "top3_mean": {"labels": [], "probs": []},
        "top5_mean": {"labels": [], "probs": []},
    }
    for bucket in grouped.values():
        sorted_probs = sorted(bucket["probs"], reverse=True)
        aggregations = {
            "mean": sum(sorted_probs) / len(sorted_probs),
            "max": sorted_probs[0],
            "top3_mean": sum(sorted_probs[:3]) / min(3, len(sorted_probs)),
            "top5_mean": sum(sorted_probs[:5]) / min(5, len(sorted_probs)),
        }
        for name, prob in aggregations.items():
            outputs[name]["labels"].append(bucket["label"])
            outputs[name]["probs"].append(prob)
    return outputs


def aggregate_metrics(rows: list[dict], labels: list[int], probs: list[float], *, min_recall: float) -> dict[str, dict]:
    output = {}
    for name, values in aggregate_file_predictions(rows, labels, probs).items():
        threshold_calibration = find_best_threshold(values["labels"], values["probs"], metric="f1")
        attention_calibration = find_attention_threshold(values["labels"], values["probs"], min_recall=min_recall)
        output[name] = {
            "count": len(values["labels"]),
            "threshold_calibration": threshold_calibration,
            "attention_threshold_calibration": attention_calibration,
        }
    return output


def metrics_by_group(
    rows: list[dict],
    labels: list[int],
    probs: list[float],
    *,
    group_field: str,
    threshold: float,
) -> dict[str, dict]:
    grouped: dict[str, dict[str, list]] = {}
    for row, label, prob in zip(rows, labels, probs):
        group = row.get(group_field) or "unknown"
        bucket = grouped.setdefault(group, {"labels": [], "probs": []})
        bucket["labels"].append(label)
        bucket["probs"].append(prob)

    output = {}
    for group, values in sorted(grouped.items()):
        group_labels = values["labels"]
        group_probs = values["probs"]
        group_metrics = metrics_from_probs(group_labels, group_probs, threshold=threshold)
        real_labels = [label for label in group_labels if label == 0]
        fake_labels = [label for label in group_labels if label == 1]
        false_positive_count = sum(
            1
            for label, prob in zip(group_labels, group_probs)
            if label == 0 and prob >= threshold
        )
        false_negative_count = sum(
            1
            for label, prob in zip(group_labels, group_probs)
            if label == 1 and prob < threshold
        )
        output[group] = {
            **group_metrics,
            "count": len(group_labels),
            "real_count": len(real_labels),
            "fake_count": len(fake_labels),
            "false_positive_rate": false_positive_count / max(len(real_labels), 1),
            "false_negative_rate": false_negative_count / max(len(fake_labels), 1),
        }
    return output


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> dict[str, float | list[float] | list[int]]:
    model.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)

            total_loss += loss.item() * labels.size(0)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    base_metrics = metrics_from_probs(all_labels, all_probs, threshold=0.5)
    return {
        "loss": total_loss / max(len(all_labels), 1),
        **base_metrics,
        "probs": all_probs,
        "labels": all_labels,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> dict[str, float]:
    predictions = collect_predictions(model, loader, criterion, device, use_amp=use_amp)
    return {
        "loss": float(predictions["loss"]),
        "auc": float(predictions["auc"]),
        "f1": float(predictions["f1"]),
        "acc": float(predictions["acc"]),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    binary_preds = [1 if prob >= 0.5 else 0 for prob in all_probs]
    if len(set(all_labels)) > 1:
        auc = float(roc_auc_score(all_labels, all_probs))
    else:
        auc = 0.0

    return {
        "loss": total_loss / max(len(all_labels), 1),
        "auc": auc,
        "f1": float(f1_score(all_labels, binary_preds, zero_division=0)),
        "acc": float(accuracy_score(all_labels, binary_preds)) if all_labels else 0.0,
    }


def write_history_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_f1",
        "train_acc",
        "val_loss",
        "val_auc",
        "val_f1",
        "val_acc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_audio_info(rows: list[dict]) -> dict[str, float | int] | None:
    if not rows:
        return None
    path = Path(rows[0]["audio_path"])
    info = sf.info(str(path))
    return {
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": float(info.duration),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_path = resolve_repo_path(args.manifest)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto nao encontrado: {manifest_path}")

    rows = load_manifest(manifest_path)
    train_rows = [row for row in rows if row["split"] == args.train_split]
    val_rows = [row for row in rows if row["split"] == args.val_split]
    test_rows = [row for row in rows if row["split"] == args.test_split]

    if not train_rows or not val_rows:
        raise RuntimeError("Manifesto precisa ter pelo menos train e val com amostras real/fake.")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    clip_seconds = float(args.clip_seconds)
    num_workers = args.num_workers if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    use_amp = bool(args.amp and device.type == "cuda")

    mel_transform = T.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=args.n_mels,
    )
    amplitude_to_db = T.AmplitudeToDB()

    train_dataset = AudioDeepfakeDataset(
        train_rows,
        sample_rate=args.sample_rate,
        clip_seconds=clip_seconds,
        mel_transform=mel_transform,
        amplitude_to_db=amplitude_to_db,
        random_crop=True,
    )
    val_dataset = AudioDeepfakeDataset(
        val_rows,
        sample_rate=args.sample_rate,
        clip_seconds=clip_seconds,
        mel_transform=mel_transform,
        amplitude_to_db=amplitude_to_db,
        random_crop=False,
    )
    test_dataset = AudioDeepfakeDataset(
        test_rows,
        sample_rate=args.sample_rate,
        clip_seconds=clip_seconds,
        mel_transform=mel_transform,
        amplitude_to_db=amplitude_to_db,
        random_crop=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.architecture, args.n_mels, dropout=args.dropout).to(device)
    pos_weight_value = resolve_pos_weight(str(args.pos_weight), train_rows)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history_rows: list[dict] = []
    best_val_auc = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "audio_deepfake_v1_best.pth"
    started_at = time.time()

    print(f"device = {device}")
    if device.type == "cuda":
        print(f"gpu = {torch.cuda.get_device_name(0)}")
        print(f"amp = {use_amp}")
    print(f"manifest = {manifest_path}")
    print(f"train = {len(train_rows)} {count_labels(train_rows)}")
    print(f"val = {len(val_rows)} {count_labels(val_rows)}")
    print(f"test = {len(test_rows)} {count_labels(test_rows)}")
    print(f"pos_weight = {pos_weight_value:.4f}")
    print(f"architecture = {args.architecture}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_auc": round(train_metrics["auc"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_auc": round(val_metrics["auc"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_acc": round(val_metrics["acc"], 6),
        }
        history_rows.append(row)

        improved = val_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )

        marker = "*" if improved else ""
        print(
            f"epoch={epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['acc']:.4f} {marker}"
        )

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        best_epoch = int(state.get("best_epoch", best_epoch))
        best_val_auc = float(state.get("best_val_auc", best_val_auc))

    val_predictions = collect_predictions(model, val_loader, criterion, device, use_amp=use_amp)
    threshold_calibration = find_best_threshold(
        val_predictions["labels"],
        val_predictions["probs"],
        metric="f1",
    )
    attention_threshold_calibration = find_attention_threshold(
        val_predictions["labels"],
        val_predictions["probs"],
        min_recall=args.attention_min_recall,
    )
    val_file_aggregation_metrics = aggregate_metrics(
        val_rows,
        val_predictions["labels"],
        val_predictions["probs"],
        min_recall=args.attention_min_recall,
    )
    calibrated_threshold = threshold_calibration["threshold"]
    val_metrics_by_accent = metrics_by_group(
        val_rows,
        val_predictions["labels"],
        val_predictions["probs"],
        group_field="accent",
        threshold=calibrated_threshold,
    )
    val_metrics_by_source = metrics_by_group(
        val_rows,
        val_predictions["labels"],
        val_predictions["probs"],
        group_field="source",
        threshold=calibrated_threshold,
    )
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp=use_amp) if test_rows else None
    test_metrics_calibrated = None
    test_metrics_by_accent = None
    test_metrics_by_source = None
    test_file_aggregation_metrics = None
    if test_rows:
        test_predictions = collect_predictions(model, test_loader, criterion, device, use_amp=use_amp)
        calibrated_metrics = metrics_from_probs(
            test_predictions["labels"],
            test_predictions["probs"],
            threshold=calibrated_threshold,
        )
        test_metrics_calibrated = {
            "loss": float(test_predictions["loss"]),
            **calibrated_metrics,
            "threshold": calibrated_threshold,
        }
        test_metrics_by_accent = metrics_by_group(
            test_rows,
            test_predictions["labels"],
            test_predictions["probs"],
            group_field="accent",
            threshold=calibrated_threshold,
        )
        test_metrics_by_source = metrics_by_group(
            test_rows,
            test_predictions["labels"],
            test_predictions["probs"],
            group_field="source",
            threshold=calibrated_threshold,
        )
        test_file_aggregation_metrics = aggregate_metrics(
            test_rows,
            test_predictions["labels"],
            test_predictions["probs"],
            min_recall=args.attention_min_recall,
        )
    elapsed_minutes = (time.time() - started_at) / 60.0

    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"
    if history_rows or not history_path.exists():
        write_history_csv(history_path, history_rows)

    summary = {
        "task": "audio_deepfake_detection",
        "model_version": "audio_deepfake_v1",
        "architecture": args.architecture,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "sample_rate": args.sample_rate,
        "clip_seconds": clip_seconds,
        "n_mels": args.n_mels,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "pos_weight": pos_weight_value,
        "seed": args.seed,
        "device": str(device),
        "amp": use_amp,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "train_labels": count_labels(train_rows),
        "val_labels": count_labels(val_rows),
        "test_labels": count_labels(test_rows),
        "train_accents": count_by_field(train_rows, "accent"),
        "val_accents": count_by_field(val_rows, "accent"),
        "test_accents": count_by_field(test_rows, "accent"),
        "train_speakers": count_by_field(train_rows, "speaker_id"),
        "val_speakers": count_by_field(val_rows, "speaker_id"),
        "test_speakers": count_by_field(test_rows, "speaker_id"),
        "sample_audio_info": sample_audio_info(train_rows),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "threshold_calibration": threshold_calibration,
        "attention_threshold_calibration": attention_threshold_calibration,
        "attention_min_recall": args.attention_min_recall,
        "val_file_aggregation_metrics": val_file_aggregation_metrics,
        "val_metrics_by_accent": val_metrics_by_accent,
        "val_metrics_by_source": val_metrics_by_source,
        "elapsed_minutes": round(elapsed_minutes, 3),
        "test_metrics": test_metrics,
        "test_metrics_calibrated": test_metrics_calibrated,
        "test_metrics_by_accent": test_metrics_by_accent,
        "test_metrics_by_source": test_metrics_by_source,
        "test_file_aggregation_metrics": test_file_aggregation_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"best_epoch = {best_epoch}")
    print(f"best_val_auc = {best_val_auc:.6f}")
    if test_metrics:
        print(
            f"test_loss = {test_metrics['loss']:.4f} "
            f"test_auc = {test_metrics['auc']:.4f} "
            f"test_f1 = {test_metrics['f1']:.4f} "
            f"test_acc = {test_metrics['acc']:.4f}"
        )
    if test_metrics_calibrated:
        print(
            f"calibrated_threshold = {test_metrics_calibrated['threshold']:.4f} "
            f"test_f1_cal = {test_metrics_calibrated['f1']:.4f} "
            f"test_acc_cal = {test_metrics_calibrated['acc']:.4f}"
        )
    print(
        f"attention_threshold = {attention_threshold_calibration['threshold']:.4f} "
        f"attention_recall_fake = {attention_threshold_calibration['recall_fake']:.4f} "
        f"attention_fpr = {attention_threshold_calibration['false_positive_rate']:.4f}"
    )
    print(f"history = {history_path}")
    print(f"summary = {summary_path}")


if __name__ == "__main__":
    main()
