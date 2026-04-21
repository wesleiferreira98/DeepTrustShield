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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
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

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.size(-1)
        if length > self.num_samples:
            if self.random_crop:
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
        waveform = self._pad_or_crop(waveform)
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


def count_labels(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["label_text"] for row in rows)
    return {"real": counts.get("real", 0), "fake": counts.get("fake", 0)}


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


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | list[float] | list[int]]:
    model.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    predictions = collect_predictions(model, loader, criterion, device)
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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_seconds = float(args.clip_seconds)
    num_workers = args.num_workers if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

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

    model = AudioCNN(args.n_mels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history_rows: list[dict] = []
    best_val_auc = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "audio_deepfake_v1_best.pth"
    started_at = time.time()

    print(f"device = {device}")
    if device.type == "cuda":
        print(f"gpu = {torch.cuda.get_device_name(0)}")
    print(f"manifest = {manifest_path}")
    print(f"train = {len(train_rows)} {count_labels(train_rows)}")
    print(f"val = {len(val_rows)} {count_labels(val_rows)}")
    print(f"test = {len(test_rows)} {count_labels(test_rows)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

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

    val_predictions = collect_predictions(model, val_loader, criterion, device)
    threshold_calibration = find_best_threshold(
        val_predictions["labels"],
        val_predictions["probs"],
        metric="f1",
    )
    test_metrics = evaluate(model, test_loader, criterion, device) if test_rows else None
    test_metrics_calibrated = None
    if test_rows:
        test_predictions = collect_predictions(model, test_loader, criterion, device)
        calibrated_metrics = metrics_from_probs(
            test_predictions["labels"],
            test_predictions["probs"],
            threshold=threshold_calibration["threshold"],
        )
        test_metrics_calibrated = {
            "loss": float(test_predictions["loss"]),
            **calibrated_metrics,
            "threshold": threshold_calibration["threshold"],
        }
    elapsed_minutes = (time.time() - started_at) / 60.0

    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"
    if history_rows or not history_path.exists():
        write_history_csv(history_path, history_rows)

    summary = {
        "task": "audio_deepfake_detection",
        "model_version": "audio_deepfake_v1",
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
        "seed": args.seed,
        "device": str(device),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "train_labels": count_labels(train_rows),
        "val_labels": count_labels(val_rows),
        "test_labels": count_labels(test_rows),
        "sample_audio_info": sample_audio_info(train_rows),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "threshold_calibration": threshold_calibration,
        "elapsed_minutes": round(elapsed_minutes, 3),
        "test_metrics": test_metrics,
        "test_metrics_calibrated": test_metrics_calibrated,
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
    print(f"history = {history_path}")
    print(f"summary = {summary_path}")


if __name__ == "__main__":
    main()
