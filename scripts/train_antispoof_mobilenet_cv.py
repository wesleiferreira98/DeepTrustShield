#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina MobileNetV3 anti-spoof usando folds do manifesto.")
    parser.add_argument("--manifest", default="data/manifests/antispoof_manifest_folds.csv")
    parser.add_argument("--output-root", default="model/weights/antispoof_mobilenet_v1_cv")
    parser.add_argument("--fold", action="append", type=int, help="Fold de validacao. Pode repetir. Padrao: todos.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--unfrozen-lr-factor", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument(
        "--export-existing",
        action="store_true",
        help="Exporta checkpoints existentes dos folds sem treinar novamente.",
    )
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AntiSpoofDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["path_abs"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["label_id"], dtype=torch.float32)
        return image, label


class AntiSpoofMobileNetV3(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def set_backbone_trainable(model: AntiSpoofMobileNetV3, trainable: bool) -> None:
    for name, param in model.backbone.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )


def make_transforms(image_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.08, p=1.0)], p=0.25),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.24, contrast=0.24, saturation=0.18, hue=0.04)],
                p=0.75,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4))], p=0.18),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.4, 2.5), value="random"),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, val_tfms


def make_weighted_sampler(dataframe: pd.DataFrame):
    counts = dataframe["label_id"].value_counts().to_dict()
    weights = dataframe["label_id"].map(lambda label: 1.0 / counts[int(label)]).to_numpy(dtype=np.float64).copy()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def compute_metrics(y_true, y_prob, threshold: float):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    apcer = fn / max(tp + fn, 1)
    bpcer = fp / max(tn + fp, 1)
    acer = (apcer + bpcer) / 2
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob):
    candidates = np.linspace(0.05, 0.95, 181)
    scored = [(thr, compute_metrics(y_true, y_prob, thr)) for thr in candidates]
    return min(scored, key=lambda item: (item[1]["acer"], item[1]["apcer"], item[1]["bpcer"]))


def smooth_binary_labels(labels, smoothing: float):
    if smoothing <= 0:
        return labels
    return labels * (1.0 - 2.0 * smoothing) + smoothing


def train_one_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    y_true, y_prob = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss_labels = smooth_binary_labels(labels, args.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, loss_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
    return total_loss / len(loader.dataset), compute_metrics(y_true, y_prob, args.threshold)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float):
    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
    metrics = compute_metrics(y_true, y_prob, threshold)
    best_threshold, best_threshold_metrics = find_best_threshold(y_true, y_prob)
    return total_loss / len(loader.dataset), metrics, float(best_threshold), best_threshold_metrics


def export_onnx(model, output_path: Path, image_size: int) -> None:
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model_cpu,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=18,
        external_data=False,
        dynamo=False,
    )


def export_existing_fold(fold: int, args: argparse.Namespace, device: torch.device) -> dict:
    output_dir = resolve_repo_path(args.output_root) / f"fold_{fold}"
    checkpoint_path = output_dir / "antispoof_mobilenet_v1_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    model = AntiSpoofMobileNetV3(dropout=args.dropout).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    onnx_path = output_dir / "antispoof_mobilenet_v1.onnx"
    export_onnx(model, onnx_path, args.image_size)

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metadata = {
        "model_version": "antispoof_mobilenet_v1",
        "task": "face_anti_spoof",
        "labels": {"0": "live", "1": "spoof"},
        "fold": fold,
        "image_size": args.image_size,
        "threshold": float(checkpoint.get("config", {}).get("best_threshold", args.threshold)),
        "fixed_threshold": args.threshold,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "epoch": int(checkpoint.get("epoch", summary.get("epoch", -1))),
        "metrics": checkpoint.get("metrics", {}),
        "best_threshold_metrics": checkpoint.get("best_threshold_metrics", {}),
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "onnx": str(onnx_path.relative_to(REPO_ROOT)),
    }
    (output_dir / "antispoof_mobilenet_v1_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"fold={fold} exported {onnx_path}")
    return summary | {"fold": fold, "onnx": str(onnx_path.relative_to(REPO_ROOT))}


def train_fold(df: pd.DataFrame, fold: int, args: argparse.Namespace, device: torch.device) -> dict:
    output_dir = resolve_repo_path(args.output_root) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_tfms, val_tfms = make_transforms(args.image_size)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    train_loader = DataLoader(
        AntiSpoofDataset(train_df, train_tfms),
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_df),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers > 0,
    )
    val_loader = DataLoader(
        AntiSpoofDataset(val_df, val_tfms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers > 0,
    )

    model = AntiSpoofMobileNetV3(dropout=args.dropout).to(device)
    set_backbone_trainable(model, False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    best_acer = float("inf")
    best_row = None
    history = []
    stale_epochs = 0
    started = time.time()
    best_path = output_dir / "antispoof_mobilenet_v1_best.pth"

    print(f"\n=== fold {fold} train={len(train_df)} val={len(val_df)} ===")
    print("train labels:", train_df["label"].value_counts().to_dict())
    print("val labels:", val_df["label"].value_counts().to_dict())

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, args.lr * args.unfrozen_lr_factor, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, args)
        val_loss, val_metrics, best_threshold, best_threshold_metrics = evaluate(
            model, val_loader, criterion, device, args.threshold
        )
        scheduler.step(val_loss)
        row = {
            "fold": fold,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_best_threshold": best_threshold,
            "val_best_acer": best_threshold_metrics["acer"],
            "val_best_apcer": best_threshold_metrics["apcer"],
            "val_best_bpcer": best_threshold_metrics["bpcer"],
        }
        history.append(row)
        improved = best_threshold_metrics["acer"] < best_acer
        if improved:
            stale_epochs = 0
            best_acer = best_threshold_metrics["acer"]
            best_row = row
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args) | {"fold": fold, "best_threshold": best_threshold},
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "best_threshold_metrics": best_threshold_metrics,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        print(
            f"fold={fold} epoch={epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"auc={val_metrics['auc']:.4f} fixed_ACER={val_metrics['acer']:.4f} "
            f"best_thr={best_threshold:.2f} best_ACER={best_threshold_metrics['acer']:.4f} "
            f"{'*' if improved else ''}"
        )
        if stale_epochs >= args.early_stopping_patience:
            print(f"fold={fold} early stopping")
            break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    if args.export_onnx:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        export_onnx(model, output_dir / "antispoof_mobilenet_v1.onnx", args.image_size)

    assert best_row is not None
    best_row = dict(best_row)
    best_row["output_dir"] = str(output_dir.relative_to(REPO_ROOT))
    best_row["time_minutes"] = (time.time() - started) / 60
    (output_dir / "summary.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    return best_row


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    manifest_path = resolve_repo_path(args.manifest)
    df = pd.read_csv(manifest_path)
    df["path_abs"] = df["path"].map(lambda item: str((REPO_ROOT / item).resolve()))
    df["label_id"] = df["label_id"].astype(int)
    df["fold"] = df["fold"].astype(int)
    folds = args.fold if args.fold else sorted(df["fold"].unique().tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    if torch.cuda.is_available():
        print("gpu =", torch.cuda.get_device_name(0))

    if args.export_existing:
        summaries = [export_existing_fold(fold, args, device) for fold in folds]
    else:
        summaries = [train_fold(df, fold, args, device) for fold in folds]
    summary_df = pd.DataFrame(summaries).sort_values("fold")
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n=== CV summary ===")
    print(summary_df[["fold", "epoch", "val_best_threshold", "val_best_apcer", "val_best_bpcer", "val_best_acer"]])
    print("mean_best_acer =", float(summary_df["val_best_acer"].mean()))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
