#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "audio" / "data" / "raw" / "asvspoof2019_la_sample"
SPLIT_MAP = {
    "train": {
        "audio_dir": "ASVspoof2019_LA_train/flac",
        "protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        "out_split": "train",
    },
    "dev": {
        "audio_dir": "ASVspoof2019_LA_dev/flac",
        "protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        "out_split": "val",
    },
    "eval": {
        "audio_dir": "ASVspoof2019_LA_eval/flac",
        "protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
        "out_split": "test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monta uma amostra pequena do ASVspoof 2019 LA no formato train/val/test com real/fake."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Raiz da base oficial extraida do ASVspoof 2019 LA.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT),
        help="Pasta de saida da amostra. Padrao: audio/data/raw/asvspoof2019_la_sample",
    )
    parser.add_argument(
        "--protocols-dir",
        default=None,
        help="Pasta dos protocolos. Padrao: <source-root>/ASVspoof2019_LA_cm_protocols",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=300,
        help="Quantidade maxima por split e por classe. Padrao: 300",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para amostragem reproducivel. Padrao: 42",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="Como materializar os arquivos. Padrao: hardlink",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove a pasta de saida antes de recriar a amostra.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_protocol(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            file_id = parts[1]
            label_text = parts[-1].strip().lower()
            if label_text == "bonafide":
                label = "real"
            elif label_text == "spoof":
                label = "fake"
            else:
                continue
            rows.append(
                {
                    "file_id": file_id,
                    "label": label,
                    "raw_label": label_text,
                }
            )
    return rows


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "audio_path",
        "relative_path",
        "filename",
        "extension",
        "split",
        "label",
        "source",
        "original_file_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    source_root = resolve_repo_path(args.source_root)
    output_root = resolve_repo_path(args.output_root)
    protocols_dir = (
        resolve_repo_path(args.protocols_dir)
        if args.protocols_dir
        else (source_root / "ASVspoof2019_LA_cm_protocols").resolve()
    )

    if not source_root.exists():
        raise FileNotFoundError(f"Base fonte nao encontrada: {source_root}")
    if not protocols_dir.exists():
        raise FileNotFoundError(f"Pasta de protocolos nao encontrada: {protocols_dir}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    sample_summary: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "fake": 0})

    for split_key, config in SPLIT_MAP.items():
        protocol_path = protocols_dir / config["protocol"]
        audio_dir = source_root / config["audio_dir"]
        out_split = config["out_split"]

        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocolo nao encontrado: {protocol_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"Pasta de audio nao encontrada: {audio_dir}")

        grouped: dict[str, list[dict]] = {"real": [], "fake": []}
        for row in parse_protocol(protocol_path):
            grouped[row["label"]].append(row)

        for label, items in grouped.items():
            random.shuffle(items)
            selected = items[: args.max_per_label]
            sample_summary[out_split][label] = len(selected)

            for item in selected:
                src = audio_dir / f"{item['file_id']}.flac"
                if not src.exists():
                    continue
                dst = output_root / out_split / label / src.name
                materialize_file(src, dst, args.mode)
                manifest_rows.append(
                    {
                        "audio_path": str(dst),
                        "relative_path": str(dst.relative_to(output_root)),
                        "filename": dst.name,
                        "extension": dst.suffix.lower(),
                        "split": out_split,
                        "label": label,
                        "source": "asvspoof2019_la",
                        "original_file_id": item["file_id"],
                    }
                )

    manifest_path = output_root / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    print(f"amostra criada em: {output_root}")
    print(f"manifesto: {manifest_path}")
    for split in ("train", "val", "test"):
        counts = sample_summary[split]
        print(f"{split}: real={counts['real']} fake={counts['fake']}")


if __name__ == "__main__":
    main()
