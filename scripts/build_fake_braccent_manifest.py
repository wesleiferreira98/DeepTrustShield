#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
LABEL_ALIASES = {
    "real": "real",
    "bonafide": "real",
    "genuine": "real",
    "original": "real",
    "fake": "fake",
    "synthetic": "fake",
    "spoof": "fake",
    "generated": "fake",
}
DEFAULT_ACCENTS = {
    "baiano",
    "carioca",
    "fluminense",
    "nordestino",
    "sulista",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera manifesto para FakeBrAccent-D preservando label e sotaque. "
            "Funciona com pastas que contenham nomes de label e sotaque no caminho."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Raiz extraida da base FakeBrAccent-D.")
    parser.add_argument("--output", required=True, help="CSV de manifesto de saida.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--accent-names",
        default=",".join(sorted(DEFAULT_ACCENTS)),
        help="Lista separada por virgula com nomes de sotaque esperados no caminho.",
    )
    return parser.parse_args()


def collect_audio_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def infer_label(path: Path) -> str | None:
    tokens = [part.lower() for part in path.parts]
    for token in tokens:
        normalized = token.replace("-", "_").replace(" ", "_")
        if normalized in LABEL_ALIASES:
            return LABEL_ALIASES[normalized]
    name = path.stem.lower()
    for token, label in LABEL_ALIASES.items():
        if token in name:
            return label
    return None


def infer_accent(path: Path, accent_names: set[str]) -> str:
    tokens = [part.lower().replace("-", "_").replace(" ", "_") for part in path.parts]
    for token in tokens:
        if token in accent_names:
            return token
    name = path.stem.lower().replace("-", "_").replace(" ", "_")
    for accent in accent_names:
        if accent in name:
            return accent
    return "unknown"


def infer_speaker_id(path: Path) -> str:
    name = path.stem
    name = name.replace("FAKE_converted_", "converted_")
    match = re.search(r"converted_([^_]+)_frase_", name)
    if match:
        return match.group(1)
    return "unknown"


def assign_splits(rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in rows:
        key = (row["label"], row["accent"])
        groups.setdefault(key, {}).setdefault(row["speaker_id"], []).append(row)

    for speaker_groups in groups.values():
        speakers = list(speaker_groups.items())
        rng.shuffle(speakers)
        total_speakers = len(speakers)
        if total_speakers >= 3:
            val_speakers = max(1, int(round(total_speakers * val_ratio)))
            test_speakers = max(1, total_speakers - int(round(total_speakers * train_ratio)) - val_speakers)
            train_speakers = max(1, total_speakers - val_speakers - test_speakers)
        elif total_speakers == 2:
            train_speakers = 1
            val_speakers = 0
            test_speakers = 1
        else:
            train_speakers = total_speakers
            val_speakers = 0
            test_speakers = 0

        train_end = train_speakers
        val_end = train_speakers + val_speakers

        for index, (_, group_rows) in enumerate(speakers):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            for row in group_rows:
                row["split"] = split


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    accent_names = {
        item.strip().lower().replace("-", "_").replace(" ", "_")
        for item in args.accent_names.split(",")
        if item.strip()
    }

    if not dataset_root.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {dataset_root}")

    rows = []
    skipped = []
    for audio_path in collect_audio_files(dataset_root):
        label = infer_label(audio_path.relative_to(dataset_root))
        if label is None:
            skipped.append(str(audio_path))
            continue
        relative_path = audio_path.relative_to(dataset_root)
        rows.append(
            {
                "audio_path": str(audio_path),
                "relative_path": str(relative_path),
                "filename": audio_path.name,
                "extension": audio_path.suffix.lower(),
                "split": "",
                "label": label,
                "source": "fake_braccent_d",
                "accent": infer_accent(relative_path, accent_names),
                "speaker_id": infer_speaker_id(relative_path),
            }
        )

    assign_splits(rows, args.train_ratio, args.val_ratio, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio_path",
        "relative_path",
        "filename",
        "extension",
        "split",
        "label",
        "source",
        "accent",
        "speaker_id",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["split"], row["label"], row["accent"])
        counts[key] = counts.get(key, 0) + 1

    print(f"manifesto salvo em: {output_path}")
    print(f"linhas: {len(rows)}")
    print(f"ignorados_sem_label: {len(skipped)}")
    for key, count in sorted(counts.items()):
        split, label, accent = key
        print(f"{split:5s} {label:4s} {accent:12s} {count}")


if __name__ == "__main__":
    main()
