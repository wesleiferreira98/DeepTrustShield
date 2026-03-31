#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera um manifesto CSV para treinamento de deteccao de fraude com audio. "
            "Espera uma estrutura train/val/test com subpastas real/ e fake/."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Diretorio raiz com train/val/test e classes real/fake",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="CSV de saida",
    )
    return parser.parse_args()


def collect_audio_files(root: Path):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {dataset_root}")

    rows = []
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            subset_root = dataset_root / split / label
            if not subset_root.exists():
                continue
            for audio_path in collect_audio_files(subset_root):
                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "split": split,
                        "label": label,
                        "source": audio_path.parent.name,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "split", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifesto salvo em: {output_path}")
    print(f"total de linhas: {len(rows)}")


if __name__ == "__main__":
    main()
