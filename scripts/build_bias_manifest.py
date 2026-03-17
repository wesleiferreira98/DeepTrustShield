#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera um CSV de auditoria de vies a partir de uma arvore de diretorios. "
            "Layouts suportados: root/group/label/imagens ou root/label/group/imagens."
        )
    )
    parser.add_argument("--root", required=True, help="Raiz do dataset")
    parser.add_argument("--output", required=True, help="CSV de saida")
    parser.add_argument(
        "--layout",
        choices=("group-label", "label-group"),
        default="group-label",
        help="Estrutura dos diretorios",
    )
    parser.add_argument(
        "--max-per-cell",
        type=int,
        default=None,
        help="Limite de imagens por combinacao group+label",
    )
    parser.add_argument(
        "--relative-to-output",
        action="store_true",
        help="Escreve image_path relativo ao diretorio do CSV de saida",
    )
    return parser.parse_args()


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def parse_group_label(root: Path, image_path: Path, layout: str):
    parts = image_path.relative_to(root).parts
    if len(parts) < 3:
        return None, None

    if layout == "group-label":
        group, label = parts[0], parts[1]
    else:
        label, group = parts[0], parts[1]

    return group.strip(), label.strip().lower()


def is_valid_label(label: str):
    return label in {"real", "fake"}


def maybe_relativize(path: Path, output_path: Path, enabled: bool):
    if not enabled:
        return str(path.resolve())
    return os.path.relpath(path.resolve(), start=output_path.parent.resolve())


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    rows = []
    skipped = 0

    for image_path in iter_images(root):
        group, label = parse_group_label(root, image_path, args.layout)
        if not group or not is_valid_label(label):
            skipped += 1
            continue

        cell_key = (group, label)
        current_count = counts.get(cell_key, 0)
        if args.max_per_cell is not None and current_count >= args.max_per_cell:
            continue

        counts[cell_key] = current_count + 1
        rows.append(
            {
                "id": str(len(rows) + 1),
                "image_path": maybe_relativize(image_path, output, args.relative_to_output),
                "label": label,
                "group": group,
            }
        )

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "label", "group"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest escrito em: {output}")
    print(f"amostras: {len(rows)}")
    print(f"ignoradas: {skipped}")
    for (group, label), count in sorted(counts.items()):
        print(f"{group} / {label}: {count}")


if __name__ == "__main__":
    main()
