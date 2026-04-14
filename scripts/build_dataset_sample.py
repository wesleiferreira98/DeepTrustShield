#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
LABELS = ("real", "fake")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cria uma amostra balanceada a partir de um dataset local train/val/test real/fake."
    )
    parser.add_argument("--input-root", required=True, help="Dataset local de entrada, ex.: data/dataset_colab_v8")
    parser.add_argument("--output-root", required=True, help="Dataset amostrado de saida")
    parser.add_argument("--max-per-class", type=int, default=1000, help="Total maximo por classe em todos os splits")
    parser.add_argument("--seed", type=int, default=42, help="Seed para amostragem")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink economiza disco; copy duplica arquivos",
    )
    parser.add_argument("--clean", action="store_true", help="Remove a saida antes de gerar")
    return parser.parse_args()


def collect_images(input_root: Path, label: str):
    rows = []
    for split in SPLITS:
        label_dir = input_root / split / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((split, path))
    return rows


def materialize(source_path: Path, target_path: Path, mode: str):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    if mode == "hardlink":
        target_path.hardlink_to(source_path.resolve())
    else:
        shutil.copy2(source_path, target_path)


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Dataset de entrada nao encontrado: {input_root}")
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    rng = random.Random(args.seed)
    summaries = []

    for label in LABELS:
        items = collect_images(input_root, label)
        rng.shuffle(items)
        sampled = items[: args.max_per_class]

        split_counters = {split: 0 for split in SPLITS}
        for split, source_path in sampled:
            split_counters[split] += 1
            target_path = (
                output_root
                / split
                / label
                / f"{split_counters[split]:06d}{source_path.suffix.lower()}"
            )
            materialize(source_path, target_path, args.mode)

        summaries.append((label, len(items), len(sampled), split_counters))

    print(f"amostra criada em: {output_root}")
    for label, total, sampled_count, split_counters in summaries:
        print(
            f"{label}: total_origem={total} amostrado={sampled_count} "
            f"train={split_counters['train']} val={split_counters['val']} test={split_counters['test']}"
        )


if __name__ == "__main__":
    main()
