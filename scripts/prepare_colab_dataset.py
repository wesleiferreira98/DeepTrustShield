#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepara um dataset local no formato train/val/test para subir no Google Drive "
            "e treinar no Colab."
        )
    )
    parser.add_argument("--real-dir", required=True, help="Diretorio com imagens reais")
    parser.add_argument("--fake-dir", required=True, help="Diretorio com imagens fake")
    parser.add_argument("--output-root", required=True, help="Diretorio de saida")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proporcao de treino")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proporcao de validacao")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Proporcao de teste")
    parser.add_argument("--seed", type=int, default=42, help="Seed do embaralhamento")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limita a quantidade de imagens por classe antes do split",
    )
    parser.add_argument(
        "--balance-to-min-class",
        action="store_true",
        help="Limita cada classe ao tamanho da menor classe antes do split",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="copy duplica arquivos; hardlink economiza espaco no disco local",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove a saida anterior antes de gerar um novo dataset_colab",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"As proporcoes devem somar 1.0, mas somaram {total:.6f}")


def collect_images(root: Path):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_items(items, train_ratio, val_ratio):
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def materialize(items, destination_dir: Path, mode: str):
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(items, start=1):
        target_path = destination_dir / f"{index:06d}{source_path.suffix.lower()}"
        if mode == "hardlink":
            if target_path.exists():
                target_path.unlink()
            target_path.hardlink_to(source_path.resolve())
        else:
            shutil.copy2(source_path, target_path)


def prepare_class(source_dir: Path, output_root: Path, class_name: str, train_ratio, val_ratio, seed, max_per_class, mode):
    images = collect_images(source_dir)
    random.Random(seed).shuffle(images)
    if max_per_class is not None:
        images = images[:max_per_class]

    split_map = split_items(images, train_ratio, val_ratio)
    for split_name, split_items_list in split_map.items():
        materialize(split_items_list, output_root / split_name / class_name, mode)

    return {
        "class_name": class_name,
        "total": len(images),
        "train": len(split_map["train"]),
        "val": len(split_map["val"]),
        "test": len(split_map["test"]),
    }


def choose_class_limit(real_images, fake_images, max_per_class, balance_to_min_class):
    if not balance_to_min_class:
        return max_per_class

    min_count = min(len(real_images), len(fake_images))
    if max_per_class is None:
        return min_count
    return min(max_per_class, min_count)


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    real_dir = Path(args.real_dir).resolve()
    fake_dir = Path(args.fake_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if not real_dir.exists():
        raise FileNotFoundError(f"Diretorio real nao encontrado: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Diretorio fake nao encontrado: {fake_dir}")
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    real_images = collect_images(real_dir)
    fake_images = collect_images(fake_dir)
    class_limit = choose_class_limit(real_images, fake_images, args.max_per_class, args.balance_to_min_class)

    summary = [
        prepare_class(
            source_dir=real_dir,
            output_root=output_root,
            class_name="real",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_per_class=class_limit,
            mode=args.mode,
        ),
        prepare_class(
            source_dir=fake_dir,
            output_root=output_root,
            class_name="fake",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_per_class=class_limit,
            mode=args.mode,
        ),
    ]

    print(f"dataset preparado em: {output_root}")
    for item in summary:
        print(
            f"{item['class_name']}: total={item['total']} "
            f"train={item['train']} val={item['val']} test={item['test']}"
        )
    print("estrutura gerada:")
    print(output_root / "train" / "real")
    print(output_root / "train" / "fake")
    print(output_root / "val" / "real")
    print(output_root / "val" / "fake")
    print(output_root / "test" / "real")
    print(output_root / "test" / "fake")


if __name__ == "__main__":
    main()
