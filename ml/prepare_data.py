"""Prepare the EcoSort dataset.

Sources
-------
* TrashNet (Gary Thung and Mindy Yang, MIT licence) for glass, paper,
  cardboard, plastic and metal. Its sixth "trash" folder is intentionally
  dropped because EcoSort has no mixed-trash class.
* The "biological" (food waste) subset of the public 12-class Garbage
  Classification dataset, used as the organic class.

The script validates every image, removes duplicates and corrupt files, and
creates a reproducible stratified 70/15/15 train/validation/test split.

Usage
-----
    python ml/prepare_data.py
    python ml/prepare_data.py --organic-source data/raw/my_organic_folder

If the data is missing, follow the manual download instructions printed by
the script. Nothing is ever generated or faked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TRASHNET_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic")
TARGET_CLASSES = ("plastic", "paper", "cardboard", "glass", "metal", "organic")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

TRASHNET_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/"
    "dataset-resized.zip"
)
ORGANIC_URL = (
    "https://huggingface.co/datasets/UdaraChamidu/"
    "Garbage-Classification-with-12-classes/resolve/main/"
    "garbage_classification.zip"
)


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def _valid_image(path: Path) -> bool:
    """Check that OpenCV can decode the file and it is big enough."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return (
            image is not None
            and image.shape[0] >= 32
            and image.shape[1] >= 32
        )
    except (OSError, cv2.error):
        return False


def _find_trashnet_dir(raw_dir: Path) -> Path | None:
    """Locate a folder that contains the five TrashNet class folders."""
    candidates = [raw_dir / "trashnet" / "dataset-resized", raw_dir / "dataset-resized"]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in TRASHNET_CLASSES):
            return candidate
    for folder in raw_dir.rglob("*"):
        if folder.is_dir() and all(
            (folder / name).is_dir() for name in TRASHNET_CLASSES
        ):
            return folder
    return None


def _find_organic_dir(raw_dir: Path, explicit: str | None = None) -> Path | None:
    """Locate the organic / biological image folder."""
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_dir() and _image_files(path):
            return path
        return None

    preferred = [
        raw_dir / "organic_waste" / "garbage_classification" / "biological",
        raw_dir / "organic_waste",
        raw_dir / "organic" / "organic_data",
        raw_dir / "organic",
    ]
    for candidate in preferred:
        if candidate.is_dir() and _image_files(candidate):
            return candidate
    for folder in raw_dir.rglob("*"):
        if folder.is_dir() and folder.name.lower() in {"biological", "organic_data"}:
            if _image_files(folder):
                return folder
    return None


def download_data() -> None:
    """Download the two public source archives (about 300 MB total)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    trashnet_zip = RAW_DIR / "trashnet.zip"
    organic_zip = RAW_DIR / "garbage12.zip"

    if not trashnet_zip.exists():
        print(f"Downloading TrashNet archive to {trashnet_zip} ...")
        urllib.request.urlretrieve(TRASHNET_URL, trashnet_zip)
    if not organic_zip.exists():
        print(f"Downloading organic/bio archive (large) to {organic_zip} ...")
        urllib.request.urlretrieve(ORGANIC_URL, organic_zip)

    if not (RAW_DIR / "trashnet" / "dataset-resized").exists():
        print("Extracting TrashNet ...")
        with zipfile.ZipFile(trashnet_zip) as archive:
            archive.extractall(RAW_DIR / "trashnet")
    organic_root = RAW_DIR / "organic_waste"
    if not _find_organic_dir(RAW_DIR):
        print("Extracting only the biological class from the organic archive ...")
        with zipfile.ZipFile(organic_zip) as archive:
            for name in archive.namelist():
                if (
                    "biological/" in name
                    and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    and "__MACOSX" not in name
                ):
                    archive.extract(name, organic_root)


def collect_images(
    trashnet_dir: Path, organic_dir: Path, max_organic: int | None = None
) -> tuple[dict[str, list[Path]], dict]:
    """Validate, de-duplicate and group images by target class."""
    grouped: dict[str, list[Path]] = {name: [] for name in TARGET_CLASSES}
    seen_hashes: set[str] = set()
    bad_files: list[str] = []
    duplicates = 0
    per_source = {}

    for name in TRASHNET_CLASSES:
        folder = trashnet_dir / name
        files = _image_files(folder)
        kept = 0
        for path in files:
            if not _valid_image(path):
                bad_files.append(str(path))
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(digest)
            grouped[name].append(path)
            kept += 1
        per_source[f"TrashNet/{name}"] = {"found": len(files), "kept": kept}
        print(f"  TrashNet/{name:<10} found {len(files):4d}  usable {kept:4d}")

    organic_files = _image_files(organic_dir)
    random.Random(SEED).shuffle(organic_files)
    if max_organic:
        organic_files = organic_files[:max_organic]
    kept_organic = 0
    for path in organic_files:
        if not _valid_image(path):
            bad_files.append(str(path))
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        grouped["organic"].append(path)
        kept_organic += 1
    per_source["Organic/biological"] = {
        "found": len(_image_files(organic_dir)),
        "kept": kept_organic,
    }
    print(f"  Organic (biological) found {len(_image_files(organic_dir)):4d}  usable {kept_organic:4d}")

    stats = {
        "bad_files_removed": len(bad_files),
        "duplicates_removed": duplicates,
        "per_source": per_source,
    }
    if bad_files:
        print(f"  Removed {len(bad_files)} corrupt/unreadable image(s).")
    if duplicates:
        print(f"  Removed {duplicates} duplicate image(s).")
    return grouped, stats


def split_and_copy(grouped: dict[str, list[Path]], output_dir: Path) -> dict:
    """Create a stratified 70/15/15 split and copy files with random names."""
    rng = random.Random(SEED)
    if not all(grouped[name] for name in TARGET_CLASSES):
        missing = [name for name in TARGET_CLASSES if not grouped[name]]
        raise SystemExit(f"Cannot continue: no usable images for {missing}.")

    split_counts = {
        split: {name: 0 for name in TARGET_CLASSES}
        for split in ("train", "val", "test")
    }
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for name in TARGET_CLASSES:
        files = grouped[name]
        labels = [name] * len(files)
        train_val, test = train_test_split(
            files, test_size=0.15, random_state=SEED, shuffle=True, stratify=labels
        )
        train, val = train_test_split(
            train_val,
            test_size=0.15 / 0.85,
            random_state=SEED,
            shuffle=True,
            stratify=[name] * len(train_val),
        )
        for split_name, split_files in (("train", train), ("val", val), ("test", test)):
            destination = output_dir / split_name / name
            destination.mkdir(parents=True, exist_ok=True)
            used_names: set[str] = set()
            for source in split_files:
                digest = hashlib.md5(source.read_bytes()).hexdigest()[:12]
                suffix = source.suffix.lower()
                filename = f"{name}_{digest}{suffix}"
                counter = 1
                while filename in used_names:
                    filename = f"{name}_{digest}_{counter}{suffix}"
                    counter += 1
                used_names.add(filename)
                shutil.copy2(source, destination / filename)
                split_counts[split_name][name] += 1
        assert len(train) + len(val) + len(test) == len(files)

    return split_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the EcoSort dataset.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the public archives if they are not present.",
    )
    parser.add_argument(
        "--organic-source",
        help="Path to an organic/bio image folder (must contain images).",
    )
    parser.add_argument(
        "--max-organic",
        type=int,
        default=None,
        help="Optional cap on the number of organic images.",
    )
    parser.add_argument(
        "--out",
        default=str(PROCESSED_DIR),
        help="Output directory for the split dataset.",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    if args.download:
        try:
            download_data()
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}", file=sys.stderr)
            print(
                "Download the files manually using the instructions below.",
                file=sys.stderr,
            )

    trashnet_dir = _find_trashnet_dir(RAW_DIR)
    organic_dir = _find_organic_dir(RAW_DIR, args.organic_source)
    if trashnet_dir is None or organic_dir is None:
        print("\nDataset files were not found.\n")
        print("Manual download steps:")
        print("  1. Download TrashNet (dataset-resized.zip):")
        print(f"     {TRASHNET_URL}")
        print("     Extract it to: data/raw/trashnet/dataset-resized/")
        print("     It must contain cardboard/, glass/, metal/, paper/, plastic/.")
        print("  2. Download the 12-class Garbage Classification archive:")
        print(f"     {ORGANIC_URL}")
        print("     Extract only the biological/ folder to:")
        print("     data/raw/organic_waste/garbage_classification/biological/")
        print("\nThen run: python ml/prepare_data.py")
        print("\nTrashNet's 'trash' folder is deliberately not used.\n")
        return 1

    print(f"TrashNet folder : {trashnet_dir}")
    print(f"Organic folder  : {organic_dir}")
    print("\nScanning and validating images ...")
    grouped, stats = collect_images(
        trashnet_dir, organic_dir, max_organic=args.max_organic
    )

    print("\nCreating a stratified 70/15/15 split ...")
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    split_counts = split_and_copy(grouped, output_dir)

    totals = {
        split: sum(counts.values()) for split, counts in split_counts.items()
    }
    total = sum(totals.values())
    summary = {
        "seed": SEED,
        "total_images": total,
        "splits": split_counts,
        "split_totals": totals,
        "class_totals": {
            name: sum(split_counts[s][name] for s in split_counts)
            for name in TARGET_CLASSES
        },
        "sources": {
            "TrashNet": "Gary Thung & Mindy Yang (MIT licence); 'trash' class excluded",
            "Organic": "12-class Garbage Classification dataset, 'biological' subset",
        },
        **stats,
    }
    (output_dir / "split_counts.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nClass counts")
    print(f"{'class':<12}{'train':>7}{'val':>7}{'test':>7}{'total':>8}")
    for name in TARGET_CLASSES:
        tr = split_counts["train"][name]
        va = split_counts["val"][name]
        te = split_counts["test"][name]
        print(f"{name:<12}{tr:>7}{va:>7}{te:>7}{tr + va + te:>8}")
    print(f"{'TOTAL':<12}{totals['train']:>7}{totals['val']:>7}{totals['test']:>7}{total:>8}")
    print(f"\nSaved split to: {output_dir}")
    print(f"Summary file  : {output_dir / 'split_counts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
