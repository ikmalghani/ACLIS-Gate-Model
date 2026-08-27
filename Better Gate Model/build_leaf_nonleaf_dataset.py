#!/usr/bin/env python3
"""Build a balanced leaf / non-leaf dataset from PlantVillage.

Steps (in this order, as required):
  1. Collect every readable image into two pools: leaf/ and non-leaf/
  2. Undersample the majority class so both pools have the same count
  3. Only then split into train / val / test (stratified, equal per class)

Default source is the 6-class PlantVillage tree (disease classes = leaf,
`others` = non-leaf). The original PlantVillage train/val/test folders are
flattened on purpose so the new split is made from the balanced pools.

Usage:
  python build_leaf_nonleaf_dataset.py
  python build_leaf_nonleaf_dataset.py --source /path/to/plantvillage
  python build_leaf_nonleaf_dataset.py --train 0.70 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}

NON_LEAF_NAMES = {
    "others",
    "other",
    "non-leaf",
    "non_leaf",
    "not_leaf",
    "nonleaf",
    "noleaf",
    "no_leaf",
    "not-leaf",
    "background",
    "bg",
    "non-plant",
    "non_plant",
}

HERE = Path(__file__).resolve().parent
IKMAL = HERE.parent.parent
DEFAULT_OUT = HERE / "leaf_nonleaf_dataset"

DEFAULT_SOURCES = [
    IKMAL / "Better Plant Disease Classifier" / "6 Class Model" / "Plant Village Others Dataset",
    IKMAL / "Better Plant Disease Classifier" / "5 Class  Model" / "aclis_ready_plantvillage_dataset",
    IKMAL / "AATBS" / "aclis_ready_plantvillage_dataset",
    IKMAL.parent / "Aclis Documentation 26thJune2026" / "Aclis Documentation" / "aclis_ready_plantvillage_dataset",
    IKMAL / "Gate Model" / "Leaf+Pest Gate Model" / "leaf_pest_others_dataset",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a balanced leaf/non-leaf dataset from PlantVillage")
    p.add_argument("--source", type=Path, default=None,
                   help="PlantVillage root (class folders, or train/val/test/class)")
    p.add_argument("--non-leaf-source", type=Path, default=None,
                   help="Optional extra root used only for non-leaf images")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT,
                   help=f"Output dataset directory (default: {DEFAULT_OUT})")
    p.add_argument("--train", type=float, default=0.70, help="Train fraction")
    p.add_argument("--val", type=float, default=0.15, help="Val fraction")
    p.add_argument("--test", type=float, default=0.15, help="Test fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of hardlinking (uses more disk)")
    p.add_argument("--force", action="store_true",
                   help="Delete an existing output directory first")
    return p.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def is_readable(path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return True
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except OSError:
        return False


def canon_label(folder_name: str) -> str | None:
    key = folder_name.strip().lower().replace(" ", "_")
    if key in SPLIT_NAMES:
        return None
    if key in NON_LEAF_NAMES:
        return "non-leaf"
    return "leaf"


def iter_class_dirs(root: Path):
    """Yield (class_dir, original_class_name) under a PlantVillage-style tree."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    split_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.lower() in SPLIT_NAMES
    ]
    search_roots = split_dirs if split_dirs else [root]
    for search in search_roots:
        for class_dir in sorted(search.iterdir()):
            if class_dir.is_dir():
                yield class_dir, class_dir.name


def collect_images(
    root: Path, label_filter: str | None = None
) -> tuple[dict[str, list[Path]], int]:
    buckets: dict[str, list[Path]] = {"leaf": [], "non-leaf": []}
    skipped = 0
    for class_dir, class_name in iter_class_dirs(root):
        label = canon_label(class_name)
        if label is None:
            continue
        if label_filter and label != label_filter:
            continue
        for path in class_dir.rglob("*"):
            if not is_image(path):
                continue
            if not is_readable(path):
                skipped += 1
                continue
            buckets[label].append(path)
    return buckets, skipped


def place_file(src: Path, dest: Path, copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if copy:
        shutil.copy2(src, dest)
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def unique_dest(dest_dir: Path, src: Path, tag: str, used: set[str]) -> Path:
    stem = f"{tag}__{src.stem}"
    name = f"{stem}{src.suffix.lower()}"
    n = 1
    while name in used:
        name = f"{stem}__{n}{src.suffix.lower()}"
        n += 1
    used.add(name)
    return dest_dir / name


def pick_default_source() -> Path:
    for cand in DEFAULT_SOURCES:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "No PlantVillage dataset found. Pass --source explicitly.\n"
        "Looked in:\n  " + "\n  ".join(str(p) for p in DEFAULT_SOURCES)
    )


def main() -> None:
    args = parse_args()
    train_f, val_f, test_f = args.train, args.val, args.test
    total_f = train_f + val_f + test_f
    if abs(total_f - 1.0) > 1e-6:
        raise SystemExit(f"Split fractions must sum to 1.0, got {total_f:.4f}")
    if min(train_f, val_f, test_f) < 0:
        raise SystemExit("Split fractions must be non-negative")

    source = (args.source or pick_default_source()).resolve()
    out = args.output.resolve()
    rng = random.Random(args.seed)

    print("=" * 70, flush=True)
    print("Build balanced leaf / non-leaf dataset from PlantVillage", flush=True)
    print("=" * 70, flush=True)
    print(f"  source : {source}", flush=True)
    print(f"  output : {out}", flush=True)
    print(f"  split  : train={train_f:.2f}  val={val_f:.2f}  test={test_f:.2f}", flush=True)
    print(f"  seed   : {args.seed}", flush=True)

    print("\n1. Collecting images into leaf / non-leaf pools…", flush=True)
    found, skipped = collect_images(source)
    extra_nonleaf_src = None
    if args.non_leaf_source:
        extra_nonleaf_src = args.non_leaf_source.resolve()
        extra, extra_skipped = collect_images(extra_nonleaf_src, label_filter="non-leaf")
        skipped += extra_skipped
        found["non-leaf"].extend(extra["non-leaf"])
    elif not found["non-leaf"]:
        # 5-class PlantVillage has only leaves; pull `others` from the 6-class tree.
        for cand in DEFAULT_SOURCES:
            if cand.resolve() == source or not cand.is_dir():
                continue
            extra, extra_skipped = collect_images(cand, label_filter="non-leaf")
            n_extra = len(extra["non-leaf"])
            skipped += extra_skipped
            if n_extra:
                extra_nonleaf_src = cand.resolve()
                found["non-leaf"].extend(extra["non-leaf"])
                print(f"  no non-leaf in source — added {n_extra} from {cand}", flush=True)
                break

    # De-duplicate by resolved path
    for label in ("leaf", "non-leaf"):
        uniq = []
        seen = set()
        for p in found[label]:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        found[label] = uniq

    n_leaf_raw = len(found["leaf"])
    n_non_raw = len(found["non-leaf"])
    print(f"  leaf     : {n_leaf_raw:,}", flush=True)
    print(f"  non-leaf : {n_non_raw:,}", flush=True)
    if skipped:
        print(f"  skipped unreadable : {skipped:,}", flush=True)

    if n_leaf_raw == 0 or n_non_raw == 0:
        raise SystemExit(
            "Need images in both classes. "
            f"leaf={n_leaf_raw} non-leaf={n_non_raw}. "
            "Pass --non-leaf-source if PlantVillage has no others/non-leaf folder."
        )

    n_keep = min(n_leaf_raw, n_non_raw)
    print(f"\n2. Balancing pools to {n_keep:,} images per class…", flush=True)
    leaf_pool = list(found["leaf"])
    non_pool = list(found["non-leaf"])
    rng.shuffle(leaf_pool)
    rng.shuffle(non_pool)
    leaf_pool = leaf_pool[:n_keep]
    non_pool = non_pool[:n_keep]

    if out.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {out}\n  Re-run with --force to replace it.")
        print(f"  removing existing {out}", flush=True)
        shutil.rmtree(out)

    pool_leaf = out / "leaf"
    pool_non = out / "non-leaf"
    pool_leaf.mkdir(parents=True)
    pool_non.mkdir(parents=True)

    placed = {"leaf": [], "non-leaf": []}
    used_names = {"leaf": set(), "non-leaf": set()}
    src_class_counts = {"leaf": defaultdict(int), "non-leaf": defaultdict(int)}

    for src in leaf_pool:
        dest = unique_dest(pool_leaf, src, src.parent.name, used_names["leaf"])
        place_file(src, dest, args.copy)
        placed["leaf"].append(dest)
        src_class_counts["leaf"][src.parent.name] += 1
    for src in non_pool:
        dest = unique_dest(pool_non, src, src.parent.name, used_names["non-leaf"])
        place_file(src, dest, args.copy)
        placed["non-leaf"].append(dest)
        src_class_counts["non-leaf"][src.parent.name] += 1

    assert len(placed["leaf"]) == len(placed["non-leaf"]) == n_keep
    print(f"  wrote {n_keep:,} → {pool_leaf}", flush=True)
    print(f"  wrote {n_keep:,} → {pool_non}", flush=True)

    print("\n3. Train / val / test split from the balanced pools…", flush=True)
    n_train = int(round(n_keep * train_f))
    n_val = int(round(n_keep * val_f))
    if n_train + n_val > n_keep:
        n_val = max(0, n_keep - n_train)
    n_test = n_keep - n_train - n_val
    split_sizes = {"train": n_train, "val": n_val, "test": n_test}
    print(f"  per class: train={n_train:,}  val={n_val:,}  test={n_test:,}", flush=True)

    split_counts = {s: {"leaf": 0, "non-leaf": 0} for s in ("train", "val", "test")}
    for label, files in placed.items():
        files = list(files)
        rng.shuffle(files)
        chunks = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, split_files in chunks.items():
            dest_dir = out / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                dest = dest_dir / src.name
                place_file(src, dest, args.copy)
            split_counts[split][label] = len(split_files)

    for split in ("train", "val", "test"):
        n_l = split_counts[split]["leaf"]
        n_n = split_counts[split]["non-leaf"]
        if n_l != n_n:
            raise RuntimeError(f"{split} counts unequal: leaf={n_l} non-leaf={n_n}")
        print(f"  [{split}] leaf={n_l:,}  non-leaf={n_n:,}  total={n_l + n_n:,}", flush=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(source),
        "non_leaf_source": str(extra_nonleaf_src) if extra_nonleaf_src else None,
        "output": str(out),
        "seed": args.seed,
        "copy": bool(args.copy),
        "split_fractions": {"train": train_f, "val": val_f, "test": test_f},
        "raw_counts": {"leaf": n_leaf_raw, "non-leaf": n_non_raw},
        "balanced_per_class": n_keep,
        "skipped_unreadable": skipped,
        "source_class_counts": {
            k: dict(sorted(v.items(), key=lambda kv: kv[0].lower()))
            for k, v in src_class_counts.items()
        },
        "split_counts": split_counts,
        "classes": ["leaf", "non-leaf"],
        "layout": {
            "balanced_pools": ["leaf/", "non-leaf/"],
            "splits": "train|val|test/{leaf,non-leaf}/",
        },
    }
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  manifest → {man_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
