#!/usr/bin/env python3
"""Write STM32 .RAW files quantized for the Better Gate INT8 TFLite.

Same idea as the 5-class On-Board Inference/raw dump: bytes go straight
into leaf_gate_getInput() with no extra requantize on the MCU.

  resize 96×96 → /255 → ImageNet mean/std → clip(x/scale + zp)

Filenames keep a 2-letter prefix so firmware parse_gt() works
(LE = leaf, NL = not_leaf). FAT 8.3: LE0001.RAW …

The 5-class 176×176 BA/FU/… RAWs are a different size and a different
input scale. Do not mix them on this card.

Run (bettergatemodelenv), then copy raw/ into a folder named RAW on the SD card:

  cd "Ikmal/Gate Model/Better Gate Model/On-Board Inference"
  python -u export_better_gate_stm32_raw.py
  python -u export_better_gate_stm32_raw.py --limit 0   # full test split
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

HERE = Path(__file__).resolve().parent
GATE_DIR = HERE.parent  # Better Gate Model/ (script lives in On-Board Inference/)
OUTPUT_DIR = GATE_DIR / "leaf_gate_output_alt"
TFLITE_PATH = OUTPUT_DIR / "aclis_leaf_gate_96x_alt_full_int8.tflite"
DATASET_TEST = GATE_DIR / "leaf_nonleaf_dataset" / "test"
RAW_DIR = HERE / "raw"

CLASSES = ["leaf", "not_leaf"]
# Folder name in the gate dataset vs RAW prefix / firmware class name.
FOLDER = {"leaf": "leaf", "not_leaf": "non-leaf"}
PREFIX = {"leaf": "LE", "not_leaf": "NL"}
IMAGE_SIZE = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _make_interpreter(tflite: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
        interp = LiteRTInterpreter(model_path=str(tflite))
    except Exception:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(tflite))
    interp.allocate_tensors()
    return interp


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    out.sort(key=lambda p: p.name.lower())
    return out


def clear_prefix_raws(raw_dir: Path, prefix: str) -> None:
    for p in raw_dir.glob(f"{prefix}*.RAW"):
        p.unlink()
    for p in raw_dir.glob(f"{prefix}*.raw"):
        p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Better Gate STM32 .RAW files")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max images per class (default 200). 0 = use the full test split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tflite", type=Path, default=TFLITE_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_TEST)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    tflite = args.tflite.resolve()
    dataset = args.dataset.resolve()
    if not tflite.is_file():
        print(f"✗ Missing TFLite: {tflite}")
        sys.exit(1)
    if not dataset.is_dir():
        print(f"✗ Missing test set: {dataset}")
        sys.exit(1)

    interp = _make_interpreter(tflite)
    inp = interp.get_input_details()[0]
    scale, zp = inp["quantization"]
    assert scale > 0, "INT8 input needs a quantization scale"
    ishape = inp["shape"]
    # NHWC [1,H,W,3]
    h = int(ishape[1]) if len(ishape) > 1 else IMAGE_SIZE
    w = int(ishape[2]) if len(ishape) > 2 else IMAGE_SIZE
    if h != IMAGE_SIZE or w != IMAGE_SIZE:
        print(f"✗ Unexpected input HxW {h}x{w} (expected {IMAGE_SIZE})")
        sys.exit(1)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    rng = random.Random(args.seed)

    print(f"TFLite : {tflite}")
    print(f"In quant: scale={scale:.8g}  zp={zp}")
    print(f"Dataset: {dataset}")
    print(f"Out dir : {RAW_DIR}")
    print(f"Firmware expects {IMAGE_SIZE}×{IMAGE_SIZE}×3 int8 "
          f"({IMAGE_SIZE * IMAGE_SIZE * 3} bytes).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for cls in CLASSES:
        src_dir = dataset / FOLDER[cls]
        files = list_images(src_dir)
        if not files:
            print(f"✗ no images in {src_dir}")
            sys.exit(1)
        rng.shuffle(files)
        if args.limit > 0:
            files = files[: args.limit]
        clear_prefix_raws(RAW_DIR, PREFIX[cls])
        n_ok = 0
        for src in files:
            try:
                img = Image.open(src).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            except (OSError, ValueError) as e:
                skipped += 1
                print(f"  skip {src.name} ({e})")
                continue
            x = np.asarray(img, dtype=np.float32) / 255.0
            x = (x - mean) / std
            q = np.clip(x / float(scale) + float(zp), -128, 127).astype(np.int8)
            n_ok += 1
            out = RAW_DIR / f"{PREFIX[cls]}{n_ok:04d}.RAW"
            out.write_bytes(q.tobytes())
            written += 1
        print(f"  {cls:<10} {n_ok:>4} / {len(files)} listed → {PREFIX[cls]}####.RAW")

    print(f"✓ {written} files  ({IMAGE_SIZE}×{IMAGE_SIZE}×3 int8)")
    if skipped:
        print(f"  skipped {skipped} unreadable JPEG(s) (not written as .RAW)")
    print("Copy the contents of raw/ into SD card folder RAW/ (not the root).")


if __name__ == "__main__":
    main()
