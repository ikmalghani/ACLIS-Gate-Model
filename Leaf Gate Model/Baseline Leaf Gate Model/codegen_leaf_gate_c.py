#!/usr/bin/env python3
"""
ACLIS Leaf-Gate TFLite → on-device C (new version of codegen_c.py)

Primary path: emit_leaf_gate_c_from_tflite.py
  TinyEngine's GenerateSourceFilesFromTFlite fails on this TFLite because the
  Colab export has a broken Flatten/Reshape ([1,128,1,1] → [1,1]) before FC.
  The emitter reads weights from the TFLite and writes a namespaced leaf-gate
  runtime that shares the disease activation arena.

Usage:
  cd "Ikmal/Leaf Gate Model"
  ../tinyengine/venv/bin/python codegen_leaf_gate_c.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_ikmal(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "ACLIS_IKMAL").is_dir() and (p / "tinyengine").is_dir():
            return p
    raise FileNotFoundError(
        "Could not find Ikmal/ (folder containing ACLIS_IKMAL/ and tinyengine/)."
    )


IKMAL = find_ikmal(HERE)
EMIT = HERE / "emit_leaf_gate_c_from_tflite.py"
TFLITE = HERE / "aclis_leaf_gate_96x_full_int8.tflite"
PY = IKMAL / "tinyengine" / "venv" / "bin" / "python"


def main() -> None:
    print("=" * 70)
    print("ACLIS Leaf-Gate codegen (TinyEngine-compatible dual-model deploy)")
    print("=" * 70)
    if not TFLITE.is_file():
        print(f"✗ Missing {TFLITE}")
        sys.exit(1)
    print(f"✓ TFLite: {TFLITE} ({TFLITE.stat().st_size/1024:.1f} KB)")

    # Optional: try official TinyEngine first (will fail on broken FC reshape)
    print("\nNote: official TinyEngine codegen is skipped for this model")
    print("      (broken Flatten/Reshape in TFLite). Using weight emitter instead.\n")

    py = str(PY if PY.is_file() else sys.executable)
    rc = subprocess.call([py, str(EMIT)])
    if rc != 0:
        sys.exit(rc)

    print(
        """
Installed under:
  ACLIS_IKMAL/Src/TinyEngine/codegen_leaf_gate/
  ACLIS_IKMAL/Inc/leaf_gate_nn.h

Shared SRAM:
  aclis_shared_arena[266344]  (disease genModel.h patched to use it)

NEXT (STM32CubeIDE):
  1. Add include path: Src/TinyEngine/codegen_leaf_gate/Include
  2. Refresh project so codegen_leaf_gate/Source/*.c and
     codegen/Source/aclis_shared_arena.c are compiled
  3. Build & flash ACLIS_IKMAL
"""
    )


if __name__ == "__main__":
    main()
