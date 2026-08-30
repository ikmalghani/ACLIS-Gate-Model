#!/usr/bin/env python3
"""
ACLIS Leaf+Pest Gate TFLite → on-device C

Targets:
  Leaf+Pest Gate Model/aclis_pest_gate_96x_full_int8.tflite

Uses emit_pest_gate_c_from_tflite.py (3-class: leaf / others / pest).
Keeps the `leaf_gate_*` C API and **replaces** ACLIS_IKMAL codegen_leaf_gate/
weights so main.cpp / .cproject stay compatible.

Usage:
  cd "Ikmal/Gate Model/Leaf+Pest Gate Model"
  ../../tinyengine/venv/bin/python codegen_pest_gate_c.py
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
EMIT = HERE / "emit_pest_gate_c_from_tflite.py"
TFLITE = HERE / "aclis_pest_gate_96x_full_int8.tflite"
PY = IKMAL / "tinyengine" / "venv" / "bin" / "python"


def main() -> None:
    print("=" * 70)
    print("ACLIS Leaf+Pest Gate codegen (3-class TFLite → C)")
    print("=" * 70)
    if not TFLITE.is_file():
        print(f"Missing {TFLITE}")
        print("  Place aclis_pest_gate_96x_full_int8.tflite in this folder first.")
        sys.exit(1)
    if not EMIT.is_file():
        print(f"Missing emitter: {EMIT}")
        sys.exit(1)

    print(f"TFLite : {TFLITE} ({TFLITE.stat().st_size / 1024:.1f} KB)")
    print(f"Emitter: {EMIT}")
    print("\nInstalls into ACLIS_IKMAL (overwrites current gate codegen):\n"
          "  ACLIS_IKMAL/Src/TinyEngine/codegen_leaf_gate/\n"
          "  ACLIS_IKMAL/Inc/leaf_gate_nn.h\n")

    py = str(PY if PY.is_file() else sys.executable)
    rc = subprocess.call(
        [
            py,
            str(EMIT),
            "--tflite",
            str(TFLITE),
            "--label",
            TFLITE.name,
        ]
    )
    if rc != 0:
        sys.exit(rc)

    print(
        """
Installed under:
  ACLIS_IKMAL/Src/TinyEngine/codegen_leaf_gate/
  ACLIS_IKMAL/Inc/leaf_gate_nn.h

Classes: leaf=0, others=1, pest=2
Cascade policy (main.cpp): disease CNN if leaf OR pest; skip if others.

NEXT (STM32CubeIDE):
  1. Refresh ACLIS_IKMAL (codegen_leaf_gate sources already in the 2-model build)
  2. Build & flash ACLIS_IKMAL
"""
    )


if __name__ == "__main__":
    main()
