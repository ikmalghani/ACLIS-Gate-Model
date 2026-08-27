#!/usr/bin/env python3
"""
ACLIS Better Gate TFLite → on-device C

Targets the Better Gate Model alt-recipe export:
  leaf_gate_output_alt/aclis_leaf_gate_96x_alt_full_int8.tflite
  (or the same filename next to this script)

Uses emit_better_gate_c_from_tflite.py so the generated C API stays
compatible with ACLIS_IKMAL (`leaf_gate_invoke`, shared arena, etc.).
Running this **replaces** `codegen_leaf_gate/` with Better Gate weights —
re-run Leaf Gate or Leaf+Pest codegen to restore those models.

Classes: leaf=0, non-leaf=1 (LEAF_GATE_CLASS_NOT_LEAF / NON_LEAF).

Usage:
  cd "Ikmal/Gate Model/Better Gate Model"
  ../../tinyengine/venv/bin/python codegen_better_gate_c.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TFLITE_NAME = "aclis_leaf_gate_96x_alt_full_int8.tflite"


def find_ikmal(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "ACLIS_IKMAL").is_dir() and (p / "tinyengine").is_dir():
            return p
    raise FileNotFoundError(
        "Could not find Ikmal/ (folder containing ACLIS_IKMAL/ and tinyengine/). "
        "Clone TinyEngine next to ACLIS_IKMAL."
    )


def locate_tflite(here: Path) -> Path:
    for cand in (here / TFLITE_NAME, here / "leaf_gate_output_alt" / TFLITE_NAME):
        if cand.is_file():
            return cand
    return here / "leaf_gate_output_alt" / TFLITE_NAME


IKMAL = find_ikmal(HERE)
EMIT = HERE / "emit_better_gate_c_from_tflite.py"
TFLITE = locate_tflite(HERE)
PY = IKMAL / "tinyengine" / "venv" / "bin" / "python"


def main() -> None:
    print("=" * 70)
    print("ACLIS Better Gate codegen (leaf / non-leaf TFLite → C)")
    print("=" * 70)
    if not TFLITE.is_file():
        print(f"Missing {TFLITE}")
        print(f"  Place {TFLITE_NAME} in this folder or under leaf_gate_output_alt/.")
        sys.exit(1)
    if not EMIT.is_file():
        print(f"Missing emitter: {EMIT}")
        sys.exit(1)

    print(f"TFLite : {TFLITE} ({TFLITE.stat().st_size / 1024:.1f} KB)")
    print(f"Emitter: {EMIT}")
    print("\nInstalls into ACLIS_IKMAL (overwrites current gate codegen):")
    print("  ACLIS_IKMAL/Src/TinyEngine/codegen_leaf_gate/")
    print("  ACLIS_IKMAL/Inc/leaf_gate_nn.h")
    print("  Classes: leaf=0, non-leaf=1 (2-class; not the 3-class pest gate)\n")

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

Classes: leaf=0, non-leaf=1
Shared SRAM:
  aclis_shared_arena[266344]  (disease genModel.h patched to use it)

NOTE: firmware main.cpp cascade is 2-class leaf / not-leaf
(disease CNN if leaf; SKIP if not_leaf).

NEXT (STM32CubeIDE):
  1. Add include path: Src/TinyEngine/codegen_leaf_gate/Include
  2. Refresh project so codegen_leaf_gate/Source/*.c and
     codegen/Source/aclis_shared_arena.c are compiled
  3. Build & flash ACLIS_IKMAL
  4. Restore pest gate with Leaf+Pest codegen_pest_gate_c.py, or
     baseline with Leaf Gate codegen_leaf_gate_c.py
"""
    )


if __name__ == "__main__":
    main()
