#!/usr/bin/env python3
"""
ACLIS Alt Leaf-Gate TFLite → on-device C

Targets the alternative training export:
  Alt Leaf Gate Model/aclis_leaf_gate_96x_alt_full_int8.tflite

Uses the same emitter as the baseline gate
(`../emit_leaf_gate_c_from_tflite.py`) so the generated C API stays
compatible with `ACLIS_IKMAL` (`leaf_gate_invoke`, shared arena, etc.).
Running this **replaces** `codegen_leaf_gate/` with the alt weights —
re-run `../codegen_leaf_gate_c.py` to restore the baseline.

Usage:
  cd "Ikmal/Leaf Gate Model/Alt Leaf Gate Model"
  ../../tinyengine/venv/bin/python codegen_leaf_gate_alt_c.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEAF_GATE = HERE.parent
IKMAL = LEAF_GATE.parent
EMIT = LEAF_GATE / "emit_leaf_gate_c_from_tflite.py"
TFLITE = HERE / "aclis_leaf_gate_96x_alt_full_int8.tflite"
PY = IKMAL / "tinyengine" / "venv" / "bin" / "python"


def main() -> None:
    print("=" * 70)
    print("ACLIS Alt Leaf-Gate codegen (distill/QAT TFLite → C)")
    print("=" * 70)
    if not TFLITE.is_file():
        print(f"✗ Missing {TFLITE}")
        print("  Place aclis_leaf_gate_96x_alt_full_int8.tflite in this folder first.")
        sys.exit(1)
    if not EMIT.is_file():
        print(f"✗ Missing emitter: {EMIT}")
        sys.exit(1)

    print(f"✓ TFLite : {TFLITE} ({TFLITE.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Emitter: {EMIT}")
    print("\nNote: installs into the same leaf_gate C tree as baseline")
    print("      (overwrites ACLIS_IKMAL/.../codegen_leaf_gate/).\n")

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

Shared SRAM:
  aclis_shared_arena[266344]  (disease genModel.h patched to use it)

NEXT (STM32CubeIDE):
  1. Add include path: Src/TinyEngine/codegen_leaf_gate/Include
  2. Refresh project so codegen_leaf_gate/Source/*.c and
     codegen/Source/aclis_shared_arena.c are compiled
  3. Build & flash ACLIS_IKMAL
  4. Compare on-device vs baseline (re-run ../codegen_leaf_gate_c.py to switch back)
"""
    )


if __name__ == "__main__":
    main()
