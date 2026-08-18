#!/usr/bin/env python3
"""Host/MCU leaf-gate parity probe: grayscale ramp → ImageNet INT8 → TFLite.

Matches firmware leaf_gate_fill_grayscale_ramp_input() in main.cpp
(when built with -DLEAF_GATE_RAMP_TEST). Compare printed int8 logits exactly
before blaming the model for camera failures.

Usage:
  python verify_leaf_gate_ramp.py
  python verify_leaf_gate_ramp.py --tflite path/to/aclis_leaf_gate_96x_full_int8.tflite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TFLITE = HERE / "aclis_leaf_gate_96x_full_int8.tflite"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Fallback if interpreter metadata unavailable (from TRAINING_RESULTS.md)
FALLBACK_SCALE = 0.018456979
FALLBACK_ZP = -16
RES = 96


def imagenet_requant_u8(u8: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """u8 [H,W,3] 0..255 → int8 via ImageNet normalize + TFLite affine.

    Rounding matches firmware: (int)(xf/scale + (xf>=0 ? 0.5f : -0.5f)) + zp
    (C cast truncates toward zero).
    """
    xf = (u8.astype(np.float32) / 255.0 - MEAN) / STD
    v = xf / scale + np.where(xf >= 0.0, 0.5, -0.5)
    q = np.trunc(v).astype(np.int32) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def make_grayscale_ramp() -> np.ndarray:
    """RGB = (y*96+x) % 256 — same as firmware LEAF_GATE_RAMP_TEST."""
    idx = (np.arange(RES * RES, dtype=np.int32) % 256).reshape(RES, RES)
    return np.stack([idx, idx, idx], axis=-1).astype(np.uint8)


def run_tflite(path: Path, x_int8: np.ndarray) -> tuple[np.ndarray, float, int, float, int]:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    if in_scale == 0:
        in_scale, in_zp = FALLBACK_SCALE, FALLBACK_ZP

    # Re-quantize with model input params if caller used fallbacks differently
    u8 = make_grayscale_ramp()
    x = imagenet_requant_u8(u8, float(in_scale), int(in_zp))
    interp.set_tensor(inp["index"], x[None, ...])
    interp.invoke()
    y = interp.get_tensor(out["index"]).reshape(-1)
    return y, float(in_scale), int(in_zp), float(out_scale), int(out_zp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    args = ap.parse_args()
    if not args.tflite.is_file():
        raise SystemExit(f"Missing TFLite: {args.tflite}")

    u8 = make_grayscale_ramp()
    # Preview quant with documented scale (firmware constants)
    preview = imagenet_requant_u8(u8, FALLBACK_SCALE, FALLBACK_ZP)

    y, in_s, in_z, out_s, out_z = run_tflite(args.tflite, preview)

    print("Leaf gate grayscale-ramp parity reference")
    print(f"  model     : {args.tflite}")
    print(f"  input     : scale={in_s:.8g}  zp={in_z}")
    print(f"  output    : scale={out_s:.8g}  zp={out_z}")
    print(f"  ramp u8   : min={u8.min()} max={u8.max()} shape={u8.shape}")
    print(f"  int8 in   : min={preview.min()} max={preview.max()} "
          f"(firmware leaf_scale/zp)")
    print(f"  int8 out  : leaf={int(y[0])}  not_leaf={int(y[1])}")
    print()
    print("On MCU (build with -DLEAF_GATE_RAMP_TEST): LCD must show the same "
          "leaf=/not_leaf= int8 scores before blaming camera preprocessing.")


if __name__ == "__main__":
    main()
