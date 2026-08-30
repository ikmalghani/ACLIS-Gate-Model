"""
Emit a namespaced leaf/pest-gate TinyEngine-style C model from
aclis_pest_gate_96x_full_int8.tflite (3-class: leaf / others / pest).

Same approach as the binary leaf-gate emitter: TFLite Flatten/Reshape is
broken for TinyEngine, so this writes an int8→float reference runtime that
links beside the disease MCUNet model.

Keeps the `leaf_gate_*` C API so ACLIS main.cpp / .cproject stay compatible,
but installs 3-class weights + macros into ACLIS.

Output:
  ACLIS_IKMAL/Src/TinyEngine/codegen_leaf_gate/
  ACLIS_IKMAL/Inc/leaf_gate_nn.h

Run:
  ../../tinyengine/venv/bin/python emit_pest_gate_c_from_tflite.py
  # or: ../../tinyengine/venv/bin/python codegen_pest_gate_c.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find_ikmal(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "ACLIS_IKMAL").is_dir() and (p / "tinyengine").is_dir():
            return p
    raise FileNotFoundError(
        "Could not find Ikmal/ (folder containing ACLIS_IKMAL/ and tinyengine/)."
    )


IKMAL = find_ikmal(HERE)
TE = IKMAL / "tinyengine"
TFLITE = HERE / "aclis_pest_gate_96x_full_int8.tflite"
# Replace current gate codegen in ACLIS_IKMAL (same leaf_gate_* API).
OUT = IKMAL / "ACLIS_IKMAL" / "Src" / "TinyEngine" / "codegen_leaf_gate"
ACLIS_INC = IKMAL / "ACLIS_IKMAL" / "Inc"
PREFIX = "leaf_gate_"
# Short label embedded in generated C comments (override via --tflite / --label).
TFLITE_LABEL = "aclis_pest_gate_96x_full_int8.tflite"

sys.path.insert(0, str(TE))
from code_generator.tflite import Model  # noqa: E402
from code_generator.converters.tflite_parser.utils import getOpCodeStr  # noqa: E402

TYPE_MAP = {0: np.float32, 2: np.int32, 3: np.uint8, 9: np.int8}


def load_model(path: Path):
    raw = path.read_bytes()
    model = Model.Model.GetRootAsModel(raw, 0)
    return model, model.Subgraphs(0)


def tensor_np(model, g, ti):
    t = g.Tensors(ti)
    b = model.Buffers(t.Buffer())
    shape = list(t.ShapeAsNumpy()) if t.ShapeLength() else []
    if b.DataLength() == 0:
        return None, shape, t
    arr = np.frombuffer(bytes(b.DataAsNumpy()), dtype=TYPE_MAP[t.Type()])
    if shape and int(np.prod(shape)) == arr.size:
        arr = arr.reshape(shape)
    return arr, shape, t


def qparams(t):
    q = t.Quantization()
    if q is None:
        return None, None
    scale = q.ScaleAsNumpy() if q.ScaleLength() else None
    zp = q.ZeroPointAsNumpy() if q.ZeroPointLength() else None
    if scale is not None and len(scale) == 1:
        scale = float(scale[0])
    if zp is not None and len(zp) == 1:
        zp = int(zp[0])
    return scale, zp


def c_array(name: str, arr: np.ndarray, ctype: str) -> str:
    flat = arr.astype(np.float32 if "float" in ctype else arr.dtype).reshape(-1)
    if "float" in ctype:
        body = ", ".join(f"{float(x):.8g}f" for x in flat)
    elif ctype.startswith("int8") or ctype.startswith("signed char"):
        body = ", ".join(str(int(x)) for x in flat)
    else:
        body = ", ".join(str(int(x)) for x in flat)
    return f"static const {ctype} {name}[{flat.size}] = {{{body}}};\n"


def emit() -> None:
    assert TFLITE.is_file(), TFLITE
    model, g = load_model(TFLITE)

    # Collect op list with tensors
    ops = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        code = getOpCodeStr(op, model)
        inputs = [op.Inputs(k) for k in range(op.InputsLength()) if op.Inputs(k) >= 0]
        outputs = [op.Outputs(k) for k in range(op.OutputsLength()) if op.Outputs(k) >= 0]
        ops.append((code, inputs, outputs))

    # Read the graph-declared input/output tensors instead of assuming
    # fixed tensor indices from an older export layout.
    if g.InputsLength() < 1 or g.OutputsLength() < 1:
        raise ValueError("TFLite subgraph is missing declared inputs/outputs")
    input_ti = g.Inputs(0)
    output_ti = g.Outputs(0)
    _, _, tin = tensor_np(model, g, input_ti)
    in_scale, in_zp = qparams(tin)
    _, out_shape_decl, tout = tensor_np(model, g, output_ti)
    out_scale, out_zp = qparams(tout)
    # Prefer declared output shape; fall back to last classifier out channels.
    if out_shape_decl:
        num_classes = int(out_shape_decl[-1]) if len(out_shape_decl) >= 1 else 3
    else:
        num_classes = 3
    if num_classes < 2:
        raise ValueError(f"Unexpected NUM_CLASSES={num_classes} from {TFLITE}")

    # Peak arena: largest activation feature map (NHWC int8) + scratch floats
    # 98*98*3 padded input worst early, or 48*48*32, etc. Use 96*96*3*2 + headroom
    peak = 96 * 96 * 3 + 50 * 50 * 32 + 128 * 4 + 4096
    # More tightly: max of known maps
    peak = max(
        98 * 98 * 3,
        50 * 50 * 32,
        26 * 26 * 48,
        14 * 14 * 64,
        6 * 6 * 128,
        48 * 48 * 32,
        24 * 24 * 48,
        12 * 12 * 64,
    )
    # Two int8 maps + float scratch for one spatial plane
    arena = peak * 2 + 128 * 4 + 2048

    # Extract layer params in order (skip PAD/TRANSPOSE/RESHAPE)
    layers = []
    for code, inputs, outputs in ops:
        if code in ("PAD", "TRANSPOSE", "RESHAPE"):
            continue
        if code == "CONV_2D":
            w, ws, wt = tensor_np(model, g, inputs[1])
            b, _, bt = tensor_np(model, g, inputs[2])
            w_s, _ = qparams(wt)
            b_s, _ = qparams(bt)
            _, ishape, it = tensor_np(model, g, inputs[0])
            _, oshape, ot = tensor_np(model, g, outputs[0])
            i_s, i_z = qparams(it)
            o_s, o_z = qparams(ot)
            layers.append(
                dict(
                    op="CONV",
                    w=w.astype(np.int8),
                    b=b.astype(np.int32),
                    w_scale=np.array(w_s, dtype=np.float32),
                    b_scale=np.array(b_s, dtype=np.float32),
                    in_shape=ishape,
                    out_shape=oshape,
                    in_scale=float(i_s),
                    in_zp=int(i_z),
                    out_scale=float(o_s),
                    out_zp=int(o_z),
                )
            )
        elif code == "DEPTHWISE_CONV_2D":
            w, ws, wt = tensor_np(model, g, inputs[1])
            b, _, bt = tensor_np(model, g, inputs[2])
            w_s, _ = qparams(wt)
            b_s, _ = qparams(bt)
            _, ishape, it = tensor_np(model, g, inputs[0])
            _, oshape, ot = tensor_np(model, g, outputs[0])
            i_s, i_z = qparams(it)
            o_s, o_z = qparams(ot)
            layers.append(
                dict(
                    op="DW",
                    w=w.astype(np.int8),
                    b=b.astype(np.int32),
                    w_scale=np.array(w_s, dtype=np.float32),
                    b_scale=np.array(b_s, dtype=np.float32),
                    in_shape=ishape,
                    out_shape=oshape,
                    in_scale=float(i_s),
                    in_zp=int(i_z),
                    out_scale=float(o_s),
                    out_zp=int(o_z),
                )
            )
        elif code == "MEAN":
            _, ishape, it = tensor_np(model, g, inputs[0])
            _, oshape, ot = tensor_np(model, g, outputs[0])
            i_s, i_z = qparams(it)
            o_s, o_z = qparams(ot)
            layers.append(
                dict(
                    op="MEAN",
                    in_shape=ishape,
                    out_shape=oshape,
                    in_scale=float(i_s),
                    in_zp=int(i_z),
                    out_scale=float(o_s),
                    out_zp=int(o_z),
                )
            )
        elif code == "FULLY_CONNECTED":
            w, _, wt = tensor_np(model, g, inputs[1])  # [2,128]
            b, _, bt = tensor_np(model, g, inputs[2])
            w_s, _ = qparams(wt)
            b_s, _ = qparams(bt)
            # Force correct input as 128-vector after MEAN
            _, oshape, ot = tensor_np(model, g, outputs[0])
            o_s, o_z = qparams(ot)
            # Use MEAN output qparams as FC input
            mean_layer = [L for L in layers if L["op"] == "MEAN"][-1]
            layers.append(
                dict(
                    op="FC",
                    w=w.astype(np.int8),
                    b=b.astype(np.int32),
                    w_scale=np.array(w_s, dtype=np.float32),
                    b_scale=np.array(b_s, dtype=np.float32),
                    in_dim=w.shape[1],
                    out_dim=w.shape[0],
                    in_scale=mean_layer["out_scale"],
                    in_zp=mean_layer["out_zp"],
                    out_scale=float(o_s),
                    out_zp=int(o_z),
                )
            )

    # Prefer classifier out channels if declared graph shape was ambiguous.
    for L in reversed(layers):
        if L["op"] == "FC":
            num_classes = int(L["out_dim"])
            break
        if L["op"] == "CONV" and int(L["out_shape"][1]) == 1 and int(L["out_shape"][2]) == 1:
            num_classes = int(L["out_shape"][3])
            break

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Include").mkdir(exist_ok=True)
    (OUT / "Source").mkdir(exist_ok=True)

    # ---- header ----
    h = []
    h.append(f"/* Auto-generated leaf/pest-gate model from {TFLITE_LABEL} */\n")
    h.append("#ifndef LEAF_GATE_GENMODEL_H_\n#define LEAF_GATE_GENMODEL_H_\n")
    h.append("#include <stdint.h>\n")
    h.append(f"#define {PREFIX}PEAK_MEM {arena}\n")
    h.append(f"#define {PREFIX}RES 96\n")
    h.append(f"#define {PREFIX}NUM_CLASSES {num_classes}\n")
    h.append(f"#define LEAF_GATE_CLASS_LEAF 0\n")
    h.append(f"#define LEAF_GATE_CLASS_OTHERS 1\n")
    h.append(f"#define LEAF_GATE_CLASS_PEST 2\n")
    h.append(f"#define LEAF_GATE_NUM_CLASSES {num_classes}\n")
    h.append(f"#define LEAF_GATE_RES 96\n")
    h.append(f"#define LEAF_GATE_OUT_SCALE {out_scale:.8g}f\n")
    h.append(f"#define LEAF_GATE_OUT_ZP {int(out_zp)}\n")
    h.append(f"signed char *{PREFIX}getInput(void);\n")
    h.append(f"signed char *{PREFIX}getOutput(void);\n")
    h.append(f"void {PREFIX}invoke(float *labels);\n")
    h.append("#endif\n")
    (OUT / "Include" / f"{PREFIX}genModel.h").write_text("".join(h))

    nn = f"""#ifndef ACLIS_LEAF_GATE_NN_H_
#define ACLIS_LEAF_GATE_NN_H_
#include "{PREFIX}genModel.h"
#endif
"""
    (OUT / "Include" / f"{PREFIX}nn.h").write_text(nn)
    ACLIS_INC.mkdir(parents=True, exist_ok=True)
    (ACLIS_INC / "leaf_gate_nn.h").write_text(
        "/* ACLIS leaf / others / pest gate — public API for main.cpp */\n"
        "#ifndef ACLIS_LEAF_GATE_NN_H_\n"
        "#define ACLIS_LEAF_GATE_NN_H_\n"
        "#include <stdint.h>\n"
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "#define LEAF_GATE_CLASS_LEAF 0\n"
        "#define LEAF_GATE_CLASS_OTHERS 1\n"
        "#define LEAF_GATE_CLASS_PEST 2\n"
        f"#define LEAF_GATE_NUM_CLASSES {num_classes}\n"
        "#define LEAF_GATE_RES 96\n"
        f"#define LEAF_GATE_OUT_SCALE {out_scale:.8g}f\n"
        f"#define LEAF_GATE_OUT_ZP {int(out_zp)}\n"
        "signed char *leaf_gate_getInput(void);\n"
        "signed char *leaf_gate_getOutput(void);\n"
        "void leaf_gate_invoke(float *labels);\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
        "#endif\n"
    )

    # ---- source ----
    s = []
    s.append('/* Auto-generated leaf/pest-gate int8 reference runtime (float accum). */\n')
    s.append("#include <stdint.h>\n#include <string.h>\n")
    s.append(f'#include "{PREFIX}genModel.h"\n\n')
    s.append("/* Reuse disease activation arena (sequential invoke — do not overlap). */\n")
    s.append("extern signed char aclis_shared_arena[266344];\n")
    s.append(f"static signed char {PREFIX}output[{num_classes}];\n")
    s.append(f"signed char *{PREFIX}getInput(void) {{ return &aclis_shared_arena[0]; }}\n")
    s.append(f"signed char *{PREFIX}getOutput(void) {{ return {PREFIX}output; }}\n\n")

    # Emit weights
    for li, L in enumerate(layers):
        if L["op"] in ("CONV", "DW", "FC"):
            s.append(c_array(f"{PREFIX}w{li}", L["w"], "int8_t"))
            s.append(c_array(f"{PREFIX}b{li}", L["b"], "int32_t"))
            s.append(c_array(f"{PREFIX}ws{li}", L["w_scale"], "float"))
            s.append(c_array(f"{PREFIX}bs{li}", L["b_scale"], "float"))

    # Helpers
    s.append(
        r"""
static inline int8_t lg_quant(float x, float scale, int zp) {
  int v = (int)(x / scale + (x >= 0 ? 0.5f : -0.5f)) + zp;
  if (v > 127) v = 127;
  if (v < -128) v = -128;
  return (int8_t)v;
}

/* NHWC int8 conv: SAME or valid via explicit pad handled by caller sizing.
 * use_relu: 1 for hidden layers (ReLU6), 0 for final logits (no activation). */
static void lg_conv2d(
    const int8_t *in, int ih, int iw, int ic,
    const int8_t *w, const int32_t *b, const float *ws, const float *bs,
    int oc, int kh, int kw, int stride, int same_pad,
    int in_zp, float in_scale, int out_zp, float out_scale,
    int8_t *out, int oh, int ow, int use_relu)
{
  const int pad = same_pad ? (kh / 2) : 0;
  for (int y = 0; y < oh; y++) {
    for (int x = 0; x < ow; x++) {
      for (int o = 0; o < oc; o++) {
        float acc = (float)b[o] * bs[o];
        for (int ky = 0; ky < kh; ky++) {
          for (int kx = 0; kx < kw; kx++) {
            int iy = y * stride + ky - pad;
            int ix = x * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            for (int c = 0; c < ic; c++) {
              int8_t iv = in[(iy * iw + ix) * ic + c];
              int8_t wv = w[((o * kh + ky) * kw + kx) * ic + c];
              acc += (float)((int)iv - in_zp) * (float)wv * in_scale * ws[o];
            }
          }
        }
        if (use_relu) {
          if (acc < 0.f) acc = 0.f;
          if (acc > 6.f) acc = 6.f;
        }
        out[(y * ow + x) * oc + o] = lg_quant(acc, out_scale, out_zp);
      }
    }
  }
}

static void lg_depthwise(
    const int8_t *in, int ih, int iw, int ch,
    const int8_t *w, const int32_t *b, const float *ws, const float *bs,
    int kh, int kw, int stride, int same_pad,
    int in_zp, float in_scale, int out_zp, float out_scale,
    int8_t *out, int oh, int ow)
{
  const int pad = same_pad ? (kh / 2) : 0;
  for (int y = 0; y < oh; y++) {
    for (int x = 0; x < ow; x++) {
      for (int c = 0; c < ch; c++) {
        float acc = (float)b[c] * bs[c];
        for (int ky = 0; ky < kh; ky++) {
          for (int kx = 0; kx < kw; kx++) {
            int iy = y * stride + ky - pad;
            int ix = x * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            int8_t iv = in[(iy * iw + ix) * ch + c];
            int8_t wv = w[((ky * kw) + kx) * ch + c];
            acc += (float)((int)iv - in_zp) * (float)wv * in_scale * ws[c];
          }
        }
        if (acc < 0.f) acc = 0.f;
        if (acc > 6.f) acc = 6.f;
        out[(y * ow + x) * ch + c] = lg_quant(acc, out_scale, out_zp);
      }
    }
  }
}

static void lg_mean_spatial(
    const int8_t *in, int ih, int iw, int ch,
    int in_zp, float in_scale, int out_zp, float out_scale,
    int8_t *out /* [ch] stored as NHWC 1x1xch */)
{
  float inv = 1.0f / (float)(ih * iw);
  for (int c = 0; c < ch; c++) {
    float acc = 0.f;
    for (int i = 0; i < ih * iw; i++) {
      acc += (float)((int)in[i * ch + c] - in_zp) * in_scale;
    }
    acc *= inv;
    out[c] = lg_quant(acc, out_scale, out_zp);
  }
}

static void lg_fc(
    const int8_t *in, int in_dim,
    const int8_t *w, const int32_t *b, const float *ws, const float *bs,
    int out_dim, int in_zp, float in_scale, int out_zp, float out_scale,
    int8_t *out)
{
  /* w: [out_dim, in_dim] */
  for (int o = 0; o < out_dim; o++) {
    float acc = (float)b[o] * bs[o];
    for (int i = 0; i < in_dim; i++) {
      acc += (float)((int)in[i] - in_zp) * (float)w[o * in_dim + i] * in_scale * ws[o];
    }
    out[o] = lg_quant(acc, out_scale, out_zp);
  }
}

"""
    )

    # invoke: pad manually then run layers
    # Arena layout: A at 0, B at peak (within shared 266344)
    s.append(f"void {PREFIX}invoke(float *labels) {{\n")
    s.append(f"  (void)labels;\n")
    s.append(f"  int8_t *A = (int8_t *)&aclis_shared_arena[0];\n")
    s.append(f"  int8_t *B = (int8_t *)&aclis_shared_arena[{peak}];\n")
    s.append(f"  int8_t *cur = A;\n")
    s.append(f"  int8_t *nxt = B;\n")

    # First CONV expects padded 98x98 — pad from 96x96 input at A into a pad buffer
    s.append(
        f"""
  /* Explicit pad 96→98 (1 pixel each side), zp={in_zp} */
  {{
    const int ih=96, iw=96, ic=3, pad=1;
    int8_t *src = A; /* model input */
    int8_t *dst = B;
    for (int i=0;i<(ih+2*pad)*(iw+2*pad)*ic;i++) dst[i]=(int8_t){in_zp};
    for (int y=0;y<ih;y++)
      for (int x=0;x<iw;x++)
        for (int c=0;c<ic;c++)
          dst[((y+pad)*(iw+2*pad)+(x+pad))*ic+c] = src[(y*iw+x)*ic+c];
    cur = B; nxt = A;
  }}
"""
    )

    for li, L in enumerate(layers):
        if L["op"] == "CONV":
            ishape = L["in_shape"]
            oshape = L["out_shape"]
            ih, iw, ic = int(ishape[1]), int(ishape[2]), int(ishape[3])
            oh, ow, oc = int(oshape[1]), int(oshape[2]), int(oshape[3])
            kh = L["w"].shape[1]
            kw = L["w"].shape[2]
            stride = max(1, (ih - kh) // max(oh - 1, 1)) if oh > 1 else 1
            if oh * 2 <= ih:
                stride = 2
            elif oh == ih and kh == 1:
                stride = 1
            # VALID when input already explicitly padded ((oh-1)*stride+kh == ih)
            same = 0 if ((oh - 1) * stride + kh) <= ih else 1
            # Final 1x1 classifier must emit raw logits (no ReLU6).
            use_relu = 0 if (li == len(layers) - 1 and kh == 1 and kw == 1 and oh == 1 and ow == 1) else 1
            s.append(
                f"  /* CONV {li}: {ih}x{iw}x{ic} -> {oh}x{ow}x{oc} stride={stride} same={same} relu={use_relu} */\n"
            )
            s.append(
                f"  lg_conv2d(cur, {ih}, {iw}, {ic}, {PREFIX}w{li}, {PREFIX}b{li}, {PREFIX}ws{li}, {PREFIX}bs{li}, "
                f"{oc}, {kh}, {kw}, {stride}, {same}, {L['in_zp']}, {L['in_scale']:.8g}f, {L['out_zp']}, {L['out_scale']:.8g}f, "
                f"nxt, {oh}, {ow}, {use_relu});\n"
            )
            s.append("  { int8_t *tmp=cur; cur=nxt; nxt=tmp; }\n")
        elif L["op"] == "DW":
            ishape = L["in_shape"]
            oshape = L["out_shape"]
            ih, iw, ch = int(ishape[1]), int(ishape[2]), int(ishape[3])
            oh, ow = int(oshape[1]), int(oshape[2])
            kh, kw = L["w"].shape[1], L["w"].shape[2]
            stride = 2 if oh * 2 <= ih else 1
            s.append(f"  /* pad + DEPTHWISE {li}: graph in {ih}x{iw} stride={stride} */\n")
            prev = layers[li - 1] if li > 0 else None
            if prev and prev["op"] in ("CONV", "DW"):
                poh, pow_ = int(prev["out_shape"][1]), int(prev["out_shape"][2])
                if poh == ih - 2 and pow_ == iw - 2:
                    s.append(
                        f"  {{ int ph={ih}, pw={iw}, ch={ch}, zp={L['in_zp']};\n"
                        f"    for(int i=0;i<ph*pw*ch;i++) nxt[i]=(int8_t)zp;\n"
                        f"    for(int y=0;y<{poh};y++) for(int x=0;x<{pow_};x++) for(int c=0;c<ch;c++)\n"
                        f"      nxt[((y+1)*pw+(x+1))*ch+c]=cur[(y*{pow_}+x)*ch+c];\n"
                        f"    int8_t *tmp=cur; cur=nxt; nxt=tmp; }}\n"
                    )
            same = 0 if ((oh - 1) * stride + kh) <= ih else 1
            s.append(
                f"  lg_depthwise(cur, {ih}, {iw}, {ch}, {PREFIX}w{li}, {PREFIX}b{li}, {PREFIX}ws{li}, {PREFIX}bs{li}, "
                f"{kh}, {kw}, {stride}, {same}, {L['in_zp']}, {L['in_scale']:.8g}f, {L['out_zp']}, {L['out_scale']:.8g}f, "
                f"nxt, {oh}, {ow});\n"
            )
            s.append("  { int8_t *tmp=cur; cur=nxt; nxt=tmp; }\n")
        elif L["op"] == "MEAN":
            ishape = L["in_shape"]
            ih, iw, ch = int(ishape[1]), int(ishape[2]), int(ishape[3])
            s.append(f"  /* MEAN {li} over {ih}x{iw}x{ch} */\n")
            s.append(
                f"  lg_mean_spatial(cur, {ih}, {iw}, {ch}, {L['in_zp']}, {L['in_scale']:.8g}f, "
                f"{L['out_zp']}, {L['out_scale']:.8g}f, nxt);\n"
            )
            s.append("  { int8_t *tmp=cur; cur=nxt; nxt=tmp; }\n")
        elif L["op"] == "FC":
            s.append(f"  /* FC {li} {L['in_dim']} -> {L['out_dim']} (fixed vs broken TFLite flatten) */\n")
            s.append(
                f"  lg_fc(cur, {L['in_dim']}, {PREFIX}w{li}, {PREFIX}b{li}, {PREFIX}ws{li}, {PREFIX}bs{li}, "
                f"{L['out_dim']}, {L['in_zp']}, {L['in_scale']:.8g}f, {L['out_zp']}, {L['out_scale']:.8g}f, "
                f"{PREFIX}output);\n"
            )

    # Classifier head is a 1x1 CONV in the current export (not FC). Without this
    # copy, leaf_gate_output stays {0,0} and argmax always picks class 0 = leaf.
    if layers and layers[-1]["op"] != "FC":
        # Classifier head is 1x1 CONV → N logits (leaf/others/pest).
        for oi in range(num_classes):
            s.append(f"  {PREFIX}output[{oi}] = cur[{oi}];\n")

    s.append("}\n")
    (OUT / "Source" / f"{PREFIX}genModel.c").write_text("".join(s))

    # Update PEAK in header to shared note
    hpath = OUT / "Include" / f"{PREFIX}genModel.h"
    hpath.write_text(
        f"/* Auto-generated leaf/pest-gate model from {TFLITE_LABEL} */\n"
        "#ifndef LEAF_GATE_GENMODEL_H_\n#define LEAF_GATE_GENMODEL_H_\n"
        "#include <stdint.h>\n"
        f"#define {PREFIX}PEAK_MEM {peak}\n"
        f"#define {PREFIX}RES 96\n"
        f"#define {PREFIX}NUM_CLASSES {num_classes}\n"
        "#define LEAF_GATE_CLASS_LEAF 0\n"
        "#define LEAF_GATE_CLASS_OTHERS 1\n"
        "#define LEAF_GATE_CLASS_PEST 2\n"
        f"#define LEAF_GATE_NUM_CLASSES {num_classes}\n"
        "#define LEAF_GATE_RES 96\n"
        f"#define LEAF_GATE_OUT_SCALE {out_scale:.8g}f\n"
        f"#define LEAF_GATE_OUT_ZP {int(out_zp)}\n"
        f"signed char *{PREFIX}getInput(void);\n"
        f"signed char *{PREFIX}getOutput(void);\n"
        f"void {PREFIX}invoke(float *labels);\n"
        "#endif\n"
    )

    print(f"Emitted {OUT}")
    print(f"  Inc header → {ACLIS_INC / 'leaf_gate_nn.h'}")
    print(f"  NUM_CLASSES={num_classes}  (leaf=0 others=1 pest=2)")
    print(f"  uses shared arena aclis_shared_arena[266344]")
    print(f"  ping-pong half={peak} bytes")
    print(f"  layers={len(layers)}")
    print(f"  input zp={in_zp} scale={in_scale}")
    print(f"  output zp={out_zp} scale={out_scale}")


def _parse_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(
        description="Emit namespaced leaf/pest-gate C from an INT8 TFLite model."
    )
    p.add_argument(
        "--tflite",
        type=Path,
        default=None,
        help="Path to full-int8 TFLite (default: aclis_pest_gate_96x_full_int8.tflite next to this script)",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Comment label written into generated C (default: TFLite file name)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.tflite is not None:
        TFLITE = Path(args.tflite).resolve()
    TFLITE_LABEL = args.label or TFLITE.name
    emit()
