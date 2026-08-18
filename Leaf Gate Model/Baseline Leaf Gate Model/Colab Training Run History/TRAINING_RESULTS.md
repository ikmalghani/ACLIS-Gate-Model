# TinyLeafGate training results

Documented from Colab runs of `aclis_leaf_gate_train_tflite.ipynb` on **2026-07-28**.

| Run | Capture | Role |
|-----|---------|------|
| **Run 2 (current)** | [`run_2_2026-07-28.pdf`](run_2_2026-07-28.pdf) | BN ε + PT-matched padding, `RandomResizedCrop`, full INT8 test eval (§7b) — **use these numbers** |
| Run 1 (superseded) | [`run_1_2026-07-28.png`](run_1_2026-07-28.png) | First export; float PT↔Keras mismatch (~2–5 logits); INT8 test set **not** evaluated |

Numbers below are from **run 2** unless noted.

## Goal

Binary **leaf / not_leaf** gate (96×96) that runs on the STM32F746 before the 5-class PlantVillage disease CNN:

```
capture → leaf/not-leaf gate → if leaf: disease inference
                             → if not:  reject (“Not a leaf”)
```

## Environment

| Item | Value |
|------|--------|
| Notebook | `../aclis_leaf_gate_train_tflite.ipynb` |
| TensorFlow | 2.20.0 |
| ONNX | 1.17.0 |
| PyTorch | 2.10.0+cu128 |
| CUDA | available |
| Input size | 96×96 RGB |
| Classes | `leaf=0`, `not_leaf=1` |
| Batch size | 64 |
| Loss | weighted `CrossEntropyLoss` |
| Train aug | `RandomResizedCrop(96, scale=0.5–1.0)`, flip, rotate, color jitter, affine, RandomErasing; ImageNet normalize |
| Val / test | full-frame `Resize(96,96)` (no center crop) |

## Dataset (`leaf_noleaf_dataset.zip`)

Kaggle layout `Leaf v NonLeaf/{train,val,test}/{Leaf,Non_Leaf}/`, normalized to `{leaf,not_leaf}/`.

| Split | leaf | not_leaf | Total |
|-------|------|----------|-------|
| train | 9065 | 8066 | 17131 |
| val | 1942 | 1728 | 3670 |
| test | 1943 | 1730 | 3673 |

Class weights (train): leaf ≈ 0.945, not_leaf ≈ 1.062.

## Model — TinyLeafGate

Depthwise-separable CNN (stem + 4 blocks + GAP + 1×1 classifier). Keras twin uses `ZeroPadding2D(1)` + `valid` convs and BN `epsilon=1e-5` to match PyTorch.

| Metric | Value |
|--------|--------|
| Total params (Keras twin) | **28,854** (~112.3 KB float) |
| Trainable | 27,538 |
| Non-trainable (BN) | 1,216 |
| Float max\|PT−Keras\| (16 val imgs) | **mean=0.00000, max=0.00000** ✅ |

## Training (run 2)

Two-phase schedule (same pattern as the disease model).

### Phase 1 — warm-up

`epochs=5`, `lr=1e-3` (Adam), all layers.

| Epoch | train | val | leaf | not_leaf | note |
|------:|------:|----:|-----:|---------:|------|
| 1/5 | 0.875 | 0.968 | 0.991 | 0.943 | saved |
| 2/5 | 0.932 | 0.976 | 0.993 | 0.957 | saved |
| 3/5 | 0.941 | 0.983 | 0.993 | 0.972 | saved |
| 4/5 | 0.947 | 0.974 | 0.995 | 0.950 | saved |
| 5/5 | 0.948 | 0.980 | 0.991 | 0.966 | saved |

### Phase 2 — fine-tune

`epochs=25`, `lr=1e-4`, early stopping patience **8**. Best checkpoint used for test + export.

| Epoch | train | val | leaf | not_leaf | note |
|------:|------:|----:|-----:|---------:|------|
| 1/25 | 0.960 | 0.988 | 0.990 | 0.986 | patience 1/8 |
| 2/25 | 0.960 | 0.989 | 0.991 | 0.986 | saved |
| 3/25 | 0.961 | 0.987 | 0.991 | 0.987 | patience 1/8 |
| 4/25 | 0.963 | 0.988 | 0.992 | 0.984 | patience 2/8 |
| 5/25 | 0.963 | 0.988 | 0.993 | 0.981 | patience 3/8 |
| 6/25 | 0.964 | 0.989 | 0.994 | 0.984 | saved |
| 7/25 | 0.967 | 0.987 | 0.992 | 0.984 | patience 1/8 |
| 8/25 | 0.967 | 0.987 | 0.992 | 0.983 | patience 2/8 |
| 9/25 | 0.961 | 0.989 | 0.992 | 0.986 | patience 3/8 |
| 10/25 | 0.963 | 0.987 | 0.990 | 0.984 | patience 4/8 |
| 11/25 | 0.965 | 0.989 | 0.992 | 0.984 | patience 5/8 |
| 12/25 | 0.962 | 0.989 | 0.992 | 0.980 | patience 6/8 |
| 13/25 | 0.964 | 0.989 | 0.992 | 0.985 | patience 7/8 |
| 14/25 | 0.966 | 0.989 | 0.994 | 0.984 | saved |
| 15/25 | 0.966 | 0.990 | 0.994 | 0.985 | saved |
| 16/25 | 0.965 | 0.990 | 0.994 | 0.984 | patience 1/8 |
| 17/25 | 0.967 | 0.988 | 0.993 | 0.986 | patience 2/8 |
| 18/25 | 0.965 | 0.989 | 0.990 | 0.986 | patience 3/8 |
| 19/25 | 0.964 | 0.989 | 0.993 | 0.985 | patience 4/8 |
| 20/25 | 0.967 | 0.990 | 0.994 | 0.986 | saved |
| 21/25 | 0.967 | 0.991 | 0.993 | 0.989 | saved |
| 22/25 | 0.971 | 0.991 | 0.994 | 0.987 | patience 1/8 |
| 23/25 | 0.967 | 0.990 | 0.997 | 0.987 | patience 2/8 |
| 24/25 | 0.968 | 0.990 | 0.993 | 0.987 | patience 3/8 |
| 25/25 | 0.968 | 0.990 | 0.995 | 0.986 | patience 4/8 |

- **Completed 25/25** (no early stop)
- **Best val accuracy: 0.9913**

Checkpoint saved to Drive:  
`/content/drive/MyDrive/leaf_gate_output/aclis_leaf_gate_96x.pth`

## Test evaluation — PyTorch FP32 (§6)

| Metric | Value |
|--------|--------|
| **Test accuracy** | **0.9910 (99.1%)** |
| Target | ≥ 0.92 → **PASS** |
| leaf | 0.993 (99.3%) |
| not_leaf | 0.988 (98.8%) |

## Test evaluation — INT8 TFLite (§7b, deployed model)

Full test set via `tf.lite.Interpreter` (n=3673).

| Metric | Value |
|--------|--------|
| **INT8 accuracy** | **0.9913 (99.1%)** |
| PyTorch FP32 (same split) | 0.9910 (99.1%) |
| \|Δ\| accuracy | **0.0003 (0.03 pts)** → within 1% ✅ |
| Pred agreement (same argmax as PT) | 0.9901 (99.0%) |
| leaf | 0.994 (99.4%), n=1943 |
| not_leaf | 0.988 (98.8%), n=1730 |

## INT8 TFLite export

Export path: Keras twin with `GlobalAveragePooling2D(keepdims=True)` → `Conv2D(2, 1×1)` (TinyEngine-friendly; avoids broken Flatten/Reshape → FC). Explicit `PAD` ops from `ZeroPadding2D(1)`.

| Item | Value |
|------|--------|
| Artifact | `../aclis_leaf_gate_96x_full_int8.tflite` |
| Size | **50.9 KB** (52 128 bytes) |
| Input | `[1,96,96,3]` int8, scale ≈ **0.018457**, zp = **-16** |
| Output | `[1,2]` int8, scale ≈ **0.073172**, zp = **9** |
| Graph | PAD + CONV/DW + MEAN + CONV classifier; **no broken `[1,1]` reshape** |
| Smoke output (zeros) | `[-51, 61]` |

Drive copy:  
`/content/drive/MyDrive/leaf_gate_output/aclis_leaf_gate_96x_full_int8.tflite`

### Export fixes vs run 1

- BN: Keras `epsilon=1e-5`, `momentum=0.9` (≡ PyTorch `eps=1e-5`, `momentum=0.1`)
- Padding: `ZeroPadding2D(1)` + `valid` (not Keras `same`) to match PyTorch `padding=1`
- Abort if float max\|PT−Keras\| > `0.01` (run 2: **0.0**)
- Abort if \|INT8_acc − PT_acc\| > `1%` (run 2: **0.03 pts**)

## STM32F746 cascade budget (from Colab §8)

| | Flash | SRAM | Notes |
|--|------:|-----:|-------|
| Disease (ACLIS report) | 723 KB | 271 KB | of 1024 / 340 KB budgets |
| Leaf gate (run 2) | **50.9 KB** | ~80 KB est. | activations shared |
| Combined | **773.9 KB** | **271 KB** shared | **FITS** both flash & SRAM |

Extra flash ≈ +50.9 KB; extra peak SRAM ≈ 0 if gate ≤ disease arena.  
Latency est.: ~877 ms if leaf (gate+disease), ~80 ms if not (gate only).

## On-device follow-ups

1. Emit C from this TFLite via `codegen_leaf_gate_c.py` / `emit_leaf_gate_c_from_tflite.py`
2. Copy `leaf_gate_output[]` after invoke; final classifier without ReLU6
3. Camera path: **full-frame 176→96 resize** + ImageNet requantize (`u8-128` → float ImageNet → int8 with input zp/scale above)
4. Host/MCU ramp parity: `verify_leaf_gate_ramp.py` vs firmware `-DLEAF_GATE_RAMP_TEST`
5. LCD shows gate **softmax confidence %** (dequantized logits); decision remains int8 argmax

## Artifacts

| Path | Role |
|------|------|
| [`run_2_2026-07-28.pdf`](run_2_2026-07-28.pdf) | Colab scroll capture — run 2 (source for this doc) |
| [`run_1_2026-07-28.png`](run_1_2026-07-28.png) | Colab scroll capture — run 1 (superseded) |
| `../aclis_leaf_gate_train_tflite.ipynb` | Colab train + export |
| `../aclis_leaf_gate_96x_full_int8.tflite` | Deployed INT8 model (run 2) |
| `../leaf_noleaf_dataset/` / `.zip` | Training data |
| `../emit_leaf_gate_c_from_tflite.py` | TFLite → C emitter |
| `../codegen_leaf_gate_c.py` | Wrapper that installs into `ACLIS_IKMAL` |
| `../verify_leaf_gate_ramp.py` | Host INT8 ramp parity check |
