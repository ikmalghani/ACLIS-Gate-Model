# Alt Leaf Gate — Training Process Report

**Source run:** Colab scroll capture [`alt_training_record.pdf`](alt_training_record.pdf) (13 pages, 2026-07-29)  
**Notebook:** [`aclis_leaf_gate_alt_distill_qat.ipynb`](aclis_leaf_gate_alt_distill_qat.ipynb)  
**Artifact:** [`aclis_leaf_gate_96x_alt_full_int8.tflite`](aclis_leaf_gate_96x_alt_full_int8.tflite) (~51.1 KB)  
**Baseline reference:** `../aclis_leaf_gate_train_tflite.ipynb` / run 2 in `../Colab Training Run History/TRAINING_RESULTS.md`

---

## 1. Goal

Keep the **same TinyLeafGate architecture** (~29k params, ~50 KB INT8) so MCU flash/SRAM stay comparable, and change only the **training recipe** to improve behaviour under **camera-like / field** conditions (OV2640 blur, JPEG, lighting), not just clean Kaggle crops.

```
capture → leaf / not-leaf gate → if leaf: disease CNN
                               → if not:  reject
```

---

## 2. What the alt training process does

The alt pipeline is deliberately different from baseline at every stage except the student topology and the TinyEngine export path.

### 2.1 Same student, different teacher

| Role | Model | Params |
|------|--------|-------:|
| **Student (deployed)** | TinyLeafGate (stem + 4 DW-separable blocks + GAP + classifier) | **27,538** trainable (~28,854 total with BN) |
| **Teacher (train only)** | MobileNetV3-Small, ImageNet → fine-tuned on leaf/not_leaf | ~1.52 M |

Isolating architecture means any gain/loss vs baseline is attributable to the recipe, not a bigger net.

### 2.2 Stage A — Teacher fine-tune (8 epochs)

- Loss: label-smoothed CE (`LABEL_SMOOTH=0.05`) + class weights  
- Optim: AdamW (`lr=3e-4`, weight decay `5e-4`), cosine anneal  
- Result from the record: best **val ≈ 0.9992** (near-ceiling on this dataset)  
- Soft logits become the distillation target for the student  

### 2.3 Stage B — Student distillation (up to 40 epochs)

Combined loss:

\[
\mathcal{L} = \alpha\,\mathrm{KL}\!\left(\mathrm{softmax}(z_t/T),\,\mathrm{softmax}(z_s/T)\right)\,T^{2}
\;+\; (1-\alpha)\,\mathcal{L}_{\text{hard}}
\]

with \(\alpha=0.7\), \(T=4\). Hard loss is MixUp CE (α_mix=0.2) with class weights.

Also used:

| Technique | Setting | Intent |
|-----------|---------|--------|
| Field augs | blur (p=0.35), JPEG q=35–90 (p=0.4), stronger color/lighting, wider crop/affine | Match OV2640 / outdoor capture |
| MixUp | α=0.2 | Smoother decision boundary, less brittle to odd frames |
| EMA | decay 0.999 | Stable export weights (checkpoint on EMA val) |
| Schedule | AdamW, 3-epoch linear warmup → cosine, lr peak `1e-3` | Single coherent schedule vs baseline’s abrupt 5+25 LR drop |
| Early stop | patience 12 | Record: ran full 40 epochs; **best EMA val = 0.9910** |

### 2.4 Stage C — QAT proxy (5 epochs)

Eager Torch QAT fuse path was skipped; instead **INT8-noise proxy** (fake-quant activations after stem / before head, STE) with continued KD (0.5 soft + 0.5 hard), AdamW `1e-4`.

Record: val stayed ~**0.989–0.990** through QAT (leaf ~0.993, not_leaf ~0.987 at epoch 5).

### 2.5 Export (unchanged TinyEngine path)

Same Keras twin as baseline: PT-matched `ZeroPadding2D(1)` + valid convs, BN ε=`1e-5`, GAP(`keepdims`) + 1×1 Conv classifier → full INT8 TFLite. Float PT↔Keras logits matched; graph marked TinyEngine-friendly.

---

## 3. Results from `alt_training_record.pdf`

### 3.1 Alt model alone

| Metric | Value |
|--------|------:|
| Clean FP32 test (record §6) | **~99.1–99.5%** (pass ≫ 92% target) |
| Clean INT8 TFLite | **99.07%** |
| Camera-stress INT8 | **98.86%** |
| \|INT8 − FP32\| gap | **~0.03 pts** (within 1% gate) |
| Pred agreement INT8↔PT | **~99.9%** |
| TFLite size | **51.1 KB** |
| Disease + alt flash | **774.1 / 1024 KB** — **FITS** |

Camera-stress = deterministic full-frame resize + fixed blur + underexposure + JPEG q=55 (same inputs for both models).

### 3.2 Side-by-side INT8 (§8 of the record) — decisive table

| Model | Split | Acc | leaf | not_leaf | FN_leaf† | FP_leaf‡ | KB |
|-------|-------|----:|-----:|---------:|---------:|---------:|---:|
| **BASELINE** | clean | **99.13%** | 99.4% | 98.8% | **11** | 21 | 50.9 |
| **ALT** | clean | 99.07% | 99.2% | 98.9% | 15 | **19** | 51.1 |
| **BASELINE** | stress | 97.63% | 99.1% | 96.0% | 17 | 70 | 50.9 |
| **ALT** | stress | **98.86%** | **99.5%** | **98.2%** | **10** | **32** | 51.1 |

† FN_leaf = true leaf → predicted not_leaf (gate rejects a real leaf)  
‡ FP_leaf = true not_leaf → predicted leaf (disease CNN runs on junk)

**Deltas (ALT − BASELINE), positive = alt better:**

| Split | Δacc | ΔFN_leaf (fewer = better) | ΔFP_leaf (fewer = better) |
|-------|-----:|--------------------------:|--------------------------:|
| clean | **−0.05 pts** | −4 (baseline fewer FN) | +2 (alt fewer FP) |
| stress | **+1.23 pts** | **+7** | **+38** |

Notebook recommendation from this table: **prefer ALT for field/camera robustness.**

---

## 4. Why this process should beat baseline in real life

Baseline already saturates the **clean Kaggle test** (~99.1%). Extra points on that split are mostly noise. The ACLIS failure mode that matters is the **board camera**: soft focus, JPEG from the Arducam path, uneven outdoor light, leaves that do not fill the frame cleanly.

### 4.1 Field-matched augmentation (main lever)

Baseline trains with geometric/color jitter + erase. Alt **adds blur, JPEG, and harsher lighting** during training, then measures both models on a fixed camera-stress set. The record shows baseline **drops ~1.5 pts** clean→stress (99.13 → 97.63), while alt drops only **~0.2 pts** (99.07 → 98.86). That is direct evidence the recipe closed the domain gap the MCU will see.

FP_leaf under stress falls from **70 → 32**: fewer junk frames pushed into the expensive disease CNN. FN_leaf also improves under stress (**17 → 10**): fewer real leaves wrongly rejected in degraded captures.

### 4.2 Knowledge distillation

A MobileNetV3-Small teacher at ~99.9% val provides **soft targets** (relative class confidences), not just hard 0/1 labels. For a tiny student, that usually improves calibration and behaviour on ambiguous frames (partial leaf, busy background) without growing the deployed net.

### 4.3 MixUp + EMA + AdamW / warmup-cosine

- **MixUp** softens overconfident boundaries — useful when field frames sit between “clear leaf” and “clear not-leaf”.  
- **EMA** exports a temporally averaged student; the record checkpoints on EMA val and uses those weights for test/export.  
- **Warmup-cosine AdamW** avoids the baseline’s sharp LR step (1e-3 → 1e-4) and adds weight decay for a small over-parameterised-looking head on limited diversity.

### 4.4 Quantization-aware fine-tune

Baseline is PTQ-only. Alt’s short INT8-noise QAT + KD keeps **INT8 ≈ FP32** (gap ~0.03 pts, agreement ~99.9%). Deployed MCU path is INT8; training that anticipates quantisation error reduces “looks great in float, worse on device” surprises.

### 4.5 What alt does *not* claim

On **clean** Kaggle INT8, alt is **slightly behind** baseline (−0.05 pts, +4 FN_leaf). That is expected: harder augs and MixUp trade a little in-distribution polish for out-of-distribution robustness. For ACLIS, stress/field behaviour is the more relevant proxy than another 0.05 pts on already-saturated clean test.

Flash cost is essentially unchanged (**+0.2 KB** vs baseline). Teacher cost is **training-time only**.

---

## 5. Process comparison (summary)

| | Baseline | Alt |
|--|----------|-----|
| Student | TinyLeafGate | **Same** |
| Teacher / KD | none | MobileNetV3-Small + KL |
| Augments | crop/flip/jitter/erase | + **blur, JPEG, harsh light** |
| Regularisation | class weights + sampler | + **MixUp, label smooth, EMA** |
| Optim | Adam 5 + 25 (lr÷10) | **AdamW warmup-cosine (40)** |
| Quant | PTQ | **QAT proxy + PTQ** |
| Eval | clean test | clean + **camera-stress** + vs baseline TFLite |
| Clean INT8 | **99.13%** | 99.07% |
| Stress INT8 | 97.63% | **98.86%** |
| Deploy size | 50.9 KB | 51.1 KB |

---

## 6. Conclusion

The alt training process keeps the **same MCU-sized TinyLeafGate** and invests complexity in **teacher KD, camera-hardened augs, MixUp/EMA, and QAT**. Against the recorded baseline TFLite, that yields a **clear win on the camera-stress proxy (+1.23 pts, large cuts in FP_leaf and FN_leaf)** with a negligible clean-set regression and the same flash budget.

**Recommendation from the training record:** flash the alt gate for field A/B (`codegen_leaf_gate_alt_c.py`), then confirm on real STM32 + OV2640 frames; keep baseline codegen available to switch back if on-device clean behaviour regresses.

---

## 7. Artifacts & next steps

| Path | Role |
|------|------|
| `aclis_leaf_gate_alt_distill_qat.ipynb` | Alt Colab recipe |
| `alt_training_record.pdf` | This run’s Colab capture (source of §3 numbers) |
| `aclis_leaf_gate_96x_alt_full_int8.tflite` | Deployable INT8 |
| `codegen_leaf_gate_alt_c.py` | Emit C into `ACLIS_IKMAL/.../codegen_leaf_gate/` |
| `../codegen_leaf_gate_c.py` | Restore baseline C weights |

On-device check: same 20–30 leaf + 20–30 non-leaf field shots under baseline firmware, then alt, and compare gate decisions / LCD confidence.
