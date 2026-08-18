# Leaf Gate Model

Tiny **leaf / not_leaf** gate for the ACLIS STM32 cascade. This model runs before the PlantVillage disease classifier and decides whether the input frame should continue down the plant-disease path.

## What’s here

| Item | Description |
|------|-------------|
| [`Baseline Leaf Gate Model/aclis_leaf_gate_96x_full_int8.tflite`](Baseline%20Leaf%20Gate%20Model/aclis_leaf_gate_96x_full_int8.tflite) | Baseline INT8 export |
| [`Baseline Leaf Gate Model/aclis_leaf_gate_train_tflite.ipynb`](Baseline%20Leaf%20Gate%20Model/aclis_leaf_gate_train_tflite.ipynb) | Baseline training and export notebook |
| [`Baseline Leaf Gate Model/emit_leaf_gate_c_from_tflite.py`](Baseline%20Leaf%20Gate%20Model/emit_leaf_gate_c_from_tflite.py) | TFLite to C emitter |
| [`Baseline Leaf Gate Model/codegen_leaf_gate_c.py`](Baseline%20Leaf%20Gate%20Model/codegen_leaf_gate_c.py) | Installs generated C into the STM32 project |
| [`Baseline Leaf Gate Model/Colab Training Run History/TRAINING_RESULTS.md`](Baseline%20Leaf%20Gate%20Model/Colab%20Training%20Run%20History/TRAINING_RESULTS.md) | Baseline training notes and metrics |
| [`Alt Leaf Gate Model/aclis_leaf_gate_96x_alt_full_int8.tflite`](Alt%20Leaf%20Gate%20Model/aclis_leaf_gate_96x_alt_full_int8.tflite) | Alternate INT8 export |
| [`Alt Leaf Gate Model/aclis_leaf_gate_alt_distill_qat.ipynb`](Alt%20Leaf%20Gate%20Model/aclis_leaf_gate_alt_distill_qat.ipynb) | Alternate recipe notebook with distillation and QAT |
| [`Alt Leaf Gate Model/codegen_leaf_gate_alt_c.py`](Alt%20Leaf%20Gate%20Model/codegen_leaf_gate_alt_c.py) | Alternate C generation script |
| [`Alt Leaf Gate Model/ALT_TRAINING_REPORT.md`](Alt%20Leaf%20Gate%20Model/ALT_TRAINING_REPORT.md) | Alternate training summary |
| [`Tests Scripts/verify_leaf_gate_ramp.py`](Tests%20Scripts/verify_leaf_gate_ramp.py) | Host-side INT8 parity check |
| [`baseline vs alt presentation slides.pdf`](baseline%20vs%20alt%20presentation%20slides.pdf) | Baseline vs alternate comparison deck |

## Dataset

The training dataset lives in `leaf_noleaf_dataset/`, with the archive stored as `leaf_noleaf_dataset.zip`.

## Helper scripts

The Python helper scripts in this folder depend on TinyEngine source code that is not included in this repository. Get it from the official TinyEngine repo: [mit-han-lab/tinyengine](https://github.com/mit-han-lab/tinyengine)

## Deployment notes

- Training used `RandomResizedCrop(96, scale=0.5–1.0)`.
- Evaluation and firmware use full-frame `Resize(96)` with no center crop.
- The camera buffer is `u8-128`, then firmware requantizes into TFLite INT8. It is not a raw cast.
- Before blaming the model, run the parity check:

```bash
python "Tests Scripts/verify_leaf_gate_ramp.py"
```

## Generated code

Generated STM32 sources are not stored in this folder. They are wired into the main firmware project instead.

## Regenerate C from TFLite

```bash
cd "Leaf Gate Model/Baseline Leaf Gate Model"
python emit_leaf_gate_c_from_tflite.py
python codegen_leaf_gate_c.py
```
