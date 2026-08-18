# Leaf+Pest Gate Model

This folder contains the combined gate model that separates leaf images from pest/other inputs for the ACLIS pipeline.

## Contents

| Item | Description |
|------|-------------|
| [`aclis_pest_gate_96x_full_int8.tflite`](aclis_pest_gate_96x_full_int8.tflite) | Exported INT8 gate model |
| [`aclis_pest_gate_distill_qat.ipynb`](aclis_pest_gate_distill_qat.ipynb) | Training notebook |
| [`emit_pest_gate_c_from_tflite.py`](emit_pest_gate_c_from_tflite.py) | TFLite to C emitter |
| [`codegen_pest_gate_c.py`](codegen_pest_gate_c.py) | Installs generated C into the firmware project |
| [`training_record.pdf`](training_record.pdf) | Training record / notes |

## Notes

- This gate is intended to run as part of the STM32 cascade before downstream classifiers.
- Dataset archives are kept out of version control via the root `.gitignore`.

## Helper scripts

The Python helper scripts in this folder depend on TinyEngine source code that is not included in this repository. Get it from the official TinyEngine repo: [mit-han-lab/tinyengine](https://github.com/mit-han-lab/tinyengine)

## Regenerate C from TFLite

```bash
cd "Leaf+Pest Gate Model"
python emit_pest_gate_c_from_tflite.py
python codegen_pest_gate_c.py
```
