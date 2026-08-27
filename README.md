# Gate Model

Repository for ACLIS gate models and supporting training / deployment assets.

## Folders

- `Leaf Gate Model/` for the leaf vs not-leaf gate
- `Leaf+Pest Gate Model/` for the leaf vs pest/other gate
- `Better Gate Model/` for the PlantVillage-balanced leaf/non-leaf dataset, local alt trainer, INT8 TFLite, and `emit_better_gate_c_from_tflite.py` / `codegen_better_gate_c.py` (installs 2-class `leaf_gate_*` C into `ACLIS_IKMAL`)

## Notes

- The repo includes notebooks, exported INT8 TFLite models, and helper scripts for C code generation.
- Large datasets and dataset archives are ignored by default.
- Install Python dependencies with `pip install -r requirements.txt` before running the helper scripts.
- TinyEngine is not vendored here; get it from the official repository: [mit-han-lab/tinyengine](https://github.com/mit-han-lab/tinyengine)
