# Local Test AIS-Sentinel-2 Ship Type Training

This repository contains the local model-training and evaluation scripts used for AIS-labelled Sentinel-2 ship-type classification experiments.

The package is intentionally limited to the reviewer-shareable local training code. It does not include raw datasets, generated patches, trained checkpoints, or experiment result folders.

## Included scripts

- `train_local_shiptype_cv.py`
  - Main patch-level cross-validation trainer.
- `train_local_4class_resnet18.py`
  - Convenience launcher for the 4-class ResNet-18 patch-split experiment.
- `train_local_4class_resnet18_mmsi_grouped.py`
  - Convenience launcher for grouped 4-class ResNet-18 experiments.
- `train_local_4class_efficientnet.py`
  - Convenience launcher for 4-class EfficientNet-B0 experiments.
- `train_local_4class_efficientnet_rgb.py`
  - RGB-only 4-class EfficientNet-B0 variant used in auxiliary checks.
- `train_local_6class_efficientnet.py`
  - Convenience launcher for 6-class EfficientNet-B0 experiments.
- `train_local_6class_resnet18_conflicting_mmsi_excluded.py`
  - 6-class ResNet-18 launcher with conflicting MMSI exclusion enabled.
- `train_local_6class_resnet18_with_florida_external_test.py`
  - Single-fold Denmark training with additional South Florida external evaluation.
- `eval_local_4class_resnet18_on_florida.py`
  - Local Florida evaluation helper for 4-class ResNet-18 outputs.

## Environment

Recommended Python version:

- Python 3.10 or newer

Install dependencies:

```bash
pip install -r requirements.txt
```

## Optional config

The main trainer can read default hyperparameters from `config.yaml`.
An example file is provided as:

```text
config.example.yaml
```

If desired, copy it to `config.yaml` and adjust the values.

## Expected dataset layout

The training scripts expect a class-folder patch dataset such as:

```text
dataset_ship_type/
├── Cargo/
├── Tanker/
├── Fishing/
├── Passenger/
├── Sailing/
└── Pleasure/
```

The default local assumption in the scripts is:

```text
../dataset_ship_type
```

You can override this using `--dataset-root`.

## Example commands

Run the main 4-class ResNet-18 patch-level cross-validation:

```bash
python train_local_4class_resnet18.py --dataset-root D:\path\to\dataset_ship_type
```

Run the generic trainer directly:

```bash
python train_local_shiptype_cv.py --class-scheme 4 --arch resnet18 --split-mode patch --dataset-root D:\path\to\dataset_ship_type
```

Run an auxiliary 4-class DenseNet121 experiment:

```bash
python train_local_shiptype_cv.py --class-scheme 4 --arch densenet121 --split-mode patch --dataset-root D:\path\to\dataset_ship_type
```

Run an auxiliary 4-class EfficientNet-B0 experiment:

```bash
python train_local_4class_efficientnet.py --dataset-root D:\path\to\dataset_ship_type
```

Run the 6-class Florida external-test helper:

```bash
python train_local_6class_resnet18_with_florida_external_test.py --dataset-root D:\path\to\dataset_ship_type --florida-root D:\path\to\florida_patches
```

## Notes

- This repository is intended for code sharing and methodological transparency.
- Dataset generation, AIS/Sentinel-2 ingestion, and database-side patch production are maintained in separate repositories and are not duplicated here.
- Large local outputs such as `results_local*` are excluded from version control.
