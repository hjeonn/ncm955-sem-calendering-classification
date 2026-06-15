# NCM955 SEM Image Classification

This repository contains MATLAB code for SEM image-based classification of NCM955 calendering-induced electrode states.

The ML task is four-class SEM image classification. The code does not perform direct regression or prediction of electrochemical capacity, capacity retention, or other battery-performance metrics.

## Data layout

Place SEM images in a folder named `data/`, with one subfolder per class:

```text
data/
├── ncm955_1/
├── ncm955_2/
├── ncm955_3/
└── ncm955_4/
```

The folder names are used as class labels.

## Requirements

- MATLAB with Deep Learning Toolbox
- EfficientNet-B0 support for `imagePretrainedNetwork`
- `pretrained_weights_from_e1.mat` only if running the NCM-domain transfer-learning model

## Final evaluation

Run all three final settings:

```matlab
main
```

Run only the ImageNet-pretrained final setting:

```matlab
main('runFromScratch',false,'runTransfer',false,'runImagenet',true)
```

Final settings:

| Setting | Mode | LR | Epochs | Batch |
|---|---|---:|---:|---:|
| FS-C | Random initialization | 3e-4 | 50 | 32 |
| IMG-D | ImageNet pretraining | 3e-4 | 50 | 32 |
| TL-A | NCM-domain transfer learning | 1e-4 | 20 | 32 |

The default protocol uses 10 repeated stratified train/validation/test splits with a 70/10/20 ratio. Data augmentation is applied only to training images after splitting.

## Hyperparameter screening

Run the full screening grid:

```matlab
run_hyperparameter_screening
```

Run only ImageNet-pretrained settings:

```matlab
run_hyperparameter_screening('runOnlyMode','imagenet_pretrained')
```

Settings are selected by validation macro-F1. Held-out test metrics are not used for hyperparameter selection.

## Confusion matrices

After running final evaluation, plot aggregated row-normalized confusion matrices:

```matlab
plot_confusion_matrices
```

The function aggregates saved `confusion_run_*_seed_*.csv` files across repeated held-out test runs.

## Main output files

- `all_run_metrics.csv`
- `all_class_metrics.csv`
- `summary_mean_sd.csv`
- `class_summary_mean_sd.csv`
- `final_method_configs.csv`
- `confusion_run_*_seed_*.csv`

## Notes

- If SEM images are crops from larger parent micrographs, set `groupSplitByParent` to `true` and edit `parentIdFromFilename()` to match the file-naming convention.
- If the transfer-learning weights are not shared publicly, run `main('runTransfer',false)`.
