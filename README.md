# CROSS Reproducibility Package

This is an anonymous reproduction package for the CROSS main experiments.
It contains only:

- the full CROSS model;
- DSAINet;
- nine selected baseline models;
- the dataset preprocessors used in the main comparison;
- LOSO/KFold training scripts and simple launch scripts.

Logs, trained weights, manuscript files, tables, figures, and local analysis
utilities are intentionally not included.

## Included models

| Public name | Config | Source file |
| --- | --- | --- |
| CROSS | `config/cross.yaml` | `model/CROSS.py` |
| DSAINet | `config/DSAINet.yaml` | `model/DSAINet.py` |
| EEGNet | `config/EEGNet.yaml` | `model/EEGNet.py` |
| ShallowConvNet | `config/ShallowConvNet.yaml` | `model/ShallowConvNet.py` |
| DeepConvNet | `config/DeepConvNet.yaml` | `model/DeepConvNet.py` |
| ADFCNN | `config/ADFCNN.yaml` | `model/ADFCNN.py` |
| CTNet | `config/CTNet.yaml` | `model/CTNet.py` |
| LMDANet | `config/LMDANet.yaml` | `model/LMDANet.py` |
| MSCFormer | `config/MSCFormer.yaml` | `model/MSCFormer.py` |
| TMSANet | `config/TMSANet.yaml` | `model/TMSANet.py` |
| Deformer | `config/Deformer.yaml` | `model/Deformer.py` |

The CROSS implementation uses a short public name (`CROSS`), a short config
file (`cross.yaml`), and one self-contained source file (`model/CROSS.py`).

## Included datasets

The main comparison uses ten datasets:

- LOSO: `BCIC-IV-2a`, `BCIC-IV-2b`, `Zhou2016`
- KFold: `OpenBMI`, `PhysioNet-MI`, `Mumtaz2017`, `ADFTD`,
  `Rockhill2021`, `EEGMat`, `Shin2018`

Dataset files are not included. Put them under a data root with the following
layout, or set `EEG_DATA_ROOT` to an equivalent location:

```text
<EEG_DATA_ROOT>/
  BCIC_IV_2a/
  BCIC_IV_2b/
  Zhou2016/
  OpenBMI/
  PhysioNet_MI/files/
  Mumtaz2017/
  ADFTD/ds004504/
  Rockhill2021/ds002778/
  EEGMat/eeg-during-mental-arithmetic-tasks-1.0.0/
  Shin2018/
```

If `EEG_DATA_ROOT` is not set, the loaders default to `./data` inside this
project.

## Environment

Recommended:

```shell
conda env create -f environment.yml
conda activate cross-repro
```

Or install the Python requirements into an existing PyTorch environment:

```shell
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your driver from the official
PyTorch instructions if GPU training is needed.

Two launch-script families are provided:

- Windows / PowerShell: `scripts/run_cross.ps1`, `scripts/run_baseline.ps1`
- Linux / macOS shell: `scripts/run_cross.sh`, `scripts/run_baseline.sh`

## Run CROSS

Windows / PowerShell:

```powershell
$env:EEG_DATA_ROOT = "D:\path\to\eeg_datasets"
powershell -ExecutionPolicy Bypass -File .\scripts\run_cross.ps1 `
  -Datasets Zhou2016 `
  -Seeds 0 `
  -Device 0
```

Equivalent direct command:

```shell
python train_loso.py --config config/cross.yaml --dataset Zhou2016 --seed 0 --device 0
```

Run several datasets/seeds:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_cross.ps1 `
  -Datasets Zhou2016,OpenBMI `
  -Seeds 0,1,2,3,4 `
  -Device 0
```

Linux / macOS:

```shell
export EEG_DATA_ROOT=/path/to/eeg_datasets
bash scripts/run_cross.sh --datasets Zhou2016,OpenBMI --seeds 0,1,2,3,4 --device 0
```

## Run baselines

Windows / PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_baseline.ps1 `
  -Models DSAINet,EEGNet `
  -Datasets Zhou2016,OpenBMI `
  -Seeds 0 `
  -Device 0
```

Linux / macOS:

```shell
export EEG_DATA_ROOT=/path/to/eeg_datasets
bash scripts/run_baseline.sh --models DSAINet,EEGNet --datasets Zhou2016,OpenBMI --seeds 0 --device 0
```

Equivalent direct command for one KFold baseline:

```shell
python train_kfold.py --config config/DSAINet.yaml --dataset OpenBMI --seed 0 --device 0
```

## Dry run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_cross.ps1 -Datasets Zhou2016 -Seeds 0 -DryRun
powershell -ExecutionPolicy Bypass -File .\scripts\run_baseline.ps1 -Models DSAINet -Datasets Zhou2016 -Seeds 0 -DryRun
```

```shell
bash scripts/run_cross.sh --datasets Zhou2016 --seeds 0 --dry-run
bash scripts/run_baseline.sh --models DSAINet --datasets Zhou2016 --seeds 0 --dry-run
```

By default, `run_cross` runs CROSS on all included datasets for seed 0, and
`run_baseline` runs all included baseline models on all included datasets for
seed 0. Use `-Datasets`, `-Models`, and `-Seeds` in PowerShell, or
`--datasets`, `--models`, and `--seeds` in shell, to select a subset.

## Outputs

Training writes runtime artifacts to:

```text
log/<MODEL>/<DATASET>_seed<SEED>.log
weight/<MODEL>/<DATASET>/seed_<SEED>/
```

These directories are ignored by `.gitignore` and are not part of the minimal
source release.

## Notes

- `train_loso.py` is used for `BCIC-IV-2a`, `BCIC-IV-2b`, and `Zhou2016`.
- `train_kfold.py` is used for the remaining datasets.
- CROSS-specific comments are placed in `config/cross.yaml`,
  `model/CROSS.py`, and `utils/load_model.py`.
- The package contains no local absolute paths or personal directory names.
