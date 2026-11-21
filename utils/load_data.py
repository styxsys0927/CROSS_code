"""Dataset routing for the CROSS reproducibility package."""

from __future__ import annotations

from preprocess.ADFTD import load_kfold_adftd
from preprocess.BCIC_IV_2a import load_loso_bcic_iv_2a
from preprocess.BCIC_IV_2b import load_loso_bcic_iv_2b
from preprocess.EEGMat import load_kfold_eegmat
from preprocess.Mumtaz2017 import load_kfold_mumtaz2017
from preprocess.OpenBMI import load_kfold_openbmi
from preprocess.PhysioNet_MI import load_kfold_physionet_mi
from preprocess.Rockhill2021 import load_kfold_rockhill2021
from preprocess.Shin2018 import load_kfold_shin2018
from preprocess.Zhou2016 import load_loso_zhou2016


DATASET_SAMPLING_RATES = {
    "BCIC-IV-2a": 250.0,
    "BCIC-IV-2b": 250.0,
    "Zhou2016": 200.0,
    "PhysioNet-MI": 250.0,
    "OpenBMI": 250.0,
    "Mumtaz2017": 200.0,
    "ADFTD": 250.0,
    "EEGMat": 500.0,
    "Shin2018": 250.0,
    "Rockhill2021": 250.0,
}


def get_dataset_sampling_rate(dataset_name: str) -> float:
    try:
        return DATASET_SAMPLING_RATES[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Sampling-rate metadata is missing for dataset: {dataset_name}"
        ) from exc


def load_data(dataset_name: str, strategy: str):
    """Load one of the datasets used in the main comparison.

    Returns
    -------
    data, labels, n_class, n_channels, n_times
        ``data`` is expected to be shaped as
        ``(n_subjects, n_trials, n_channels, n_times)``.
    """

    if dataset_name == "BCIC-IV-2a" and strategy == "LOSO":
        data, labels = load_loso_bcic_iv_2a()
        return data, labels, 4, 22, 1000

    if dataset_name == "BCIC-IV-2b" and strategy == "LOSO":
        data, labels = load_loso_bcic_iv_2b()
        return data, labels, 2, 3, 1000

    if dataset_name == "Zhou2016" and strategy == "LOSO":
        data, labels = load_loso_zhou2016()
        return data, labels, 3, 14, 1000

    if dataset_name == "PhysioNet-MI" and strategy == "KFold":
        data, labels = load_kfold_physionet_mi()
        return data, labels, 4, 64, 1000

    if dataset_name == "OpenBMI" and strategy == "KFold":
        data, labels = load_kfold_openbmi()
        return data, labels, 2, 20, 1000

    if dataset_name == "Mumtaz2017" and strategy == "KFold":
        data, labels = load_kfold_mumtaz2017()
        return data, labels, 2, 19, 1000

    if dataset_name == "ADFTD" and strategy == "KFold":
        data, labels = load_kfold_adftd()
        return data, labels, 3, 19, 1000

    if dataset_name == "EEGMat" and strategy == "KFold":
        data, labels = load_kfold_eegmat()
        return data, labels, 2, 19, 1000

    if dataset_name == "Shin2018" and strategy == "KFold":
        data, labels = load_kfold_shin2018()
        return data, labels, 2, 28, 1000

    if dataset_name == "Rockhill2021" and strategy == "KFold":
        data, labels = load_kfold_rockhill2021()
        return data, labels, 2, 32, 1000

    raise ValueError(f"Unsupported dataset/strategy: {dataset_name}/{strategy}")
