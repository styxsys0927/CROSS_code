import io
import os
import warnings
import zipfile
from typing import Iterable, List, Tuple

import mne
import numpy as np
from scipy.io import loadmat
from preprocess.paths import dataset_path

mne.set_log_level("ERROR")
warnings.filterwarnings(
    "ignore",
    message="Channels contain different lowpass filters.*",
    category=RuntimeWarning,
    module="mne",
)

# DSR session markers. For each marker, keep 20s before and 20s after the marker,
# then split each 20s segment into five 4s windows.
ANCHOR_CODE = 48
SEG_LEN_S = 20.0
WIN_LEN_S = 4.0

root_dir = dataset_path('Shin2018')
target_channels = [
    "Fp1", "AFF5h", "AFz", "F1", "FC5", "FC1", "T7",
    "C3", "Cz", "CP5", "CP1", "P7", "P3", "Pz",
    "POz", "O1", "Fp2", "AFF6h", "F2", "FC2", "FC6",
    "C4", "T8", "CP2", "CP6", "P4", "P8", "O2",
]

bandpass = (0.5, 40.0)
resample_rate = 250.0


def _subject_zip(subject_id: int) -> str:
    return os.path.join(root_dir, f"VP{subject_id:03d}-EEG.zip")


def _load_dsr_mat(subject_id: int):
    zip_path = _subject_zip(subject_id)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing file: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        cnt = loadmat(io.BytesIO(zf.read("cnt_dsr.mat")), squeeze_me=True, struct_as_record=False)["cnt_dsr"]
        mrk = loadmat(io.BytesIO(zf.read("mrk_dsr.mat")), squeeze_me=True, struct_as_record=False)["mrk_dsr"]
    return cnt, mrk


def _preprocess_cnt(cnt) -> mne.io.BaseRaw:
    ch_names = [str(ch) for ch in np.atleast_1d(cnt.clab).tolist()]
    data = np.asarray(cnt.x, dtype=np.float64).T  # (channels, samples), unit already uV
    sfreq = float(cnt.fs)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    present = [ch for ch in target_channels if ch in raw.ch_names]
    missing = [ch for ch in target_channels if ch not in present]
    if missing:
        print(f"[WARN] Missing channels: {missing}")

    raw.pick_channels(present, ordered=True)
    raw.filter(bandpass[0], bandpass[1], picks="eeg", verbose=False)
    raw.resample(resample_rate, npad="auto", verbose=False)
    return raw


def _extract_task_rest_windows(raw: mne.io.BaseRaw, marker_times_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    x = raw.get_data()  # (C, samples), uV

    seg_samp = int(round(SEG_LEN_S * sfreq))
    win_samp = int(round(WIN_LEN_S * sfreq))
    n_win = seg_samp // win_samp
    n_time = x.shape[1]

    X_list, y_list = [], []

    def add_segment(seg_start: int, label: int):
        seg_end = seg_start + seg_samp
        if seg_start < 0 or seg_end > n_time:
            return
        for k in range(n_win):
            start = seg_start + k * win_samp
            end = start + win_samp
            X_list.append(x[:, start:end])
            y_list.append(label)

    for marker_ms in marker_times_ms:
        anchor = int(round((float(marker_ms) / 1000.0) * sfreq))
        add_segment(anchor - seg_samp, label=0)
        add_segment(anchor, label=1)

    if not X_list:
        return (
            np.empty((0, len(target_channels), win_samp), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def load_kfold_shin2018(
    subject_ids: Iterable[int] = range(1, 27),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    all_subject_data, all_subject_labels = [], []

    for sid in subject_ids:
        cnt, mrk = _load_dsr_mat(sid)
        raw = _preprocess_cnt(cnt)

        marker_desc = np.asarray(mrk.event.desc)
        marker_times = np.asarray(mrk.time)
        anchors = marker_times[marker_desc == ANCHOR_CODE]

        X_subj, y_subj = _extract_task_rest_windows(raw, anchors)
        all_subject_data.append(X_subj)
        all_subject_labels.append(y_subj)

    return all_subject_data, all_subject_labels
