from bisect import bisect_left
import os
import json
import numpy as np
import torch
import torchaudio
from config import Config


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f_name:
        data = json.load(f_name)
    return data


def save_json(json_path, data_obj):
    with open(json_path, "w", encoding="utf-8") as f_name:
        json.dump(data_obj, f_name, indent=4)


def load_npy(npy_path):
    data = np.load(npy_path, mmap_mode='r')
    return data


def save_npy(npy_path, np_array):
    with open(npy_path, "wb") as f_name:
        np.save(f_name, np_array)


def get_ego4d_metadata(types="clip"):
    path_ego_meta = os.path.join(Config.EGO4D_DATA_PATH, "/ego4d.json")
    return {
        clip[f"{types}_uid"]: clip for clip in load_json(path_ego_meta)[f"{types}s"]
    }


def modality_checker(meta_video):
    has_imu = meta_video["has_imu"]
    has_audio = (
        False if meta_video["video_metadata"]["audio_start_sec"] is None else True
    )
    return has_imu, has_audio


def get_windows_in_clip(s_time, e_time, window_sec, stride):
    windows = []
    for window_start, window_end in zip(
        np.arange(s_time, e_time, stride),
        np.arange(s_time + window_sec, e_time, stride),
    ):
        windows.append([window_start, window_end])
    return windows


def resample(signals, timestamps, original_sample_rate, resample_rate):
    signals = torch.from_numpy(np.copy(signals))
    timestamps = torch.from_numpy(np.copy(timestamps)).unsqueeze(-1)
    signals = torchaudio.functional.resample(
        waveform=signals.data.T,
        orig_freq=original_sample_rate,
        new_freq=resample_rate,
    ).T.numpy()

    nsamples = len(signals)
    period = 1 / resample_rate
    initial_seconds = timestamps[0] / 1e3
    ntimes = (torch.arange(nsamples) * period).view(-1, 1) + initial_seconds
    timestamps = (ntimes * 1e3).squeeze().numpy()
    return signals, timestamps


def delta(first_num, second_num):
    return abs(first_num - second_num)


def padIMU(signal, duration_sec):
    expected_elements = round(duration_sec) * 50
    if signal.shape[0] > expected_elements:
        signal = signal[:expected_elements, :]
    elif signal.shape[0] < expected_elements:
        padding = expected_elements - signal.shape[0]
        padded_zeros = np.zeros((padding, 6))
        signal = np.concatenate([signal, padded_zeros], 0)
    return signal


def resampleIMU(signal, timestamps, sampling_rate=50):
    sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
    if sampling_rate != 50:
        signal, timestamps = resample(signal, timestamps, sampling_rate, 50)
    return signal, timestamps


def tosec(value):
    return value / 1000


def toms(value):
    return value * 1000


def bisect_left(a, x):
    from bisect import bisect_left as _bisect_left
    return _bisect_left(a, x)

