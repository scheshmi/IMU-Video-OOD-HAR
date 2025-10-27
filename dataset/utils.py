import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid


def sample_frame_indices(clip_len, seg_len):
    sampling_rate = seg_len // clip_len
    sampled_indices = []
    for i in range(clip_len):
        start_idx = i * sampling_rate
        end_idx = min((i + 1) * sampling_rate, seg_len)
        random_frame_idx = np.random.randint(start_idx, end_idx)
        sampled_indices.append(random_frame_idx)
    return sampled_indices


def mpu_data_convert(ori_mpu_data, acc_sensitivity=8192, gyro_sensitivity=16.4):
    acc_x = ori_mpu_data[0] / acc_sensitivity
    acc_y = ori_mpu_data[1] / acc_sensitivity
    acc_z = ori_mpu_data[2] / acc_sensitivity
    gyro_x = ori_mpu_data[3] / gyro_sensitivity
    gyro_y = ori_mpu_data[4] / gyro_sensitivity
    gyro_z = ori_mpu_data[5] / gyro_sensitivity
    return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]


def median_filter(true_mpu_datas, kernel_size=3):
    if true_mpu_datas.ndim == 2 and true_mpu_datas.shape[1] == 6:
        filter_datas = []
        for i in range(6):
            filter_data = signal.medfilt(true_mpu_datas[:, i], kernel_size=kernel_size)
            filter_datas.append(filter_data)
        return np.array(filter_datas).astype(np.float32)
    else:
        filter_datas = np.apply_along_axis(
            signal.medfilt, axis=0, arr=true_mpu_datas, kernel_size=kernel_size
        )
        return filter_datas.astype(np.float32)


def trapz(filter_datas):
    mean = np.mean(filter_datas, axis=1).reshape((3, 1))
    filter_datas = filter_datas - mean
    angles = []
    init_angle = filter_datas[:, 0].reshape((1, 3))
    for i in range(1, filter_datas.shape[1]+1):
        angle = trapezoid(filter_datas[:, 0:i])
        angles.append(angle)
    angles = np.array(angles) + init_angle
    return angles.astype(np.float32)


def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - mean) / std


def padIMU(signal, duration_sec, sampling_rate=50):
    expected_elements = round(duration_sec) * sampling_rate
    if signal.shape[0] > expected_elements:
        signal = signal[:expected_elements, :]
    elif signal.shape[0] < expected_elements:
        padding = expected_elements - signal.shape[0]
        padded_zeros = np.zeros((padding, 6))
        signal = np.concatenate([signal, padded_zeros], 0)
    return signal


def pad_dataframe(data, max_length):
    padding_needed = max_length - len(data)
    if padding_needed > 0:
        padding_df = pd.DataFrame(np.zeros((padding_needed, data.shape[1])), columns=data.columns)
        df_padded = pd.concat([data, padding_df], ignore_index=True)
        return df_padded
    else:
        return data


def windowing_imu(imu_data, window_size, stride):
    num_windows = (imu_data.shape[0] - window_size) // stride + 1
    windows = np.array([imu_data[i:i + window_size]
                       for i in range(0, num_windows * stride, stride)])
    return windows


def get_windows_in_clip(s_time, e_time, window_sec, stride):
    windows = []
    for window_start, window_end in zip(
        np.arange(s_time, e_time, stride),
        np.arange(s_time + window_sec, e_time, stride),
    ):
        windows.append([window_start, window_end])
    return windows


def get_imu_windows(s_time, e_time, window_sec, stride, freq=50):
    windows = []
    window_size = int(window_sec * freq)
    stride_size = int(stride * freq)

    for window_start in np.arange(s_time * freq, e_time * freq, stride_size):
        window_end = window_start + window_size
        windows.append([window_start, window_end])
    return windows


def batch_feature_to_numpy(batch_feature):
    numpy_dict = {}
    for key, value in batch_feature.data.items():
        if isinstance(value, list):
            numpy_dict[key] = np.array(value)
        else:
            numpy_dict[key] = value.numpy()
    return numpy_dict

