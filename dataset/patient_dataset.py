import torch
import numpy as np
import pandas as pd
import random
import copy
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from .utils import (
    mpu_data_convert,
    median_filter,
    trapz,
    z_score_normalize,
    padIMU,
    get_imu_windows,
)


class PatientIMU(torch.utils.data.Dataset):
    def __init__(self, imu_filenames, avg_imu_length, classes, window_size=512):
        super(PatientIMU, self).__init__()
        self.imu_filenames = imu_filenames
        self.window_size = window_size
        self.avg_imu_length = avg_imu_length
        self.classes = classes
        self.label_encoder = LabelEncoder().fit(classes)

    def __len__(self):
        return len(self.imu_filenames)

    def _imu_input(self, imu_path):
        imu_data = pd.read_csv(imu_path, parse_dates=['time'], date_format="%Y/%m/%d %h:%M:%S")
        imu_data.drop(columns=['time', 'date'], inplace=True)
        imu_data = torch.from_numpy(imu_data.to_numpy().astype(np.float32))

        frequency = 50
        window_duration = 5
        window_size = frequency * window_duration

        center_index = len(imu_data) // 2
        start_index = max(0, center_index - window_size // 2)
        end_index = min(len(imu_data), center_index + window_size // 2)

        imu_data = imu_data[start_index:end_index]
        imu_data = padIMU(imu_data, 5)

        true_imu_data = []
        for i in range(len(imu_data)):
            ori_data = imu_data[i]
            true_mpu_data = mpu_data_convert(ori_data, acc_sensitivity=4096, gyro_sensitivity=32.8)
            true_imu_data.append(true_mpu_data)

        filter_datas = median_filter(np.array(true_imu_data), kernel_size=5)
        angles = trapz(filter_datas[3:6, :])

        process_acce_data = filter_datas[0:3, :].T
        process_gyro_data = angles

        return process_acce_data, process_gyro_data

    def __getitem__(self, index):
        imu_path = self.imu_filenames[index]
        label = self.label_encoder.transform([imu_path.split('/')[5]])
        labels = F.one_hot(torch.tensor(label), num_classes=len(self.classes)).type(torch.FloatTensor)

        acce_data, gyro_data = self._imu_input(imu_path)
        imu_data = np.concatenate([acce_data, gyro_data], axis=1)

        return imu_data, labels[0], imu_path.split('/')[5]


class PatientIMUPerSec(torch.utils.data.Dataset):
    def __init__(self, imu_filenames, avg_imu_length, classes, va_name, window_size=512,
                 cache_path='/home/saeid/datasets/cache_windows_patients', min_length_sec=4, stratified=None):
        super(PatientIMUPerSec, self).__init__()
        self.imu_filenames = imu_filenames
        self.window_size = window_size
        self.avg_imu_length = avg_imu_length
        self.va_name = va_name
        self.classes = classes
        self.cache_path = cache_path
        self.label_encoder = LabelEncoder().fit(classes)
        self.min_length_sec = min_length_sec
        self.stratified = stratified
        self.imu_windows = []

        for file in self.imu_filenames:
            imu_data = pd.read_csv(file, parse_dates=['time'], date_format="%Y/%m/%d %h:%M:%S")

            if len(imu_data) < self.min_length_sec * 50:
                continue

            length = len(imu_data) // 50
            windows_in_imu = get_imu_windows(
                s_time=0,
                e_time=length,
                window_sec=5.0,
                stride=5.0,
            )
            random.shuffle(windows_in_imu)
            for (w_s, w_e) in windows_in_imu:
                input_dict = {
                    "window_start": w_s,
                    "window_end": w_e,
                    "imu_path": file,
                }
                self.imu_windows.append(input_dict)

        if stratified:
            class_counts = {cls: 0 for cls in self.classes}
            filtered_windows = []
            for window in self.imu_windows:
                imu_path = window["imu_path"]
                class_label = imu_path.split('/')[6]
                if class_counts[class_label] < self.stratified:
                    filtered_windows.append(window)
                    class_counts[class_label] += 1
            self.imu_windows = filtered_windows

    def __len__(self):
        return len(self.imu_windows)

    def _imu_input(self, imu_path, w_s, w_e):
        imu_data = pd.read_csv(imu_path, parse_dates=['time'], date_format="%Y/%m/%d %h:%M:%S")
        imu_data.drop(columns=['time', 'date'], inplace=True)
        imu_data = imu_data.to_numpy()

        signal = median_filter(np.array(imu_data), kernel_size=5)
        signal = z_score_normalize(signal)

        imu_data = signal[w_s:w_e]
        imu_data = padIMU(imu_data, 5)

        return imu_data

    def __getitem__(self, index):
        dict_out = copy.deepcopy(self.imu_windows[index])
        imu_path = dict_out["imu_path"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]

        label = self.label_encoder.transform([imu_path.split('/')[6]])
        labels = F.one_hot(torch.tensor(label), num_classes=len(self.classes)).type(torch.FloatTensor)

        imu_data = self._imu_input(imu_path, w_s, w_e)
        imu_data = torch.tensor(imu_data, dtype=torch.float32)

        return imu_data, labels[0], imu_path.split('/')[6], dict_out

