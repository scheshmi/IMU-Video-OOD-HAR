import torch
import numpy as np
import pandas as pd
import torchaudio
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from .utils import median_filter, z_score_normalize


class IMUDataset(torch.utils.data.Dataset):
    def __init__(self, imu_filenames, avg_imu_length, classes, resample=None, window_size=512):
        super(IMUDataset, self).__init__()
        self.imu_filenames = imu_filenames
        self.window_size = window_size
        self.avg_imu_length = avg_imu_length
        self.resample = resample
        self.classes = classes
        self.label_encoder = LabelEncoder().fit(classes)

    def __len__(self):
        return len(self.imu_filenames)

    def _imu_input(self, imu_path):
        mpu_datas = pd.read_csv(imu_path, header=None)

        if self.resample is not None:
            mpu_datas = torch.from_numpy(mpu_datas.to_numpy().astype(np.float32))
            mpu_datas = torchaudio.functional.resample(
                waveform=mpu_datas.T,
                orig_freq=25,
                new_freq=self.resample).T.numpy()

        mpu_datas = median_filter(mpu_datas, kernel_size=5)
        signal = z_score_normalize(mpu_datas)

        frequency = 50
        window_duration = 5
        window_size = frequency * window_duration

        center_index = len(mpu_datas) // 2
        start_index = max(0, center_index - window_size // 2)
        end_index = min(len(mpu_datas), center_index + window_size // 2)

        signal = signal[start_index:end_index]
        return signal

    def __getitem__(self, index):
        imu_path = self.imu_filenames[index]
        imu_path = imu_path.replace('video', 'sensor').replace('mp4', 'csv')

        label = self.label_encoder.transform([imu_path.split('/')[6]])
        labels = F.one_hot(torch.tensor(label), num_classes=len(self.classes)).type(torch.FloatTensor)

        imu_data = self._imu_input(imu_path)

        return imu_data, labels[0], imu_path.split('/')[6]

