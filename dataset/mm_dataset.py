import torch
import numpy as np
import pandas as pd
import torchvision
import torchaudio
import random
import copy
from decord import VideoReader
from .utils import (
    sample_frame_indices,
    mpu_data_convert,
    median_filter,
    trapz,
    z_score_normalize,
    padIMU,
    pad_dataframe,
    windowing_imu,
    get_windows_in_clip,
    batch_feature_to_numpy,
)


class IMUVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_filenames, clip_len, image_processor, avg_imu_length, window_size=512):
        self.video_filenames = video_filenames
        self.clip_len = clip_len
        self.image_processor = image_processor
        self.window_size = window_size
        self.avg_imu_length = avg_imu_length

    def __len__(self):
        return len(self.video_filenames)

    def _imu_input(self, imu_path):
        mpu_datas = pd.read_csv(imu_path, header=None)

        if len(mpu_datas) < self.avg_imu_length:
            mpu_datas = pad_dataframe(mpu_datas, self.avg_imu_length)
        else:
            mpu_datas = mpu_datas[:self.avg_imu_length]

        true_mpu_datas = []
        for i in range(len(mpu_datas)):
            ori_data = mpu_datas.loc[i]
            true_mpu_data = mpu_data_convert(ori_data)
            true_mpu_datas.append(true_mpu_data)

        filter_datas = median_filter(np.array(true_mpu_datas), kernel_size=5)
        angles = trapz(filter_datas[3:6, :])

        process_acce_data = filter_datas[0:3, :].T
        process_gyro_data = angles
        return process_acce_data, process_gyro_data

    def _video_input(self, video_path):
        video_clip = torchvision.io.read_video(video_path, pts_unit='sec')
        video, _, metadata = video_clip
        indices = sample_frame_indices(self.clip_len, len(video))
        frames = video[indices]
        video_input = self.image_processor(list(frames))
        return video_input

    def __getitem__(self, index):
        input = {}
        video_path = self.video_filenames[index]
        imu_path = video_path.replace('video', 'sensor').replace('mp4', 'csv')
        label = video_path.split('/')[6]

        video = self._video_input(video_path)
        acce_data, gyro_data = self._imu_input(imu_path)
        imu_data = np.concatenate([acce_data, gyro_data], axis=1)
        imu_data = windowing_imu(imu_data, self.window_size, 16)

        input['video'] = batch_feature_to_numpy(video)["pixel_values"]
        input['imu'] = imu_data

        return index, input, label


class IMUVideoDatasetV2(torch.utils.data.Dataset):
    def __init__(self, imu_filenames, image_processor, clip_len, window_size=512, min_length_sec=4):
        super(IMUVideoDatasetV2, self).__init__()
        self.imu_filenames = imu_filenames
        self.window_size = window_size
        self.image_processor = image_processor
        self.min_length_sec = min_length_sec
        self.clip_len = clip_len
        self.imu_windows = []

        for file in self.imu_filenames:
            imu_data = pd.read_csv(file)
            length = len(imu_data) // 25
            windows_in_imu = get_windows_in_clip(
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

    def __len__(self):
        return len(self.imu_windows)

    def _imu_input(self, imu_path, w_s, w_e):
        w_s = int(w_s * 50)
        w_e = int(w_e * 50)

        imu_data = pd.read_csv(imu_path)
        imu_data = torch.from_numpy(imu_data.to_numpy().astype(np.float32))
        imu_data = torchaudio.functional.resample(
            waveform=imu_data.T,
            orig_freq=25,
            new_freq=50).T.numpy()

        signal = median_filter(np.array(imu_data), kernel_size=5)
        signal = z_score_normalize(signal)

        imu_data = signal[w_s:w_e]
        imu_data = padIMU(imu_data, 5)

        return imu_data

    def _video_input(self, uid, w_s, w_e):
        video_clip = VideoReader(uid, num_threads=20)
        video = video_clip.get_batch(range(int(25*w_s), int(25*w_e))).asnumpy()
        indices = sample_frame_indices(self.clip_len, len(video))
        frames = video[indices]
        return frames

    def __getitem__(self, index):
        dict_out = copy.deepcopy(self.imu_windows[index])
        imu_path = dict_out["imu_path"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]

        imu_data = self._imu_input(imu_path, w_s, w_e)
        video_path = imu_path.replace('sensor', 'video').replace('csv', 'mp4')

        video = self._video_input(video_path, w_s, w_e)

        input = {'imu': imu_data, 'video': video}
        label = imu_path.split('/')[6]

        return input, label

