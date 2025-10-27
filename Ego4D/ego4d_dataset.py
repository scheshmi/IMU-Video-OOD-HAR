import copy
import random
import os
import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from .utils_ego import (
    get_ego4d_metadata,
    modality_checker,
    get_windows_in_clip,
    load_json,
    load_npy,
    delta,
    bisect_left,
    toms, tosec,
    padIMU,
)
from typing import Optional
from config import Config

random.seed(1234)


class Ego4dDatasetUnsupervised(torch.utils.data.Dataset):
    def __init__(
        self,
        image_processor,
        window_sec=5.0,
        target_frames_in_window=10,
        cache_imu=True,
        window_sample_rate=1.0,
        max_n_windows_per_video=None,
        shuffle_windows=True,
    ):
        self.cache_imu = {"cache": cache_imu, "path": Config.EGO4D_IMU_CACHE}
        if cache_imu and not os.path.exists(self.cache_imu["path"]):
            os.makedirs(self.cache_imu["path"], exist_ok=True)
        
        self.window_sec = window_sec
        self.target_frames_in_window = target_frames_in_window
        self.meta_video = get_ego4d_metadata("video")
        self.video_cache = {}
        self.image_processor = image_processor

        bad_imus = []
        if window_sec == 5.0:
            path_bad_imu_json = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "bad_imu_windows_5.0.json"
            )
            bad_imus = load_json(path_bad_imu_json)
            bad_imus = set([f"{uid}_{start}_{end}" for uid, start, end in bad_imus])

        self.window_idx = []
        for video_uid in tqdm(self.meta_video.keys()):
            if not self.check_modality_clip_uid(video_uid):
                continue
            if video_uid in ['f49d9e6e-184c-41db-851b-9a00651dcef3', '863dd7a6-ea03-468e-9eb7-92c81ba0adb4']:
                continue
            
            video_duration = self.meta_video[video_uid]["video_metadata"]["video_duration_sec"]

            windows_in_clip = get_windows_in_clip(
                s_time=0,
                e_time=video_duration,
                window_sec=window_sec,
                stride=window_sec,
            )
            
            n_windows_per_video = 0
            if max_n_windows_per_video is not None and shuffle_windows:
                random.shuffle(windows_in_clip)

            for (w_s, w_e) in windows_in_clip:
                if f"{video_uid}_{w_s}_{w_e}" in bad_imus:
                    continue

                input_dict = {
                    "window_start": w_s,
                    "window_end": w_e,
                    "video_uid": video_uid,
                }
                if (
                    max_n_windows_per_video is not None
                    and n_windows_per_video >= max_n_windows_per_video
                ):
                    continue
                if window_sample_rate != 1.0 and random.random() > window_sample_rate:
                    continue

                self.window_idx.append(input_dict)
                n_windows_per_video += 1

    def check_modality_clip_uid(self, video_uid):
        has_imu, has_audio = modality_checker(self.meta_video[video_uid])
        if not has_imu:
            return False
        return True

    def __len__(self):
        return len(self.window_idx)

    def sample_frame_indices(self, clip_len, seg_len):
        sampling_rate = seg_len // clip_len
        sampled_indices = []
        for i in range(clip_len):
            start_idx = i * sampling_rate
            end_idx = min((i + 1) * sampling_rate, seg_len)
            random_frame_idx = np.random.randint(start_idx, end_idx)
            sampled_indices.append(random_frame_idx)
        return sampled_indices

    def _video_input(self, uid, w_s, w_e, target_frames_in_window):
        cache_path = Config.EGO4D_VIDEO_CACHE
        if os.path.exists(os.path.join(cache_path, f"{uid}_{w_s}_{w_e}.npz")):
            video = np.load(os.path.join(cache_path, f"{uid}_{w_s}_{w_e}.npz"), mmap_mode='r+')['arr_0']
        else:
            video_clip = VideoReader(uid, num_threads=20)
            video = video_clip.get_batch(range(int(10*w_s), int(10*w_e))).asnumpy()

        indices = self.sample_frame_indices(target_frames_in_window, len(video))
        frames = video[indices]
        video_input = self.image_processor(list(frames), do_rescale=False)
        return video_input

    def batch_feature_to_numpy(self, batch_feature):
        numpy_dict = {}
        for key, value in batch_feature.data.items():
            if isinstance(value, list):
                numpy_dict[key] = np.array(value)
            else:
                numpy_dict[key] = value.numpy()
        return numpy_dict

    def median_filter(self, true_mpu_datas, kernel_size=3):
        filter_datas = np.apply_along_axis(
            signal.medfilt, axis=0, arr=true_mpu_datas, kernel_size=kernel_size
        )
        return filter_datas.astype(np.float32)

    def z_score_normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1
        return (data - mean) / std

    def get_imu_frames(self, uid, video_start_sec, video_end_sec, cache):
        signal = load_npy(os.path.join(Config.EGO4D_DATA_PATH, f"processed_imu/{uid}.npy"))
        timestamps = load_npy(
            os.path.join(Config.EGO4D_DATA_PATH, f"processed_imu/{uid}_timestamps.npy")
        )
        signal = np.transpose(signal)

        signal = self.median_filter(np.array(signal), kernel_size=5)
        signal = self.z_score_normalize(signal)

        if toms(video_start_sec) > timestamps[-1] or toms(video_end_sec) > timestamps[-1]:
            return None

        start_id = bisect_left(timestamps, toms(video_start_sec))
        end_id = bisect_left(timestamps, toms(video_end_sec))

        if (
            delta(video_start_sec, tosec(timestamps[start_id])) > 4
            or delta(video_end_sec, tosec(timestamps[end_id])) > 4
        ):
            return None

        if start_id == end_id:
            start_id -= 1
            end_id += 1

        signal, timestamps = signal[start_id:end_id], timestamps[start_id:end_id]
        signal = padIMU(signal, video_end_sec - video_start_sec)

        sample_dict = {
            "signal": torch.tensor(signal, dtype=torch.float32),
            "sampling_rate": 50,
        }

        return sample_dict

    def __getitem__(self, idx):
        dict_out = copy.deepcopy(self.window_idx[idx])
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]

        input = {}

        video = self._video_input(
            uid=uid,
            w_s=w_s,
            w_e=w_e,
            target_frames_in_window=self.target_frames_in_window,
        )

        input['video'] = self.batch_feature_to_numpy(video)["pixel_values"]

        input["imu"] = self.get_imu_frames(
            uid=uid,
            video_start_sec=w_s,
            video_end_sec=w_e,
            cache=self.cache_imu,
        )['signal']

        return idx, input


class Ego4dDatasetUnsupervisedV2(torch.utils.data.Dataset):
    def __init__(
        self,
        image_processor,
        window_sec=5.0,
        target_frames_in_window=10,
        cache_imu=True,
        window_sample_rate=1.0,
        max_n_windows_per_video=None,
        shuffle_windows=True,
    ):
        self.cache_imu = {"cache": cache_imu, "path": Config.EGO4D_IMU_CACHE}
        if cache_imu and not os.path.exists(self.cache_imu["path"]):
            os.makedirs(self.cache_imu["path"], exist_ok=True)
        
        self.window_sec = window_sec
        self.target_frames_in_window = target_frames_in_window
        self.meta_video = get_ego4d_metadata("video")
        self.video_cache = {}
        self.image_processor = image_processor

        bad_imus = []
        if window_sec == 5.0:
            path_bad_imu_json = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "bad_imu_windows_5.0.json"
            )
            bad_imus = load_json(path_bad_imu_json)
            bad_imus = set([f"{uid}_{start}_{end}" for uid, start, end in bad_imus])

        self.window_idx = []
        for video_uid in tqdm(self.meta_video.keys()):
            if not self.check_modality_clip_uid(video_uid):
                continue
            if video_uid in ['f49d9e6e-184c-41db-851b-9a00651dcef3', '863dd7a6-ea03-468e-9eb7-92c81ba0adb4']:
                continue
            
            video_duration = self.meta_video[video_uid]["video_metadata"]["video_duration_sec"]

            windows_in_clip = get_windows_in_clip(
                s_time=0,
                e_time=video_duration,
                window_sec=window_sec,
                stride=window_sec,
            )
            
            n_windows_per_video = 0
            if max_n_windows_per_video is not None and shuffle_windows:
                random.shuffle(windows_in_clip)

            for (w_s, w_e) in windows_in_clip:
                if f"{video_uid}_{w_s}_{w_e}" in bad_imus:
                    continue

                input_dict = {
                    "window_start": w_s,
                    "window_end": w_e,
                    "video_uid": video_uid,
                }
                if (
                    max_n_windows_per_video is not None
                    and n_windows_per_video >= max_n_windows_per_video
                ):
                    continue
                if window_sample_rate != 1.0 and random.random() > window_sample_rate:
                    continue

                self.window_idx.append(input_dict)
                n_windows_per_video += 1

    def check_modality_clip_uid(self, video_uid):
        has_imu, has_audio = modality_checker(self.meta_video[video_uid])
        if not has_imu:
            return False
        return True

    def __len__(self):
        return len(self.window_idx)

    def sample_frame_indices(self, clip_len, seg_len):
        sampling_rate = seg_len // clip_len
        sampled_indices = []
        for i in range(clip_len):
            start_idx = i * sampling_rate
            end_idx = min((i + 1) * sampling_rate, seg_len)
            random_frame_idx = np.random.randint(start_idx, end_idx)
            sampled_indices.append(random_frame_idx)
        return sampled_indices

    def _video_input(self, uid, w_s, w_e, target_frames_in_window):
        cache_path = Config.EGO4D_VIDEO_CACHE
        if os.path.exists(os.path.join(cache_path, f"{uid}_{w_s}_{w_e}.npz")):
            video = np.load(os.path.join(cache_path, f"{uid}_{w_s}_{w_e}.npz"), mmap_mode='r+')['arr_0']
        else:
            video_clip = VideoReader(uid, num_threads=20)
            video = video_clip.get_batch(range(int(10*w_s), int(10*w_e))).asnumpy()

        indices = self.sample_frame_indices(target_frames_in_window, len(video))
        frames = video[indices]
        return video

    def median_filter(self, true_mpu_datas, kernel_size=3):
        filter_datas = np.apply_along_axis(
            signal.medfilt, axis=0, arr=true_mpu_datas, kernel_size=kernel_size
        )
        return filter_datas.astype(np.float32)

    def z_score_normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1
        return (data - mean) / std

    def get_imu_frames(self, uid, video_start_sec, video_end_sec, cache):
        signal = load_npy(os.path.join(Config.EGO4D_DATA_PATH, f"processed_imu/{uid}.npy"))
        timestamps = load_npy(
            os.path.join(Config.EGO4D_DATA_PATH, f"processed_imu/{uid}_timestamps.npy")
        )
        signal = np.transpose(signal)

        signal = self.median_filter(np.array(signal), kernel_size=5)
        signal = self.z_score_normalize(signal)

        if toms(video_start_sec) > timestamps[-1] or toms(video_end_sec) > timestamps[-1]:
            return None

        start_id = bisect_left(timestamps, toms(video_start_sec))
        end_id = bisect_left(timestamps, toms(video_end_sec))

        if (
            delta(video_start_sec, tosec(timestamps[start_id])) > 4
            or delta(video_end_sec, tosec(timestamps[end_id])) > 4
        ):
            return None

        if start_id == end_id:
            start_id -= 1
            end_id += 1

        signal, timestamps = signal[start_id:end_id], timestamps[start_id:end_id]
        signal = padIMU(signal, video_end_sec - video_start_sec)

        sample_dict = {
            "signal": torch.tensor(signal, dtype=torch.float32),
            "sampling_rate": 50,
        }

        return sample_dict

    def __getitem__(self, idx):
        dict_out = copy.deepcopy(self.window_idx[idx])
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]

        input = {}

        video = self._video_input(
            uid=uid,
            w_s=w_s,
            w_e=w_e,
            target_frames_in_window=self.target_frames_in_window,
        )

        input['video'] = video

        input["imu"] = self.get_imu_frames(
            uid=uid,
            video_start_sec=w_s,
            video_end_sec=w_e,
            cache=self.cache_imu,
        )['signal']

        return idx, input

