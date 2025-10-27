import os

class Config:
    EGO4D_DATA_PATH = "/scr/Saeid/ego4d_data/processed/full_videos"
    EGO4D_VIDEO_CACHE = "/scr/Saeid/ego4d_data/video_clips"
    EGO4D_IMU_CACHE = "/scr/Saeid/ego4d_data/video_imu"
    
    CHECKPOINT_DIR = "./checkpoint"
    FINAL_EXP_DIR = "./final_exp"
    
    PRETRAINED_VIDEOMAE = "MCG-NJU/videomae-base-ssv2"
    PRETRAINED_PATCHTST = "namctin/patchtst_etth1_pretrain"
    PRETRAINED_CLIP = "ViT-B/32"
    
    DEFAULT_SEED = 42
    
    IMU_SAMPLING_RATE = 50
    VIDEO_FPS = 10
    
    LOSS_TYPE = 'sigmoid'
    
    @staticmethod
    def get_checkpoint_path(filename):
        return os.path.join(Config.CHECKPOINT_DIR, filename)
    
    @staticmethod
    def get_final_exp_path(filename):
        return os.path.join(Config.FINAL_EXP_DIR, filename)

