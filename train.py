import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoImageProcessor
from tqdm import tqdm
from Ego4D.ego4d_dataset import Ego4dDatasetUnsupervisedV2, Ego4dDatasetUnsupervised
from dataset import IMUVideoDataset, IMUVideoDatasetV2
from model import IMUVideoCrossModel, IMU2CLIP
from config import Config
from utils import AvgMeter, get_lr, get_filepaths

os.environ['OMP_NUM_THREADS'] = '2'

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

dist.init_process_group("nccl", rank=local_rank, world_size=int(os.environ["WORLD_SIZE"]))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for IMU-Video contrastive learning')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['imu_video_cross', 'imu2clip'],
                        help='Model architecture: imu_video_cross (IMUVideoCrossModel) or imu2clip (IMU2CLIP)')
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ego4d', 'mmea'],
                        help='Dataset: ego4d or mmea')
    
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to MMEA dataset (required if dataset=mmea)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers')
    
    parser.add_argument('--clip_len', type=int, default=10,
                        help='Number of frames per video clip')
    
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save model every N epochs')
    
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    scaler = torch.cuda.amp.GradScaler()

    for batch in tqdm_object:
        _, input_data = batch
        input_data['video'] = torch.squeeze(input_data['video'], 1)
        input_data['imu'] = torch.squeeze(input_data['imu'], 1)
        input_data = {k: v.to(device) for k, v in input_data.items()}

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(input_data)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == "batch":
            lr_scheduler.step()

        count = input_data['video'].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    
    return loss_meter.avg


def main():
    args = parse_args()
    
    seed = Config.DEFAULT_SEED
    set_seed(seed)

    image_processor = AutoImageProcessor.from_pretrained(Config.PRETRAINED_VIDEOMAE)
    
    if args.dataset == 'ego4d':
        if args.model == 'imu_video_cross':
            train_data = Ego4dDatasetUnsupervised(image_processor, 5, args.clip_len, max_n_windows_per_video=None)
        else:
            train_data = Ego4dDatasetUnsupervisedV2(image_processor, 5, args.clip_len, max_n_windows_per_video=None)
    else:
        if not args.data_path:
            raise ValueError("--data_path is required for MMEA dataset")
        
        video_files = get_filepaths(args.data_path)
        
        if args.model == 'imu_video_cross':
            train_data = IMUVideoDataset(
                video_filenames=video_files,
                clip_len=args.clip_len,
                image_processor=image_processor,
                avg_imu_length=512,
                window_size=512
            )
        else:
            imu_files = [f.replace('video', 'sensor').replace('mp4', 'csv') for f in video_files]
            train_data = IMUVideoDatasetV2(
                imu_filenames=imu_files,
                image_processor=image_processor,
                clip_len=args.clip_len,
                window_size=512
            )

    sampler = DistributedSampler(
        train_data,
        shuffle=True,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank()
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=5,
        persistent_workers=True,
        drop_last=True
    )

    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    if args.model == 'imu_video_cross':
        model = IMUVideoCrossModel(temperature=np.log(10), bias=-10, loss_type=Config.LOSS_TYPE)
    else:
        model = IMU2CLIP(temperature=0.1)
    
    model = model.float()
    model = model.to('cuda:' + str(local_rank))
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=50,
        find_unused_parameters=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_loader)
    tmax = args.epochs * steps_per_epoch

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tmax, eta_min=1e-7)

    step = "batch"
    train_losses = []
    best_loss = float('inf')
    
    model_name = 'imu_video_cross' if args.model == 'imu_video_cross' else 'imu2clip'
    dataset_name = args.dataset

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}/{args.epochs} | Model: {model_name} | Dataset: {dataset_name}")
        model.train()
        sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, local_rank)
        train_losses.append(train_loss)

        if (dist.get_rank() == 0) and ((epoch + 1) % args.save_interval == 0):
            save_path = Config.get_final_exp_path(f"{model_name}_{dataset_name}_{epoch + 1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print(f'Best loss: {best_loss}')


if __name__ == '__main__':
    main()

