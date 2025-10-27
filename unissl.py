import os
import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    Trainer,
    TrainingArguments,
    PatchTSTForPretraining
)
from transformers import AutoImageProcessor
from dataset import IMUDataset
from config import Config


def main():
    np.random.seed(Config.DEFAULT_SEED)

    with open('remaining_class_files.txt', 'r') as f:
        ssl_data = [line.strip() for line in f.readlines()]

    image_processor = AutoImageProcessor.from_pretrained(Config.PRETRAINED_VIDEOMAE)

    context_length = 512
    forecast_horizon = 96
    patch_length = 16
    num_workers = 20
    batch_size = 512
    avg_imu_length = 512
    classes = np.unique([path.split('/')[6] for path in ssl_data])

    train_dataset = IMUDataset(
        ssl_data,
        classes=classes,
        avg_imu_length=avg_imu_length
    )

    config = PatchTSTConfig(
        num_input_channels=6,
        context_length=250,
        patch_length=16,
        patch_stride=16,
        mask_type='random',
        random_mask_ratio=0.4,
        use_cls_token=True,
    )
    
    model = PatchTSTForPretraining(config)

    training_args = TrainingArguments(
        output_dir=Config.get_checkpoint_path("patchtst-pd/pretrain/output/"),
        overwrite_output_dir=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=100,
        per_device_train_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir=Config.get_checkpoint_path("patchtst-pd/pretrain/logs/"),
        greater_is_better=False,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()

