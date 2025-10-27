import os
import json
import random
import argparse
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from model import LinearProbing, IMU2CLIP, IMUVideoCrossModel, IMUPTST
from dataset import IMUDataset, PatientIMUPerSec
from utils import AvgMeter, get_lr, get_filepaths
from config import Config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, step, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        input_data, labels, _ = batch
        input_data = torch.squeeze(input_data, 1).to(device)
        input_data = input_data.permute(0, 2, 1)
        labels = labels.to(device)

        logits = model(input_data)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = input_data.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, criterion, valid_loader, device):
    loss_meter = AvgMeter()
    n_correct = 0
    n_samples = 0
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    pred_list = []
    gt_list = []

    for batch in tqdm_object:
        input_data, labels, _ = batch
        input_data = torch.squeeze(input_data, 1).to(device)
        input_data = input_data.permute(0, 2, 1)
        labels = labels.to(device)

        logits = model(input_data)
        loss = criterion(logits, labels)

        _, preds = logits.max(1)
        _, gt = labels.max(1)
        gt_list.append(gt.cpu().numpy())
        pred_list.append(preds.cpu().numpy())
        n_correct += (preds == gt).sum().item()
        n_samples += labels.size(0)

        count = input_data.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter, n_correct / n_samples, np.concatenate(pred_list), np.concatenate(gt_list)


def calculate_per_class_accuracy(y_true, y_pred, classes):
    per_class_acc = {}
    for i, class_name in enumerate(classes):
        class_indices = (y_true == i)
        if np.sum(class_indices) > 0:
            accuracy = np.sum((y_pred == i) & class_indices) / np.sum(class_indices)
            per_class_acc[class_name] = accuracy
        else:
            per_class_acc[class_name] = 0.0
    return per_class_acc


def parse_args():
    parser = argparse.ArgumentParser(description='Downstream task training (linear probing / finetuning)')
    
    parser.add_argument('--model', type=str, default='clip', 
                        choices=['imu2clip', 'clip', 'imuptst'],
                        help='Model architecture')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained checkpoint')
    
    parser.add_argument('--mode', type=str, default='lp',
                        choices=['lp', 'finetune'],
                        help='Linear probing or finetuning')
    
    parser.add_argument('--dataset', type=str, default='mmea',
                        choices=['mmea', 'patient_persec'],
                        help='Dataset type: mmea (UESTC-MMEA-CL), patient_persec (PatientIMUPerSec)')
    
    parser.add_argument('--backbone_lr', type=float, default=1e-6,
                        help='Learning rate for backbone')
    
    parser.add_argument('--classifier_lr', type=float, default=1e-3,
                        help='Learning rate for classifier head')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for training')
    
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of data loading workers')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Number of samples per class for training')
    
    parser.add_argument('--val_samples', type=int, default=20,
                        help='Number of samples per class for validation')
    
    
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save best model (optional)')
    
    parser.add_argument('--report', action='store_true',
                        help='Print detailed classification report')
    
    return parser.parse_args()


def load_mmea_data(args):
    with open('ssl_data.txt', 'r') as filehandle:
        ssl_data = json.load(filehandle)

    with open('sl_data.txt', 'r') as filehandle:
        sl_data = json.load(filehandle)

    data = ssl_data + sl_data
    
    specific_classes = {"12_standing", '20_walking', '29_sit_stand', '2_downstairs', '4_fall'}
    filtered_data = [path for path in data if path.split('/')[6] in specific_classes]
    classes = np.unique([path.split('/')[6] for path in filtered_data])

    train_data = []
    val_data = []

    np.random.seed(args.seed)
    for cls in classes:
        class_samples = [path for path in filtered_data if path.split('/')[6] == cls]
        np.random.shuffle(class_samples)
        val_data.extend(class_samples[:args.val_samples])
        shuffle_samples = class_samples[args.val_samples:]
        np.random.seed(args.seed)
        np.random.shuffle(shuffle_samples)
        train_data.extend(shuffle_samples[:args.train_samples])

    train_dataset = IMUDataset(
        train_data, 
        512, 
        classes=classes, 
        resample=50, 
        window_size=250
    )
    val_dataset = IMUDataset(
        val_data, 
        512, 
        classes=classes, 
        resample=50, 
        window_size=250
    )

    return train_dataset, val_dataset, classes, train_data, val_data


def load_patient_persec_data(args):
    imu_path = get_filepaths('/home/saeid/datasets/Patient_IMU')
    
    classes = np.array(['bending', 'walking', 'standing', 'turning', 'sitting'])
    
    train_data = []
    val_data = []
    
    for cls in classes:
        class_samples = [path for path in imu_path if path.split('/')[6] == cls]
        np.random.seed(args.seed)
        np.random.shuffle(class_samples)
        val_data.extend(class_samples[:args.val_samples])
        train_data.extend(class_samples[args.val_samples:args.val_samples + args.train_samples])
    
    train_dataset = PatientIMUPerSec(
        train_data, 
        250, 
        classes, 
        'VA_12', 
        window_size=250,
        stratified=100
    )
    val_dataset = PatientIMUPerSec(
        val_data, 
        250, 
        classes, 
        'VA_12', 
        window_size=250,
        stratified=20
    )
    
    return train_dataset, val_dataset, classes, train_data, val_data


def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    set_seed(args.seed)

    if args.dataset == 'mmea':
        train_dataset, val_dataset, classes, train_data, val_data = load_mmea_data(args)
    elif args.dataset == 'patient_persec':
        train_dataset, val_dataset, classes, train_data, val_data = load_patient_persec_data(args)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")

    print(f"Dataset: {args.dataset}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Classes: {classes}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    trainable = (args.mode == 'finetune')
    
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Mode: {args.mode} (Trainable: {trainable})")

    if args.model == 'imuptst':
        model = IMUPTST(
            128*6, 
            len(classes), 
            model_name=args.checkpoint,
            pretrained=True,
            trainable=trainable
        ).to(device)
        
        params = [
            {"params": model.classifier.parameters(), "lr": args.classifier_lr},
            {"params": model.model.parameters(), "lr": args.backbone_lr},
        ]
        optimizer = torch.optim.AdamW(params)
        
    else:
        if args.model == 'imu2clip':
            clip = IMU2CLIP().to(device)
        else:
            clip = IMUVideoCrossModel().to(device)
        
        ddp_dict = torch.load(args.checkpoint)
        new_state_dict = OrderedDict()
        for key, value in ddp_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        
        clip.load_state_dict(new_state_dict)

        model = LinearProbing(
            clip, 
            512, 
            len(classes), 
            trainable=trainable,
            joint_emd=False
        ).to(device)

        params = [
            {"params": model.clip_model.parameters(), "lr": args.backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.classifier_lr},
        ]
        optimizer = torch.optim.AdamW(params)

    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    tmax = args.epochs * steps_per_epoch

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tmax, eta_min=1e-7)

    step = "batch"
    best_preds = None
    best_gt = None
    best_loss = float('inf')
    best_acc = float('-inf')
    best_bal_val = float('-inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch + 1}/{args.epochs}")
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, step, device)
        
        model.eval()
        with torch.no_grad():
            valid_loss, val_acc, preds, gts = valid_epoch(model, criterion, val_loader, device)
            bal_acc = balanced_accuracy_score(gts, preds)
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Balanced Accuracy: {bal_acc:.4f}")

        if best_acc < val_acc:
            best_acc = val_acc
            best_preds = preds
            best_gt = gts
            best_bal_val = bal_acc
            
            if args.save_model:
                save_path = args.save_model
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f'Best Loss: {best_loss:.4f}')
    print(f'Best Accuracy: {best_acc:.4f}')
    print(f'Best Balanced Accuracy: {best_bal_val:.4f}')
    
    best_f1 = f1_score(best_gt, best_preds, average='macro')
    print(f"Best F1 Score (macro): {best_f1:.4f}")
    
    if args.report:
        print("\nPer-Class Accuracy:")
        per_class_acc = calculate_per_class_accuracy(best_gt, best_preds, classes)
        for class_name, acc in per_class_acc.items():
            print(f"  {class_name}: {acc:.4f}")
        
        print("\nClassification Report:")
        report = classification_report(best_gt, best_preds, target_names=classes, zero_division=0)
        print(report)


if __name__ == '__main__':
    main()
