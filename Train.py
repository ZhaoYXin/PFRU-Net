import logging
import argparse
import numpy as np
import cv2
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR,SequentialLR, LinearLR

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import KFold

from Data_utils import BUSIDataset, read_filelist_from_txt
from metrics import Dice, IoU, Precision, Recall, Specificity, PA,F1_socre
from loss import FocalTverskyLoss,ComboLoss,BCEDiceLoss,TverskyLoss
from model.PFRUNet import PFRUNet

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    return logging.getLogger()

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def train(args):
    seed_value = args.seed
    set_random_seed(seed_value)
    EPOCHS = args.epochs
    BS = args.batch_size
    N_FOLDS = 4
    weight_decay=args.weight_decay
    filepath = args.filepath
    trainfilepath = args.train_filepath

    log_dir_base = os.path.join(args.log_dir)
    logger = setup_logging(log_dir_base)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info(f"Using model:{args.modelName}")
    logger.info(f"lr={args.lr}, BS={BS}, EPOCHS={EPOCHS}, N_FOLDS={N_FOLDS},weight_decay={weight_decay}")
    logger.info(f"loss=ComboLoss{[0.7,0.3],(0.5,0.5)}, scheduler=cos")

    logger.info(f"Starting {N_FOLDS}-fold cross-validation...")
    all_folds_test_metrics = []
    all_folds_test_metrics1 = []
    focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=4/3)
    bce_loss = nn.BCEWithLogitsLoss()
    for fold in range(1, N_FOLDS + 1):
        seed_value = args.seed
        set_random_seed(seed_value)
        logger.info(f'{"-" * 20} FOLD {fold}/{N_FOLDS} {"-" * 20}')
        fold_log_dir = os.path.join(log_dir_base, f'fold_{fold}')
        writer = SummaryWriter(log_dir=fold_log_dir)

        fold_train_dir = os.path.join(trainfilepath, f'fold_{fold}')
        train_files_path = os.path.join(args.fold_dir, f'fold_{fold}', 'train.txt')
        valid_files_path = os.path.join(args.fold_dir, f'fold_{fold}', 'val.txt')
        test_files_path = os.path.join(args.fold_dir, f'fold_{fold}', 'test.txt')

        train_files_fold = read_filelist_from_txt(train_files_path)
        val_files_fold = read_filelist_from_txt(valid_files_path)
        test_files = read_filelist_from_txt(test_files_path)

        logger.info(f"Fold {fold} - Training files: {len(train_files_fold)}, Validation files: {len(val_files_fold)}, Testing files: {len(test_files)}")

        train_dataset_fold = BUSIDataset(filepath, train_files_fold,augment=True)
        val_dataset_fold = BUSIDataset(filepath, val_files_fold,augment=False)
        test_dataset = BUSIDataset(filepath, test_files,augment=False)
        train_loader = DataLoader(train_dataset_fold, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

        model = PFRUNet().to(device)
        criterion = ComboLoss(
            losses=[focal_tversky, bce_loss],
            weights=[0.5, 0.5]
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        warmup_epochs = args.warmup_epochs
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

        cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        best_val_dice = 0.0
        history = {'loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            trigger = 0
            num_train_images = len(train_dataset_fold)
            with tqdm(total=num_train_images, desc=f"Fold {fold} Epoch {epoch + 1}/{EPOCHS}", unit='img') as pbar:
                for images, masks, _ in train_loader:
                    images, masks = images.to(device), masks.to(device)

                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, masks)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    pbar.update(images.shape[0])
                    pbar.set_postfix(loss=loss.item())

            avg_train_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_train_loss)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            model.eval()
            val_loss = 0
            val_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'specificity': [], 'f1': [],'ACC': []} #,'auc': []
            with torch.no_grad():
                for images, masks,_ in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
                    preds = (outputs > args.threshold).float()

                    for metric_name, metric_func in [('dice', Dice), ('iou', IoU), ('precision', Precision),
                                                     ('recall', Recall), ('specificity', Specificity),
                                                     ('f1', F1_socre), ('ACC', PA)]:
                        val_metrics[metric_name].append(metric_func(masks, preds))

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)

            avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
            history['val_dice'].append(avg_val_metrics['dice'])
            history['val_iou'].append(avg_val_metrics['iou'])
            writer.add_scalar('Dice/val', avg_val_metrics['dice'], epoch)
            writer.add_scalar('IoU/val', avg_val_metrics['iou'], epoch)

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"lr: {current_lr:.6e} - Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} "
                        f"- Val Dice: {avg_val_metrics['dice']:.4f}, Val IoU: {avg_val_metrics['iou']:.4f}")

            trigger += 1

            if avg_val_metrics['dice'] > best_val_dice:
                best_val_dice = avg_val_metrics['dice']
                fold_save_dir = os.path.join(args.save_dir, f'fold_{fold}')
                os.makedirs(fold_save_dir, exist_ok=True)
                model_path = os.path.join(fold_save_dir, 'best_model.pth')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'val_dice': best_val_dice}, model_path)
                logger.info(f"Best model for fold {fold} saved to {model_path} with Dice: {best_val_dice:.4f}")
                trigger = 0

            if args.early_stopping >= 0 and trigger >= args.early_stopping:
                print("=> early stopping")
                break

        fold_save_dir = os.path.join(args.save_dir, f'fold_{fold}')
        os.makedirs(fold_save_dir, exist_ok=True)
        model_path = os.path.join(fold_save_dir, f'fold_{fold }.pth')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'val_dice': avg_val_metrics['dice']},
                   model_path)
        logger.info(
            f"Last model for fold {fold} saved to {model_path} with Dice: {avg_val_metrics['dice']:.4f}")

        writer.close()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['val_dice'], label='Validation Dice')
        plt.plot(history['val_iou'], label='Validation IoU')
        plt.title(f'Fold {fold} Metric Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(fold_log_dir, 'training_curves.png')
        plt.savefig(plot_path)
        logger.info(f"Training curves for fold {fold} saved to {plot_path}")
        plt.close()

        best_model_path = os.path.join(args.save_dir, f'fold_{fold}', 'best_model.pth')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        test_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'specificity': [], 'f1': [],'ACC': []}  #,'auc': []
        with torch.no_grad():
            for images, masks, _ in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = (outputs > args.threshold).float()
                for metric_name, metric_func in [('dice', Dice), ('iou', IoU), ('precision', Precision),
                                                 ('recall', Recall), ('specificity', Specificity), ('f1', F1_socre), ('ACC', PA)]:  #, ('auc', AUC)
                    test_metrics[metric_name].append(metric_func(masks, preds))

        avg_test_metrics = {k: np.mean(v) for k, v in test_metrics.items()}
        all_folds_test_metrics.append(avg_test_metrics)
        logger.info(
            f"Fold {fold} 测试集评估结果: Dice={avg_test_metrics['dice']:.4f}, IoU={avg_test_metrics['iou']:.4f}, "
            f"Precision={avg_test_metrics['precision']:.4f}, Recall={avg_test_metrics['recall']:.4f}, "
            f"Specificity={avg_test_metrics['specificity']:.4f}, F1={avg_test_metrics['f1']:.4f},"
            f"ACC={avg_test_metrics['ACC']:.4f}")

        best_model_path = os.path.join(args.save_dir, f'fold_{fold}', f'fold_{fold}.pth')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        test_metrics1 = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'specificity': [], 'f1': [],'ACC': []}
        with torch.no_grad():
            for images, masks, _ in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = (outputs > args.threshold).float()
                for metric_name, metric_func in [('dice', Dice), ('iou', IoU), ('precision', Precision),
                                                 ('recall', Recall), ('specificity', Specificity), ('f1', F1_socre), ('ACC', PA)]:
                    test_metrics1[metric_name].append(metric_func(masks, preds))

        avg_test_metrics1 = {k: np.mean(v) for k, v in test_metrics1.items()}
        all_folds_test_metrics1.append(avg_test_metrics1)
        logger.info(
            f"Fold {fold} 测试集评估结果: Dice={avg_test_metrics1['dice']:.4f}, IoU={avg_test_metrics1['iou']:.4f}, "
            f"Precision={avg_test_metrics1['precision']:.4f}, Recall={avg_test_metrics1['recall']:.4f}, "
            f"Specificity={avg_test_metrics1['specificity']:.4f}, F1={avg_test_metrics1['f1']:.4f},"
            f"ACC={avg_test_metrics1['ACC']:.4f}")



    logger.info(f'{"-" * 20} 4-Fold Cross-Validation Best Results {"-" * 20}')
    mean_metrics = {k: 0.0 for k in all_folds_test_metrics[0].keys()}
    for i, metrics in enumerate(all_folds_test_metrics):
        logger.info(f"Fold {i + 1} Test Metrics: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                    f"Specificity={metrics['specificity']:.4f}, F1={metrics['f1']:.4f},"
                    f"ACC={metrics['ACC']:.4f}")
        for k in mean_metrics:
            mean_metrics[k] += metrics[k]
    for k in mean_metrics:
        mean_metrics[k] /= N_FOLDS
    logger.info("Average Test Metrics across all folds:")
    for name, value in mean_metrics.items():
        logger.info(f"{name.capitalize()}: {value:.4f}")



    logger.info(f'{"-" * 20} 4-Fold Cross-Validation Final Results {"-" * 20}')
    mean_metrics1 = {k: 0.0 for k in all_folds_test_metrics1[0].keys()}
    for i, metrics in enumerate(all_folds_test_metrics1):
        logger.info(f"Fold {i + 1} Test Metrics: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                    f"Specificity={metrics['specificity']:.4f}, F1={metrics['f1']:.4f},"
                    f"ACC={metrics['ACC']:.4f}")
        for k in mean_metrics1:
            mean_metrics1[k] += metrics[k]
    for k in mean_metrics1:
        mean_metrics1[k] /= N_FOLDS
    logger.info("Average Test Metrics across all folds:")
    for name, value in mean_metrics1.items():
        logger.info(f"{name.capitalize()}: {value:.4f}")
    logger.info("训练完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName',type=str,default='PFRUNet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay',type =float,default=8e-3)
    parser.add_argument('--filepath', type=str, default='/tmp/pycharm_project_980/dataset/BUSI_processed')
    parser.add_argument('--train_filepath', type=str, default='/tmp/pycharm_project_980/dataset/BUSI')
    parser.add_argument('--fold_dir', type=str, default='/tmp/pycharm_project_980/dataset/BUSI')
    parser.add_argument('--save_dir', type=str, default='/tmp/pycharm_project_980/results/PFRUNet')
    parser.add_argument('--log_dir', type=str, default='/tmp/pycharm_project_980/TC/BUSI')
    parser.add_argument('--early_stopping', type=int, default=-1,metavar='N', help='early stopping (default: -1)')

    args = parser.parse_args()

    seed_value = args.seed
    set_random_seed(seed_value)

    print("参数设置:")
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"  {key}: {value}")
    print("-" * 20)
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)