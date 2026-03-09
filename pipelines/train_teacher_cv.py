import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os, sys, argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import losses, miners

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    img_size = 300 if 'efficientnet' in args.backbone else 224
    save_dir = os.path.join("weights", args.backbone)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Chiến lược Mining & Loss (Ép tham số chuẩn gốc)
    miner_hard = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hardest")
    loss_metric_func = losses.TripletMarginLoss(margin=0.2)

    # 2. Loader (Ép n_classes=4, n_samples=4)
    train_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(
        PillDataset(df_train, transform=train_transform), 
        batch_sampler=BalancedBatchSampler(df_train['label_idx'].values, n_classes=4, n_samples=4),
        num_workers=4
    )

    model = PillTeacher(backbone_name=args.backbone, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 3. Scheduler: Ép giảm LR tại 60 và 85 (Chiến lược ổn định hội tụ)
    scheduler = MultiStepLR(optimizer, milestones=[60, 85], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_map = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} Epoch {epoch}")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device).long()
            
            with torch.amp.autocast('cuda'):
                norm_emb, logits_sce, logits_cos = model(imgs, labels)
                
                # Ép 3 loại Loss kết hợp
                hard_triplets = miner_hard(norm_emb, labels)
                l_metric = loss_metric_func(norm_emb, labels, hard_triplets)
                l_ce = F.cross_entropy(logits_sce, labels, label_smoothing=0.1)
                l_cos = F.cross_entropy(logits_cos * 16.0, labels)
                
                loss = (l_metric + l_ce + l_cos) / 2 # Accumulation=2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Đánh giá (Truyền đúng 4 tham số như hàm mới đã sửa)
        if epoch % 5 == 0 or epoch >= args.epochs - 2:
            val_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            val_loader = DataLoader(PillDataset(df_val, transform=val_transform), batch_size=16)
            ref_loader = DataLoader(PillDataset(df_ref, transform=val_transform), batch_size=16)
            
            metrics = evaluate_retrieval(model, val_loader, ref_loader, device)
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{f_idx}_best.pth"))
                print(f"⭐ Saved Best mAP: {best_map:.4f}")
            else:
                print(f"⭐ Skip cause mAP {metrics['mAP']} < {best_map:.4f}")

        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Teacher for Pill Retrieval")
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'convnext_base', 'efficientnet_v2_s'])
    parser.add_argument('--fold', type=int, default=0, help="Fold to train (0-3)")
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)
    
    # Logic chia Fold chuẩn bài báo
    fold = args.fold
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    df_train = df_all[(df_all['fold'] != fold) & (df_all['fold'] != 4) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    train_one_fold(args, fold, num_classes, df_train, df_val, df_ref_gallery, device)