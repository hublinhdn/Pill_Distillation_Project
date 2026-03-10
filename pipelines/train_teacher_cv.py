import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import tqdm

# Sử dụng thư viện Metric Learning để khai thác Triplet Hardest
from pytorch_metric_learning import losses, miners

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    # Cấu hình chuẩn Benchmark cho 10GB VRAM
    n_classes, n_samples = 4, 4 
    accumulation_steps = 2 
    img_size = 300 if 'efficientnet' in args.backbone else 224

    save_dir = os.path.join("weights", args.backbone)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Augmentation chuẩn Benchmark (ImageNet Style)
    train_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loader sử dụng Sampler để đảm bảo luôn có cặp ảnh cùng loại trong batch
    train_loader = DataLoader(
        PillDataset(df_train, train_transform), 
        batch_sampler=BalancedBatchSampler(df_train['label_idx'].values, n_classes, n_samples),
        num_workers=4, pin_memory=True
    )
    
    # Eval loader gộp cả Val và Ref như code gốc của bạn
    val_loader = DataLoader(
        PillDataset(pd.concat([df_val, df_ref]), val_transform), 
        batch_size=32, shuffle=False, num_workers=4
    )

    model = PillTeacher(backbone_name=args.backbone, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    
    # Scheduler giảm LR tại mốc 60 và 85 để hội tụ sâu
    scheduler = MultiStepLR(optimizer, milestones=[60, 85], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # 2. Định nghĩa các loại Loss chiến lược
    loss_metric_func = losses.TripletMarginLoss(margin=0.2)
    miner_hard = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hardest")
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_map = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | Ep {epoch}")
        
        # --- CHIẾN LƯỢC ÉP LOSS ĐỘNG ---
        # Tăng trọng số Metric Loss sau Epoch 50 để tinh chỉnh không gian Embedding
        w_metric = 1.0 if epoch > 50 else 0.5
        
        for i, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device).long()
            
            with torch.amp.autocast('cuda'):
                # Lưu ý: Model trả về bộ 3 theo thứ tự: logits_sce, logits_csce, norm_emb
                logits_sce, logits_csce, norm_emb = model(imgs, labels)
                
                # Khai thác những bộ ba (Triplets) khó nhất trong batch
                indices_tuple = miner_hard(norm_emb, labels)
                l_metric = loss_metric_func(norm_emb, labels, indices_tuple)
                
                l_sce = criterion_sce(logits_sce, labels)
                l_csce = F.cross_entropy(logits_csce, labels)
                
                # Ép loss theo giai đoạn huấn luyện
                loss = (l_sce + l_csce + w_metric * l_metric) / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix({
                'L': f"{loss.item()*accumulation_steps:.2f}", 
                'W_m': w_metric,
                'Best': f"{best_val_map:.3f}"
            })

        scheduler.step()

        # Đánh giá định kỳ
        if (epoch % 5 == 0) or (epoch >= args.epochs - 5):
            torch.cuda.empty_cache()
            metrics = evaluate_retrieval(model, val_loader, device)
            curr_map = metrics['mAP']
            print(f"📊 Epoch {epoch} mAP (Flip): {curr_map:.4f} (R1: {metrics['Rank-1']:.4f})")

            if curr_map > best_val_map:
                best_val_map = curr_map
                save_path = os.path.join(save_dir, f"fold{f_idx}_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"⭐ BEST updated: {best_val_map:.4f}")
                    
    return best_val_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda")
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)
    
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == args.fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    df_train = df_all[(df_all['fold'] != args.fold) & (df_all['fold'] != 4) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    train_one_fold(args, args.fold, num_classes, df_train, df_val, df_ref_gallery, device)