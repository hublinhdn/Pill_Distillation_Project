import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import os
import sys
import numpy as np
import gc
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import losses, miners

# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillRetrievalModel
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    L_SCE, L_CSCE, L_TRIPLET, L_CONTRASTIVE = 1.0, 0.2, 1.0, 1.0
    
    # ĐỘNG: QUẢN LÝ VRAM THEO BACKBONE
    if 'convnext_large' in args.backbone:
        img_size = 384
        n_classes_batch, n_samples = 4, 2  
        accumulation_steps = 16            
        lr_backbone, lr_head = 2e-5, 2e-4
    elif 'convnext_base' in args.backbone:
        img_size = 384
        n_classes_batch, n_samples = 8, 2  
        accumulation_steps = 8             
        lr_backbone, lr_head = 3e-5, 3e-4
    elif 'resnet101' in args.backbone:
        img_size = 448
        n_classes_batch, n_samples = 8, 2  
        accumulation_steps = 8
        lr_backbone, lr_head = 3e-5, 3e-4
    elif 'resnet18' in args.backbone:
        img_size = 384
        n_classes_batch, n_samples = 16, 2 
        accumulation_steps = 4
        lr_backbone, lr_head = 4e-5, 4e-4
    else: 
        img_size = 448
        n_classes_batch, n_samples = 16, 2 
        accumulation_steps = 4
        lr_backbone, lr_head = 4e-5, 4e-4

    print(f"⚙️ Config {args.backbone}: Size={img_size}x{img_size} | Batch={n_classes_batch*n_samples} | Accum={accumulation_steps}")

    resize_scale = int(img_size * 1.15) 

    # DUAL TRANSFORM
    train_transform = transforms.Compose([
        transforms.Resize((resize_scale, resize_scale)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])
    
    train_transform_ref = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        PillDataset(df_train, transform=train_transform, transform_ref=train_transform_ref), 
        batch_sampler=BalancedBatchSampler(df_train['sub_label_idx'].values, n_classes_batch, n_samples),
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        PillDataset(pd.concat([df_val, df_ref]).reset_index(drop=True), transform=val_transform), 
        batch_size=n_classes_batch * n_samples, shuffle=False, num_workers=4, pin_memory=True
    )

    model = PillRetrievalModel(num_classes=num_classes, backbone_type=args.backbone,pooling_type=args.pooling).to(device)

    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")

    backbone_params = [p for n, p in model.named_parameters() if 'features' in n]
    head_params = [p for n, p in model.named_parameters() if 'features' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone}, 
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=5e-2)
    
    TOTAL_EPOCHS = 60 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_val_map = 0.0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | Ep {epoch}")
        
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            sub_labels = sub_labels.to(device).long()

            with torch.amp.autocast('cuda'):
                logits_sce, logits_csce, norm_embedding = model(imgs, labels=sub_labels)

                loss_sce = criterion_sce(logits_sce, sub_labels)
                loss_csce = criterion_sce(logits_csce, sub_labels)
                hard_pairs = miner(norm_embedding, sub_labels)
                loss_triplet = criterion_triplet(norm_embedding, sub_labels, hard_pairs)
                loss_contrastive = criterion_contrastive(norm_embedding, sub_labels)

                loss = (L_SCE * loss_sce + L_CSCE * loss_csce + L_TRIPLET * loss_triplet + L_CONTRASTIVE * loss_contrastive) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'L': f"{loss.item()*accumulation_steps:.3f}", 'Best': f"{best_val_map:.3f}"})

        scheduler.step()

        if (epoch % 5 == 0) or (epoch > TOTAL_EPOCHS - 10):
            # Mặc định sử dụng Both Sides để lưu Model tốt nhất
            val_metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=True)
            val_map = val_metrics['mAP']
            print(f"📊 Epoch {epoch} mAP: {val_map:.4f} (Rank-1: {val_metrics['Rank-1']:.4f})")
            
            if val_map > best_val_map:
                best_val_map = val_map
                os.makedirs('weights', exist_ok=True)
                torch.save(model.state_dict(), f"weights/best_{args.backbone}_{args.pooling}_fold{f_idx}.pth")
                
    return best_val_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='convnext_base,resnet50,mobilenet_v3_large')
    parser.add_argument('--pooling', type=str, default='gem', choices=['gem', 'mpncov']) # <-- Thêm dòng này
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df_all = load_epill_full_data()
    total_sub_classes = df_all['sub_label_idx'].nunique()

    fold = 0
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    cond_train = (df_all['fold'] != fold) & (df_all['is_ref'] == 0)
    train_labels = df_all[cond_train]['label_idx'].unique()
    cond_ref_train = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
    df_train = df_all[cond_train | cond_ref_train].reset_index(drop=True)
    
    backbone_list = [b.strip() for b in args.backbone.split(',')]
    print(f"🚀 BẮT ĐẦU CHIẾN DỊCH VỚI: {backbone_list}")
    
    results_summary = {}
    pooling = args.pooling

    for current_backbone in backbone_list:
        print(f"\n{'='*60}")
        print(f"🔥 ĐANG HUẤN LUYỆN: {current_backbone.upper()} - pooling: {pooling}")
        print(f"{'='*60}")
        
        args.backbone = current_backbone
        best_map = train_one_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)
        results_summary[current_backbone] = best_map
        
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("🏆 BÁO CÁO TỔNG KẾT (FOLD 0)")
    print("="*50)
    for bb, mAP in results_summary.items():
        print(f" - {bb.ljust(20)}: Best mAP = {mAP:.4f}")
    print("="*50)

    
    
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/batch_experiment_{pooling}_summary.txt", "w") as f:
        f.write(f"Batch Experiment Results for pooling: {pooling}\n")
        for bb, mAP in results_summary.items():
            f.write(f"{bb}: {mAP:.4f}\n")

if __name__ == '__main__':
    main()