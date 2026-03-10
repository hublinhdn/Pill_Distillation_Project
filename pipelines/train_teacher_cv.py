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
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data
from pytorch_metric_learning import losses, miners

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    # Cấu hình chuẩn Benchmark cho 10GB VRAM
    n_classes, n_samples = 4, 4 
    accumulation_steps = 2 
    
    # 1. Augmentation chuẩn Benchmark (ImageNet Style)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(PillDataset(df_train, train_transform), 
                              batch_sampler=BalancedBatchSampler(df_train['label_idx'].values, n_classes, n_samples),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(PillDataset(pd.concat([df_val, df_ref]), val_transform), 
                            batch_size=32, shuffle=False, num_workers=4)
    
    model = PillTeacher(num_classes=num_classes, backbone_type=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[60, 85], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # 2. Mining Khắc nghiệt: Triplet Hardest
    loss_metric_func = losses.TripletMarginLoss(margin=0.2)
    miner_hard = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hardest")
    
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_map = 0.0

    for epoch in range(1, 101):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | {args.backbone} | Ep {epoch}")
        # optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | Ep {epoch}")
        
        # Tăng trọng số Metric Loss sau Epoch 50
        w_metric = 1.0 if epoch > 50 else 0.5
        
        for i, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device).long()
            
            with torch.amp.autocast('cuda'):
                logits_sce, logits_csce, norm_emb = model(imgs, labels)
                
                # Khai thác những bộ ba (Triplets) khó nhất trong batch
                indices_tuple = miner_hard(norm_emb, labels)
                l_metric = loss_metric_func(norm_emb, labels, indices_tuple)
                
                l_sce = criterion_sce(logits_sce, labels)
                l_csce = F.cross_entropy(logits_csce, labels)
                
                loss = (l_sce + l_csce + w_metric * l_metric) / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix({'L': f"{loss.item()*2:.2f}", 'Best': f"{best_val_map:.3f}"})

        scheduler.step()

        if (epoch % 5 == 0) or (epoch >= 80):
            torch.cuda.empty_cache()
            metrics = evaluate_retrieval(model, val_loader, device)
            curr_map = metrics['mAP']
            print(f"📊 Epoch {epoch} mAP (Flip): {curr_map:.4f} (R1: {metrics['Rank-1']:.4f})")

            if curr_map > best_val_map:
                print(f"📊 Epoch {epoch} ==> BEST updated")
                best_val_map = curr_map
                torch.save(model.state_dict(), f'weights/teacher_cv_f{f_idx}.pth')
                    
    return best_val_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101', 'convnext_base'])
    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('weights', exist_ok=True)

    # ... (khởi tạo thiết bị và load data)
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)
    
    # 1. Tách biệt tập Reference gốc (toàn bộ) để dùng làm Gallery khi Eval
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    
    # 2. Xác định các fold dùng cho Cross-Validation (0, 1, 2, 3)
    # Loại bỏ Fold 4 (Hold-out) khỏi quá trình train/val này
    cv_folds = [0, 1, 2, 3]

    # --- BIẾN LƯU TRỮ REPORT ---
    results_summary = []
    
    for fold in cv_folds:
        print(f"\n🚀 Training Fold {fold}...")
        
        # --- TẬP VAL ---
        # Chỉ gồm ảnh Consumer của fold hiện tại
        df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
        
        # --- TẬP TRAIN ---
        # Điều kiện 1: Là ảnh Consumer của các fold CV khác (không phải fold hiện tại và không phải fold 4)
        cond_cons = (df_all['fold'].isin(cv_folds)) & (df_all['fold'] != fold) & (df_all['is_ref'] == 0)
        
        # Điều kiện 2: Là ảnh Reference NHƯNG chỉ của những nhãn (labels) xuất hiện trong cond_cons
        # Điều này đảm bảo mô hình không "nhìn trộm" mặt chuẩn của thuốc trong tập Val
        train_labels = df_all[cond_cons]['label_idx'].unique()
        cond_ref = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
        
        df_train = df_all[cond_cons | cond_ref].reset_index(drop=True)
        
        # Chạy huấn luyện
        best_map = train_one_fold(args, fold, num_classes, df_train, df_val, df_ref_gallery, device)
        # Lưu lại kết quả của fold vào danh sách
        results_summary.append({
            'fold': fold,
            'best_mAP': best_map
        })
        print(f"✅ Fold {fold} Finished. Best mAP: {best_map:.4f}")
    # --- TỔNG HỢP VÀ XUẤT REPORT ---
    report_path = f"reports/final_report_{args.backbone}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"PILL IDENTIFICATION REPORT - BACKBONE: {args.backbone}\n")
        f.write("="*50 + "\n")
        
        all_maps = [res['best_mAP'] for res in results_summary]
        for res in results_summary:
            line = f"Fold {res['fold']}: Best mAP = {res['best_mAP']:.4f}\n"
            f.write(line)
            print(line, end="") # In ra màn hình luôn
            
        mean_map = np.mean(all_maps)
        std_map = np.std(all_maps)
        
        summary_line = f"\nAVERAGE mAP: {mean_map:.4f} (+/- {std_map:.4f})\n"
        f.write("-" * 20 + summary_line)
        f.write("="*50 + "\n")
        print(summary_line)

    print(f"📝 Report đã được lưu tại: {report_path}")

if __name__ == '__main__':
    main()