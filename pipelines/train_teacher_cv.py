import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
import numpy as np
import gc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset, get_transforms, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data
from pytorch_metric_learning import losses, miners

def train_one_fold(f_idx, num_classes, df_train, df_val, df_ref, device):
    # Cấu hình Upsize ngầm & Mixed Precision
    n_classes = 4 
    n_samples = 4 
    accumulation_steps = 2 
    warmup_epochs = 5
    patience = 20
    
    sampler = BalancedBatchSampler(df_train['label_idx'].values, n_classes=n_classes, n_samples=n_samples)
    train_loader = DataLoader(PillDataset(df_train, get_transforms(is_train=True, size=224)), 
                              batch_sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PillDataset(pd.concat([df_val, df_ref]), get_transforms(is_train=False, size=224)), 
                            batch_size=32, shuffle=False, num_workers=4)

    model = PillTeacher(num_classes=num_classes).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 2e-5},
        {'params': model.reduce_conv.parameters(), 'lr': 2e-4},
        {'params': model.fc_projection.parameters(), 'lr': 2e-4},
        {'params': model.fc_ce.parameters(), 'lr': 2e-4},
        {'params': [model.proxy_cos], 'lr': 4e-4} # Tăng LR cho Proxy để hội tụ nhanh hơn
    ], weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # Loss functions
    loss_metric_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5) # Thông số chuẩn SOTA
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1) # Thêm Label Smoothing như benchmark
    criterion_csce = nn.CrossEntropyLoss()

    best_val_map = 0.0
    counter = 0

    for epoch in range(1, 101):
        model.train()
        
        # --- CHIẾN LƯỢC KHẮT KHE ---
        # 1. Mining linh động: Càng về sau càng chọn mẫu khó hơn
        if epoch < 40:
            current_epsilon = 0.1
        elif epoch < 70:
            current_epsilon = 0.05
        else:
            current_epsilon = 0.02 # Cực kỳ khắt khe
            
        miner = miners.MultiSimilarityMiner(epsilon=current_epsilon)
        
        # 2. Trọng số Loss linh động
        # Sau epoch 50, tăng mạnh Metric Loss và Proxy Loss
        w_metric = 0.5 if epoch < 50 else 1.0
        w_csce = 0.5 if epoch < 50 else 1.2 # Đẩy mạnh phân cực embedding
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | Ep {epoch} [eps={current_epsilon}]")
        
        for i, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device).long()
            
            with torch.amp.autocast('cuda'):
                logits_sce, logits_csce, norm_emb = model(imgs, labels)
                
                # Metric Loss với Hard Negative Mining
                indices_tuple = miner(norm_emb, labels)
                l_metric = loss_metric_func(norm_emb, labels, indices_tuple)
                
                # Classification Losses
                l_sce = criterion_sce(logits_sce, labels)
                l_csce = criterion_csce(logits_csce, labels)
                
                # Tổng hợp Loss theo trọng số chiến lược
                total_loss = (l_sce + w_csce * l_csce + w_metric * l_metric) / accumulation_steps

            scaler.scale(total_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix({
                'L': f"{total_loss.item()*accumulation_steps:.2f}",
                'm_w': w_metric,
                'Best': f"{best_val_map:.3f}"
            })

        if epoch > warmup_epochs:
            scheduler.step()

        # Đánh giá định kỳ hoặc ở cuối
        if (epoch % 5 == 0) or (epoch >= 80):
            torch.cuda.empty_cache()
            gc.collect()
            metrics = evaluate_retrieval(model, val_loader, device)
            curr_map = metrics['mAP']
            print(f"📊 Epoch {epoch} mAP: {curr_map:.4f} (Rank-1: {metrics.get('Rank-1', 0):.4f})")

            if curr_map > best_val_map:
                best_val_map = curr_map
                counter = 0
                torch.save(model.state_dict(), f'weights/teacher_upsize_khat_khe_f{f_idx}.pth')
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break
                    
    return best_val_map

def main():
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('weights', exist_ok=True)
    
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)
    
    df_ref = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_train = df_all[df_all['fold'] != 0].reset_index(drop=True) # Ví dụ Fold 0
    df_val = df_all[df_all['fold'] == 0].reset_index(drop=True)
    
    train_one_fold(0, num_classes, df_train, df_val, df_ref, device)

if __name__ == '__main__':
    main()