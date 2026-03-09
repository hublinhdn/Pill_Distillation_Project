import torch
import torch.nn as nn
import torch.nn.functional as F
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

def train_one_fold(f_idx, num_classes, df_train, df_val, df_ref, device):
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

    model = PillTeacher(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    
    # Giảm LR mạnh tại các mốc cuối để ổn định hội tụ
    scheduler = MultiStepLR(optimizer, milestones=[60, 85], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # 2. Mining Khắc nghiệt: Triplet Hardest
    loss_metric_func = losses.TripletMarginLoss(margin=0.2)
    miner_hard = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hardest")
    
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_map = 0.0

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
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
                torch.save(model.state_dict(), f'weights/teacher_benchmark_f{f_idx}.pth')
                    
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