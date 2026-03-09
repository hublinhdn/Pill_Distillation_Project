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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset, get_transforms, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data
from pytorch_metric_learning import losses, miners

def train_one_fold(f_idx, num_classes, df_train, df_val, df_ref, device):
    # Cấu hình "Upsize ngầm"
    n_classes = 4 # Mỗi batch lấy 4 loại thuốc
    n_samples = 4 # Mỗi loại 4 ảnh -> Batch thật = 16
    accumulation_steps = 2 # Batch ảo = 32
    
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
        {'params': [model.proxy_cos], 'lr': 2e-4}
    ], weight_decay=1e-2)

    scaler = torch.cuda.amp.GradScaler()
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    loss_func = losses.MultiSimilarityLoss()
    criterion_sce = nn.CrossEntropyLoss()
    criterion_csce = nn.CrossEntropyLoss()

    best_val_map = 0.0
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | Ep {epoch}")
        
        for i, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device).long()
            
            with torch.cuda.amp.autocast():
                logits_sce, logits_csce, norm_emb = model(imgs, labels)
                indices_tuple = miner(norm_emb, labels)
                l_metric = loss_func(norm_emb, labels, indices_tuple)
                l_sce = criterion_sce(logits_sce, labels)
                l_csce = criterion_csce(logits_csce, labels)
                loss = (l_sce + 0.5 * l_csce + 0.5 * l_metric) / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix({'L': f"{loss.item()*accumulation_steps:.2f}", 'Best': f"{best_val_map:.3f}"})

        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            metrics = evaluate_retrieval(model, val_loader, device)
            if metrics['mAP'] > best_val_map:
                best_val_map = metrics['mAP']
                torch.save(model.state_dict(), f'weights/teacher_upsize_fold_{f_idx}.pth')
                
    return best_val_map

def main():
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('weights', exist_ok=True)
    
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)
    
    # Train thử Fold 0
    df_ref = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_train_cons = df_all[(df_all['is_ref']==0) & (df_all['fold']!=0) & (df_all['fold']!=4)]
    df_val_query = df_all[(df_all['is_ref']==0) & (df_all['fold']==0)]
    df_train = pd.concat([df_train_cons, df_ref]).reset_index(drop=True)
    
    train_one_fold(0, num_classes, df_train, df_val_query, df_ref, device)

if __name__ == "__main__":
    main()