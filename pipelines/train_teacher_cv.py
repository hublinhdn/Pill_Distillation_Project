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
    # Cấu hình an toàn cho 10GB VRAM, size 224
    pooling_type='mpn-cov'
    n_classes = 6 if pooling_type == 'mpn-cov' else 8
    n_samples = 4
    
    sampler = BalancedBatchSampler(df_train['label_idx'].values, n_classes=n_classes, n_samples=n_samples)
    train_loader = DataLoader(PillDataset(df_train, get_transforms(is_train=True, size=224)), 
                              batch_sampler=sampler, num_workers=2, pin_memory=True)
    
    val_df = pd.concat([df_val, df_ref]).reset_index(drop=True)
    val_loader = DataLoader(PillDataset(val_df, get_transforms(is_train=False, size=224)), 
                            batch_size=16, num_workers=2)

    model = PillTeacher(num_classes=num_classes, backbone_type='resnet50', pooling_type=pooling_type).to(device)
    
    # optimizer = torch.optim.AdamW([
    #     {'params': model.features.parameters(), 'lr': 2e-5, 'name': 'backbone'},
    #     {'params': model.reduce_conv.parameters(), 'lr': 2e-4, 'name': 'head'},
    #     {'params': model.fc_bilinear.parameters(), 'lr': 2e-4, 'name': 'head'},
    #     {'params': model.fc_ce.parameters(), 'lr': 2e-4, 'name': 'head'},
    #     {'params': [model.proxy_cos], 'lr': 2e-4, 'name': 'head'}
    # ], weight_decay=1e-2)

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 2e-5, 'name': 'backbone'},
        {'params': model.reduce_conv.parameters(), 'lr': 2e-4, 'name': 'head'},
        {'params': model.fc_projection.parameters(), 'lr': 2e-4, 'name': 'head'}, # Đổi từ fc_bilinear
        {'params': model.fc_ce.parameters(), 'lr': 2e-4, 'name': 'head'},
        {'params': [model.proxy_cos], 'lr': 2e-4, 'name': 'head'}
    ], weight_decay=1e-2)
    
    epochs = 100
    warmup_epochs = 5
    scheduler = CosineAnnealingLR(optimizer, T_max=(epochs-warmup_epochs), eta_min=1e-6)

    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_csce = nn.CrossEntropyLoss()
    criterion_triplet = losses.TripletMarginLoss(margin=0.2)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=1, neg_margin=0)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    best_val_map = 0
    patience = 10
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | Ep {epoch}/{epochs}")
        for imgs, labels, _ in pbar:
            # FIX LỖI "Short" bằng cách thêm .long()
            imgs, labels = imgs.to(device), labels.to(device).long()
            
            logits_sce, logits_csce, norm_emb = model(imgs, labels)
            
            l_sce = criterion_sce(logits_sce, labels)
            l_eta = criterion_csce(logits_csce, labels)
            loss = l_sce + l_eta
            
            if epoch > 5:
                indices = miner(norm_emb, labels)
                loss += 0.5 * criterion_triplet(norm_emb, labels, indices)
                loss += 0.5 * criterion_contrastive(norm_emb, labels, indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'Best': f"{best_val_map:.3f}"})
        
        if epoch > warmup_epochs: scheduler.step()

        if (epoch % 5 == 0) or (epoch > 80):
            torch.cuda.empty_cache()
            gc.collect()
            metrics = evaluate_retrieval(model, val_loader, device)
            curr_map = metrics['mAP']
            print(f"📊 Epoch {epoch} mAP: {curr_map:.4f}")

            if curr_map > best_val_map:
                best_val_map, counter = curr_map, 0
                torch.save(model.state_dict(), f'weights/teacher_fold_{f_idx}_best.pth')
            else:
                counter += 1
                if counter >= patience: break
                    
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