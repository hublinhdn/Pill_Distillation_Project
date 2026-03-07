import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_loader import PillDataset, get_transforms, BalancedBatchSampler
from models.teacher_model import PillTeacher

def train_one_fold(fold_idx, df_train, num_epochs=40, batch_size=32, save_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = df_train['label_idx'].nunique()

    # Balanced Sampler giúp Triplet Loss cực mạnh
    sampler = BalancedBatchSampler(df_train['label_idx'].values, n_classes=8, n_samples=4)
    train_loader = DataLoader(
        PillDataset(df_train, transform=get_transforms(is_train=True, size=256), root_dir='data/raw/ePillID/classification_data'),
        batch_sampler=sampler, num_workers=4, pin_memory=True
    )

    model = PillTeacher(num_classes=num_classes).to(device)
    
    # Bộ 4 Loss kết hợp theo Benchmark
    criterion_arc = nn.CrossEntropyLoss()
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3, swap=True)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.MultiSimilarityMiner()

    # Optimizer với LR thấp cho ResNet Pretrained
    optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    accumulation_steps = 4 # Giả lập Batch 128
    
    print(f"🚀 Training ResNet-50 Teacher Fold {fold_idx} | Simulated Batch: 128")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        w_metric = min(1.0, epoch / 10.0)
        
        optimizer.zero_grad()
        for i, (imgs, labels, _) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            (logits_arc, logits_ce), features = model(imgs, labels)
            
            loss_arc = criterion_arc(logits_arc, labels)
            loss_ce = criterion_ce(logits_ce, labels)
            
            feat_norm = F.normalize(features, p=2, dim=1)
            hard_pairs = miner(feat_norm, labels)
            loss_tri = criterion_triplet(feat_norm, labels, hard_pairs)
            loss_con = criterion_contrastive(feat_norm, labels, hard_pairs)
            
            # Tổ hợp Loss Benchmark
            loss = (loss_arc + loss_ce) + w_metric * (loss_tri + loss_con)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
        
        scheduler.step()
        print(f"Fold {fold_idx} - Ep [{epoch+1}/{num_epochs}] Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('weights', save_name))
    return model