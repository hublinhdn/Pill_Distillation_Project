import torch
import pandas as pd
import os
import sys
import argparse
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.student_model import PillStudent
from utils.dataset_loader import PillDataset, get_transforms, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data
from pytorch_metric_learning import losses, miners

def train_one_fold(num_classes, df_train, device, is_sanity=False):
    # Balanced Sampler giúp Student học tốt hơn trong điều kiện ít dữ liệu
    sampler = BalancedBatchSampler(df_train['label_idx'].values, n_classes=8, n_samples=4)
    
    train_loader = DataLoader(
        PillDataset(df_train, get_transforms(is_train=True, size=224)), # Student dùng size 224
        batch_sampler=sampler, 
        num_workers=0 if is_sanity else 4
    )
    
    model = PillStudent(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2) # LR cao hơn Teacher
    
    # Baseline thường dùng cặp Loss kinh điển: CE + Triplet
    crit_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    crit_tri = losses.TripletMarginLoss(margin=0.3)
    miner = miners.MultiSimilarityMiner()

    epochs = 1 if is_sanity else 50 # Student cần nhiều epoch hơn để hội tụ
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, emb = model(imgs, labels)
            
            indices = miner(emb, labels)
            loss = crit_ce(logits, labels) + crit_tri(emb, labels, indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if is_sanity: break 
        
        if (epoch + 1) % 10 == 0 or is_sanity:
            print(f"Student Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('weights', exist_ok=True)

    full_df = load_epill_full_data()
    num_classes = full_df['label_idx'].nunique()

    df_ref = full_df[full_df['is_ref'] == 1].reset_index(drop=True)
    df_cons = full_df[full_df['is_ref'] == 0].reset_index(drop=True)

    # Chạy CV fold 0 để lấy baseline nhanh
    cv_folds = [0] if args.sanity else [0, 1, 2, 3]

    for f_idx in cv_folds:
        print(f"\n--- 🧪 TRAINING STUDENT BASELINE FOLD {f_idx} ---")
        
        df_train_cons = df_cons[(df_cons['fold'] != f_idx) & (df_cons['fold'] != 4)]
        df_train = pd.concat([df_train_cons, df_ref]).reset_index(drop=True)
        
        if args.sanity:
            df_train = df_train.head(100)
            df_val_query = df_cons[df_cons['fold'] == f_idx].head(50)
        else:
            df_val_query = df_cons[df_cons['fold'] == f_idx]

        model = train_one_fold(num_classes, df_train, device, args.sanity)
        
        eval_df = pd.concat([df_val_query, df_ref]).reset_index(drop=True)
        val_loader = DataLoader(PillDataset(eval_df, get_transforms(is_train=False, size=224)), batch_size=32)
        
        metrics = evaluate_retrieval(model, val_loader, device)
        print(f"📊 Student Baseline Fold {f_idx}: mAP = {metrics['mAP']:.4f}")
        
        torch.save(model.state_dict(), f'weights/student_baseline_fold_{f_idx}.pth')

if __name__ == "__main__":
    main()