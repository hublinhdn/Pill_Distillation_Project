import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
import argparse
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.teacher_model import PillTeacher
from models.student_model import PillStudent
from utils.dataset_loader import PillDataset, get_transforms, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from data_utils import load_epill_full_data

def train_kd_fold(f_idx, df_train, df_ref, num_classes, teacher_path, device, args):
    # 1. Khởi tạo Teacher (Freeze)
    teacher = PillTeacher(num_classes=num_classes).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    
    # 2. Khởi tạo Student
    student = PillStudent(num_classes=num_classes).to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
    sampler = BalancedBatchSampler(df_train['label_idx'].values, n_classes=8, n_samples=4)
    train_loader = DataLoader(PillDataset(df_train, get_transforms(True, size=224)), batch_sampler=sampler)

    print(f"👩‍🏫 Distilling Teacher Fold {f_idx} to Student...")

    for epoch in range(1 if args.sanity else 40):
        student.train()
        total_loss = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Teacher inference
            with torch.no_grad():
                (t_logits_ce, t_logits_cos), t_emb = teacher(imgs, labels)
            
            # Student inference
            s_logits, s_emb = student(imgs, labels)
            
            # LOSS 1: Distillation (MSE ép Student bắt chước không gian của Teacher)
            loss_distill = F.mse_loss(s_emb, t_emb)
            
            # LOSS 2: Student Classification
            loss_ce = F.cross_entropy(s_logits, labels)
            
            # Tổng hợp (Trọng số 0.7 cho Teacher và 0.3 cho Student là tỉ lệ vàng)
            loss = 7.0 * loss_distill + 0.3 * loss_ce
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if args.sanity: break
            
        print(f"KD Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
    
    return student

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--teacher_fold', type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_df = load_epill_full_data()
    num_classes = full_df['label_idx'].nunique()
    
    df_ref = full_df[full_df['is_ref'] == 1].reset_index(drop=True)
    df_cons = full_df[full_df['is_ref'] == 0].reset_index(drop=True)
    
    f_idx = args.teacher_fold
    teacher_path = f'weights/teacher_fold_{f_idx}.pth'
    
    if not os.path.exists(teacher_path):
        print(f"❌ Không tìm thấy file trọng số Teacher: {teacher_path}")
        return

    # Chuẩn bị Data
    df_train_cons = df_cons[(df_cons['fold'] != f_idx) & (df_cons['fold'] != 4)]
    df_train = pd.concat([df_train_cons, df_ref]).reset_index(drop=True)
    df_val_query = df_cons[df_cons['fold'] == f_idx]

    # Train KD
    student = train_kd_fold(f_idx, df_train, df_ref, num_classes, teacher_path, device, args)
    
    # Eval
    eval_df = pd.concat([df_val_query, df_ref]).reset_index(drop=True)
    val_loader = DataLoader(PillDataset(eval_df, get_transforms(False, size=224)), batch_size=32)
    metrics = evaluate_retrieval(student, val_loader, device)
    
    print(f"✨ Student KD Fold {f_idx} Results: mAP = {metrics['mAP']:.4f}")
    torch.save(student.state_dict(), f'weights/student_kd_fold_{f_idx}.pth')

if __name__ == "__main__":
    main()