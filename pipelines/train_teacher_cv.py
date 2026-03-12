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
from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    # --- 1. CẤU HÌNH HỆ SỐ LOSS CỰC MẠNH ---
    L_SCE = 1.0        # Trọng số Softmax Cross Entropy
    L_CSCE = 1.0       # TĂNG LÊN 1.0: Để ArcFace làm chủ đạo phân cụm
    L_TRIPLET = 1.0    # Trọng số Triplet Loss
    L_CONTRASTIVE = 1.0 # Trọng số Contrastive Loss
    
    # --- 2. CHIẾN THUẬT BATCH CHUYÊN BIỆT CHO HÀNG NGÀN SUB-CLASS ---
    n_classes, n_samples = 16, 2 
    accumulation_steps = 4 
    
    # --- 3. AUGMENTATION (TÍCH HỢP RANDOM ERASING) ---
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)) # Xóa ngẫu nhiên chống học vẹt
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 4. DATALOADER (DÙNG SUB_LABEL_IDX) ---
    train_loader = DataLoader(
        PillDataset(df_train, train_transform), 
        batch_sampler=BalancedBatchSampler(df_train['sub_label_idx'].values, n_classes, n_samples),
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        PillDataset(pd.concat([df_val, df_ref]).reset_index(drop=True), val_transform), 
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- 5. KHỞI TẠO MÔ HÌNH ---
    # Tổng số class phải nhân 2 vì mỗi thuốc có 2 mặt (sub_classes)
    model = PillTeacher(num_classes=num_classes, backbone_type=args.backbone).to(device)

    # --- 6. HÀM LOSS VÀ MINER ---
    # Label smoothing 0.1 giúp mô hình không bị "tự mãn" và overfit
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")

    # --- 7. OPTIMIZER & ONECYCLE LR SCHEDULER ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-2)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4, 
        epochs=100, 
        steps_per_epoch=len(train_loader),
        pct_start=0.1 # Dành 10 epoch đầu để Warmup
    )

    best_val_map = 0.0
    print(f"Bắt đầu huấn luyện Fold {f_idx} với {len(train_loader)} batches/epoch...")

    # --- 8. VÒNG LẶP HUẤN LUYỆN ---
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {f_idx} | Ep {epoch}")
        
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            # Quan trọng: Train bằng sub_labels (ID của từng mặt)
            sub_labels = sub_labels.to(device)

            logits_sce, logits_csce, norm_embedding = model(imgs, labels=sub_labels)
            
            # Tính các Loss
            loss_sce = criterion_sce(logits_sce, sub_labels)
            loss_csce = criterion_sce(logits_csce, sub_labels)
            
            hard_pairs = miner(norm_embedding, sub_labels)
            loss_triplet = criterion_triplet(norm_embedding, sub_labels, hard_pairs)
            loss_contrastive = criterion_contrastive(norm_embedding, sub_labels)

            # Tổng hợp Loss
            loss = (L_SCE * loss_sce + 
                    L_CSCE * loss_csce + 
                    L_TRIPLET * loss_triplet + 
                    L_CONTRASTIVE * loss_contrastive) / accumulation_steps
            
            loss.backward()

            # Gradient Accumulation & Step
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Clip gradient chống nổ bùng (Gradient Explosion)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                # CHÚ Ý: OneCycleLR phải được step() ở cuối mỗi batch cập nhật
                scheduler.step()
                
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'L': f"{loss.item()*accumulation_steps:.2f}", 'Best': f"{best_val_map:.3f}"})

        # --- ĐÁNH GIÁ (EVALUATION) MỖI 5 EPOCH ---
        if epoch % 5 == 0:
            val_metrics = evaluate_retrieval(model, val_loader, device)
            val_map = val_metrics['mAP']
            val_r1 = val_metrics['Rank-1']
            print(f"📊 Epoch {epoch} mAP: {val_map:.4f} (R1: {val_r1:.4f})")
            
            if val_map > best_val_map:
                best_val_map = val_map
                os.makedirs('weights', exist_ok=True)
                save_path = f"weights/best_teacher_fold{f_idx}_{args.backbone}.pth"
                torch.save(model.state_dict(), save_path)
                
    return best_val_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Validation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    df_all = load_epill_full_data()
    # 1. Tạo chuỗi kết hợp ID thuốc và mặt trước/sau (ví dụ: "105_1", "105_0")
    df_all['sub_label_raw'] = df_all['label_idx'].astype(str) + "_" + df_all['is_front'].astype(str)
    # 2. pd.factorize tự động đánh số lại chuỗi trên thành các số nguyên liên tục: 0, 1, 2, ..., N-1
    df_all['sub_label_idx'] = pd.factorize(df_all['sub_label_raw'])[0]
    # 3. Đếm CHÍNH XÁC tổng số sub-class đã được tạo ra
    total_sub_classes = df_all['sub_label_idx'].nunique()
    print(f"✅ Đã chuẩn hóa nhãn! Tổng số mặt thuốc (sub-classes): {total_sub_classes}")
    # ----------------------

    # num_classes = df_all['label_idx'].nunique() # Tự đếm số class gốc ngay tại đây
    

    # FIX LỖI DATA SPLIT: Danh sách tất cả các fold hợp lệ để train
    cv_folds_all = [0, 1, 2, 3] 
    
    # Chỉ định fold muốn chạy (ví dụ chạy thử fold 0)
    folds_to_run = [0] 

    results_summary = []
    
    for fold in folds_to_run:
        print(f"\n🚀 Training Fold {fold}...")
        
        # Tập Reference (Gallery) - Luôn dùng cho mọi fold
        df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
        
        # Tập Validation (Consumer)
        df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
        
        # Tập Train (Consumer) - Lấy các ảnh thuộc cv_folds_all nhưng khác fold hiện tại
        cond_cons = (df_all['fold'].isin(cv_folds_all)) & (df_all['fold'] != fold) & (df_all['is_ref'] == 0)
        
        # Đảm bảo Train set chỉ chứa các loại thuốc có xuất hiện trong gallery
        train_labels = df_all[cond_cons]['label_idx'].unique()
        cond_ref = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
        
        df_train = df_all[cond_cons | cond_ref].reset_index(drop=True)
        
        # Chạy huấn luyện
        best_map = train_one_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)
        
        results_summary.append({
            'fold': fold,
            'best_mAP': best_map
        })
        print(f"✅ Fold {fold} Finished. Best mAP: {best_map:.4f}\n")

    # Xuất Report
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/final_report_{args.backbone}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"PILL IDENTIFICATION REPORT - BACKBONE: {args.backbone}\n")
        f.write("="*50 + "\n")
        
        for res in results_summary:
            line = f"Fold {res['fold']}: Best mAP = {res['best_mAP']:.4f}\n"
            f.write(line)
            print(line, end="")

if __name__ == '__main__':
    main()