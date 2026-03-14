import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import losses, miners

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillRetrievalModel
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def train_kd_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    L_SCE, L_CSCE, L_TRIPLET, L_CONTRASTIVE = 1.0, 0.2, 1.0, 1.0
    ALPHA_KD = args.alpha 
    
    img_size = 384 if 'convnext' in args.teacher else 448
    n_classes_batch, n_samples = 4, 2  
    accumulation_steps = 16            
    
    print(f"⚙️ KD Config: {args.teacher} -> {args.student} | Size={img_size}")
    print(f"🧪 KD Method: {args.kd_type.upper()} | Alpha={ALPHA_KD} | Temp={args.temperature}")

    resize_scale = int(img_size * 1.15) 
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
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- KHỞI TẠO MÔ HÌNH ---
    teacher = PillRetrievalModel(num_classes=num_classes, backbone_type=args.teacher, pooling_type='gem').to(device)
    teacher.load_state_dict(torch.load(args.teacher_weight, map_location=device))
    teacher.eval() 
    for param in teacher.parameters():
        param.requires_grad = False

    student = PillRetrievalModel(num_classes=num_classes, backbone_type=args.student, pooling_type='gem').to(device)

    # --- KHỞI TẠO LOSSES ---
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
    
    # KD Losses
    criterion_mse = nn.MSELoss()
    criterion_cosine = nn.CosineEmbeddingLoss()
    criterion_kld = nn.KLDivLoss(reduction='batchmean')

    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_val_map = 0.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"KD Fold {f_idx} | Ep {epoch}")
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            sub_labels = sub_labels.to(device).long()

            with torch.amp.autocast('cuda'):
                # Forward Sư phụ
                with torch.no_grad():
                    t_logits_sce, _, t_norm_emb = teacher(imgs, labels=sub_labels)
                
                # Forward Đệ tử
                s_logits_sce, s_logits_csce, s_norm_emb = student(imgs, labels=sub_labels)

                # --- 1. STUDENT LOSS TỰ THÂN ---
                loss_sce = criterion_sce(s_logits_sce, sub_labels)
                loss_csce = criterion_sce(s_logits_csce, sub_labels)
                hard_pairs = miner(s_norm_emb, sub_labels)
                loss_triplet = criterion_triplet(s_norm_emb, sub_labels, hard_pairs)
                loss_contrastive = criterion_contrastive(s_norm_emb, sub_labels)
                student_loss = L_SCE * loss_sce + L_CSCE * loss_csce + L_TRIPLET * loss_triplet + L_CONTRASTIVE * loss_contrastive

                # --- 2. DISTILLATION LOSS (TRUYỀN CÔNG) ---
                kd_loss = 0.0
                
                if args.kd_type in ['mse', 'hybrid']:
                    kd_loss += criterion_mse(s_norm_emb, t_norm_emb)
                    
                if args.kd_type in ['cosine', 'hybrid']:
                    target_cos = torch.ones(s_norm_emb.size(0)).to(device)
                    kd_loss += criterion_cosine(s_norm_emb, t_norm_emb, target_cos)
                    
                if args.kd_type in ['kl', 'hybrid']:
                    T = args.temperature
                    s_log_probs = F.log_softmax(s_logits_sce / T, dim=1)
                    t_probs = F.softmax(t_logits_sce / T, dim=1)
                    kl_val = criterion_kld(s_log_probs, t_probs) * (T * T)
                    # Giảm scale của KL để không lấn át Cosine/MSE nếu chạy Hybrid
                    kd_loss += kl_val if args.kd_type == 'kl' else (kl_val * 0.1) 

                # TỔNG HỢP LOSS
                loss = (student_loss + ALPHA_KD * kd_loss) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({'S_Loss': f"{student_loss.item():.2f}", 'KD': f"{kd_loss.item():.4f}"})

        scheduler.step()

        if (epoch % 5 == 0) or (epoch > args.epochs - 10):
            val_metrics = evaluate_retrieval(student, val_loader, device, flag_both_side=True)
            val_map = val_metrics['mAP']
            print(f"📊 Epoch {epoch} mAP: {val_map:.4f}")
            
            if val_map > best_val_map:
                best_val_map = val_map
                os.makedirs('weights', exist_ok=True)
                torch.save(student.state_dict(), f"weights/best_kd_{args.student}_{args.kd_type}_fold{f_idx}.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, default='convnext_base')
    parser.add_argument('--teacher_weight', type=str, required=True)
    parser.add_argument('--student', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # --- CÁC THAM SỐ ĐỂ BENCHMARK ---
    parser.add_argument('--kd_type', type=str, default='cosine', choices=['mse', 'cosine', 'kl', 'hybrid'], help='Phương pháp truyền công')
    parser.add_argument('--alpha', type=float, default=10.0, help='Trọng số của KD Loss (Thường MSE/Cosine cần 10-50, KL cần 1-5)')
    parser.add_argument('--temperature', type=float, default=4.0, help='Nhiệt độ T cho KL-Divergence')
    
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
    
    train_kd_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)

if __name__ == '__main__':
    main()