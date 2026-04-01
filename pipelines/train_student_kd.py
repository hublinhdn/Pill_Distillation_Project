import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import os
import sys
import traceback
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import losses, miners
# ==========================================
# 🔒 CHÈN ĐOẠN NÀY ĐỂ CỐ ĐỊNH RANDOM SEED
# ==========================================
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from models.model_category_config import super_large_backbones, large_backbones, medium_backbones, small_backbones, vit_backbones, swin_backbones
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def train_kd_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    # ==========================================
    # ⚙️ 1. CẤU HÌNH THÔNG SỐ ĐỘNG
    # ==========================================
    L_SCE, L_CSCE, L_TRIPLET, L_CONTRASTIVE = 1.0, 0.2, 1.0, 1.0
    ALPHA_KD = args.alpha 
    
    student_name = args.student.lower()
    teacher_name = args.teacher.lower()
    
    # is_pure_vit = any(x in student_name for x in pure_transformer_backbones)
    is_swin_backbone = any(x in student_name for x in swin_backbones)
    is_vit_backbone = any(x in student_name for x in vit_backbones)
    is_pure_vit = is_swin_backbone or is_vit_backbone

    is_fragile = any(x in student_name for x in ['mobilenet', 'ghostnet', 'shufflenet', 'squeezenet'])
    is_massive = any(x in student_name for x in ['large', 'xlarge']) and not is_pure_vit
    
    if is_pure_vit:
        L_CSCE, L_TRIPLET, L_CONTRASTIVE = 1.0, 0.0, 0.0 
    
    if any(x in teacher_name for x in super_large_backbones):
        n_classes_batch, n_samples = 4, 2  # Physical Batch = 8 ảnh (Thay vì 2, 2 như cũ)
        accumulation_steps = 16            # Effective Batch = 128
        lr_backbone, lr_head = 2e-5, 2e-4  # Tăng LR lên một chút vì Batch đã to hơn
    elif any(x in teacher_name for x in large_backbones):
        n_classes_batch, n_samples = 8, 2  # Physical Batch = 16 ảnh
        accumulation_steps = 8             # Effective Batch = 128 (Giữ nguyên)
        lr_backbone, lr_head = 3e-5, 3e-4  # Tăng LR lên một chút do Batch to ra
    elif any(x in teacher_name for x in medium_backbones):
        n_classes_batch, n_samples = 8, 2  # Physical Batch = 16 ảnh
        accumulation_steps = 8             # Effective Batch = 128
        lr_backbone, lr_head = 3e-5, 3e-4
    elif any(x in teacher_name for x in small_backbones):
        n_classes_batch, n_samples = 16, 2 # Physical Batch = 32 ảnh
        accumulation_steps = 4             # Effective Batch = 128
        lr_backbone, lr_head = 5e-5, 5e-4  
    else: 
        n_classes_batch, n_samples = 16, 2 # Physical Batch = 32 ảnh
        accumulation_steps = 4             # Effective Batch = 128
        lr_backbone, lr_head = 5e-5, 5e-4  



    # 🛡️ PHÂN BỔ LEARNING RATE THEO MẠNG STUDENT (Vì ta đang train Student)
    if any(x in student_name for x in super_large_backbones + large_backbones):
        lr_backbone, lr_head = 2e-5, 2e-4  
    elif any(x in student_name for x in medium_backbones):
        lr_backbone, lr_head = 3e-5, 3e-4
    else: 
        lr_backbone, lr_head = 5e-5, 5e-4  

    # IMAGE SIZE: Bắt buộc tuân theo Teacher
    img_size = 392 if 'patch14' in teacher_name else 384
    resize_scale = int(img_size * 1.15) 
    
    print(f"⚙️ KD Config: Thầy [{args.teacher}] -> Trò [{args.student}] | Size={img_size}")
    print(f"🧪 Method: {args.kd_type.upper()} | Alpha={ALPHA_KD} | Temp={args.temperature}")
    print(f"📊 Tài nguyên an toàn: Batch={n_classes_batch*n_samples} | Accum={accumulation_steps}")

    # ==========================================
    # 🖼️ 2. DATA AUGMENTATION
    # ==========================================
    train_transform = transforms.Compose([
        transforms.Resize((resize_scale, resize_scale)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
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
        batch_size=n_classes_batch * n_samples, shuffle=False, num_workers=4, pin_memory=True
    )

    # ==========================================
    # 🧠 3. KHỞI TẠO MÔ HÌNH (THẦY & TRÒ)
    # ==========================================
    teacher = PillRetrievalModel(num_classes=num_classes, backbone_type=args.teacher, pooling_type='gem').to(device)
    teacher.load_state_dict(torch.load(args.teacher_weight, map_location=device))
    teacher.eval() 
    for param in teacher.parameters():
        param.requires_grad = False

    student = PillRetrievalModel(num_classes=num_classes, backbone_type=args.student, pooling_type='gem').to(device)

    # ==========================================
    # 🎯 4. LOSS & OPTIMIZER CHUẨN MỰC
    # ==========================================
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
    
    criterion_mse = nn.MSELoss()
    criterion_cosine = nn.CosineEmbeddingLoss()
    criterion_kld = nn.KLDivLoss(reduction='batchmean')

    backbone_params = [p for n, p in student.named_parameters() if 'features' in n]
    head_params = [p for n, p in student.named_parameters() if 'features' not in n]

    wd_value = 0.1 if any(x in student_name for x in ['large', 'xlarge']) else 5e-2

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone}, 
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=wd_value)
    
    TOTAL_EPOCHS = args.epochs 
    WARMUP_EPOCHS = 5
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(TOTAL_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    scaler = torch.amp.GradScaler('cuda')
    best_val_map, r1 = 0.0, 0.0

    # ==========================================
    # 🚀 5. VÒNG LẶP HUẤN LUYỆN CHƯNG CẤT
    # ==========================================
    for epoch in range(1, TOTAL_EPOCHS + 1):
        if is_fragile or is_massive:
            freeze_epochs = 2 if is_fragile else 4 
            if epoch <= freeze_epochs:
                for param in student.features.parameters():
                    param.requires_grad = False
            elif epoch == freeze_epochs + 1:
                for param in student.features.parameters():
                    param.requires_grad = True

        student.train()
        pbar = tqdm(train_loader, desc=f"KD Fold {f_idx} | Ep {epoch} | Best {best_val_map:.4f} | R1 {r1:.4f}")
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            sub_labels = sub_labels.to(device).long()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    t_logits_sce, _, t_norm_emb = teacher(imgs, labels=sub_labels)
                
                s_logits_sce, s_logits_csce, s_norm_emb = student(imgs, labels=sub_labels)

                loss_sce = criterion_sce(s_logits_sce, sub_labels)
                loss_csce = criterion_sce(s_logits_csce, sub_labels)
                
                if L_TRIPLET > 0 or L_CONTRASTIVE > 0:
                    hard_pairs = miner(s_norm_emb, sub_labels)
                    loss_triplet = criterion_triplet(s_norm_emb, sub_labels, hard_pairs)
                    loss_contrastive = criterion_contrastive(s_norm_emb, sub_labels)
                else:
                    loss_triplet = torch.tensor(0.0, device=device)
                    loss_contrastive = torch.tensor(0.0, device=device)

                student_loss = L_SCE * loss_sce + L_CSCE * loss_csce + L_TRIPLET * loss_triplet + L_CONTRASTIVE * loss_contrastive

                # DISTILLATION LOSS (TRUYỀN CÔNG)
                kd_loss = torch.tensor(0.0, device=device)
                
                if args.kd_type in ['mse', 'hybrid']:
                    kd_loss += criterion_mse(s_norm_emb, t_norm_emb.detach())
                    
                if args.kd_type in ['cosine', 'hybrid']:
                    target_cos = torch.ones(s_norm_emb.size(0), device=device)
                    kd_loss += criterion_cosine(s_norm_emb, t_norm_emb.detach(), target_cos)
                    
                if args.kd_type in ['kl', 'hybrid']:
                    T = args.temperature
                    # 🛡️ ÉP KIỂU FP32 ĐỂ TRÁNH LỖI NaN TRONG AUTOCAST
                    s_log_probs = F.log_softmax(s_logits_sce.float() / T, dim=1)
                    t_probs = F.softmax(t_logits_sce.float() / T, dim=1).detach()
                    kl_val = criterion_kld(s_log_probs, t_probs) * (T * T)
                    kd_loss += kl_val if args.kd_type == 'kl' else (kl_val * 0.1) 

                loss = (student_loss + ALPHA_KD * kd_loss) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                clip_val = 1.0 if (is_fragile or is_pure_vit) else 5.0
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=clip_val)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({'S_Loss': f"{student_loss.item():.2f}", 'KD': f"{kd_loss.item():.4f}"})

        scheduler.step()

        if (epoch % 5 == 0) or (epoch > TOTAL_EPOCHS - 10):
            val_metrics = evaluate_retrieval(student, val_loader, device, flag_both_side=True)
            val_map = val_metrics['mAP']
            val_r1 = val_metrics['Rank-1']
            print(f"📊 Epoch {epoch} mAP: {val_map:.4f} Rank-1: {val_r1}")
            
            if val_map > best_val_map:
                best_val_map = val_map
                r1 = val_r1
                os.makedirs('weights/kd_models', exist_ok=True)
                torch.save(student.state_dict(), f"weights/kd_models/best_{teacher_name}_kd_{student_name}_kd_type{args.kd_type}_fold{f_idx}.pth")
                
    return best_val_map, r1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, required=True)
    parser.add_argument('--teacher_weight', type=str, required=True)
    parser.add_argument('--student', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--kd_type', type=str, default='cosine', choices=['mse', 'cosine', 'kl', 'hybrid'])
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--summary_file', type=str, default='KD_Summary.txt')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    print(f"🌱 Setting random seed to: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # Nếu dùng multi-GPU
        # Đảm bảo các phép toán của cuDNN chạy ổn định và giống hệt nhau ở mỗi lần
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # ==========================================

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        device = torch.device('mps')
    else: 
        device = torch.device('cpu')

    os.makedirs('reports_kd', exist_ok=True)
    report_file = args.summary_file
    report_file_path = os.path.join('reports_kd', report_file)
    # Kiểm tra xem tạ có tồn tại không trước khi chạy
    if not os.path.exists(report_file_path):
        with open(report_file_path, "w") as f:
            f.write(f"The first time to run kD ==> create file\n")
            f.write("="*50 + "\n")
            f.write(f"{'Teacher'.ljust(35)} --> {'student'.ljust(35)} | {'alpha'.ljust(4)} | {'kd_type'.ljust(6)} | {'mAP'.ljust(6)} | {'Rank-1'.ljust(6)}\n")
            f.write("-"*50 + "\n")
    
    try:
        df_all = load_epill_full_data()
        total_sub_classes = df_all['sub_label_idx'].nunique()

        fold = 0
        df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
        df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
        
        cond_train = (df_all['fold'] != fold) & (df_all['is_ref'] == 0)
        train_labels = df_all[cond_train]['label_idx'].unique()
        cond_ref_train = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
        df_train = df_all[cond_train | cond_ref_train].reset_index(drop=True)
        
        map, r1 = train_kd_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)


        with open(report_file_path, "a") as f:
            f.write(f"{args.teacher.ljust(35)} --> {args.student.ljust(35)} | {args.alpha:.4f} | {args.kd_type.ljust(6)} | {map:.4f} | {r1:.4f}\n")
            
        
    except Exception as e:
        # 🛡️ GHI LẠI TOÀN BỘ LỖI VÀO FILE NẾU BỊ CRASH
        
        with open('reports_kd/KD_Crash_Traceback.txt', 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"CRASH LOG: Teacher [{args.teacher}] -> Student [{args.student}]\n")
            f.write(traceback.format_exc())
        raise e # Vẫn quăng lỗi để script tổng biết mà đánh dấu FAILED

if __name__ == '__main__':
    main()