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
import torchvision.transforms.functional as TF
import traceback

# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from models.model_category_config import super_large_backbones, large_backbones, medium_backbones, small_backbones, vit_backbones, swin_backbones
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data
folder_weight = 'weights/phase3'

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    # L_SCE, L_CSCE, L_TRIPLET, L_CONTRASTIVE = 1.0, 0.2, 1.0, 1.0
    USE_SHAPE_LOSS, L_SHAPE = False, 1.0 

    L_SCE = args.w_sce
    L_CSCE = args.w_csce
    L_TRIPLET = args.w_triplet
    L_CONTRASTIVE = args.w_cont

    TOTAL_EPOCHS = args.epochs 
    WARMUP_EPOCHS = 5
    
    backbone_name = args.backbone.lower()
    is_swin_backbone = any(x in backbone_name for x in swin_backbones)
    is_vit_backbone = any(x in backbone_name for x in vit_backbones)
    is_pure_vit = is_swin_backbone or is_vit_backbone
    is_fragile = any(x in backbone_name for x in small_backbones)
    is_massive = any(x in backbone_name for x in super_large_backbones)

    # is_fragile = any(x in backbone_name for x in ['mobilenet', 'ghostnet'])
    # is_massive = any(x in backbone_name for x in ['large', 'xlarge']) and not is_pure_vit
    if is_pure_vit:
        L_CSCE, L_TRIPLET, L_CONTRASTIVE = 0.0, 0.0, 0.0 # Tắt cả Contrastive
    
    # PHÂN BỔ TÀI NGUYÊN ĐỘNG
    if any(x in backbone_name for x in super_large_backbones):
        # n_classes_batch, n_samples = 2, 2  # Physical Batch = 4 ảnh
        # accumulation_steps = 32            # Effective Batch = 128
        # lr_backbone, lr_head = 1e-5, 1e-4  
        # 🚀 ĐÃ BẬT CHECKPOINTING -> NÂNG BATCH TỪ 4 LÊN 8
        n_classes_batch, n_samples = 4, 2  # Physical Batch = 8 ảnh (Thay vì 2, 2 như cũ)
        accumulation_steps = 16            # Effective Batch = 128
        lr_backbone, lr_head = 2e-5, 2e-4  # Tăng LR lên một chút vì Batch đã to hơn
    elif any(x in backbone_name for x in large_backbones):
        # n_classes_batch, n_samples = 4, 2  # Physical Batch = 8 ảnh
        # accumulation_steps = 16            # Effective Batch = 128
        # lr_backbone, lr_head = 2e-5, 2e-4
        # 🚀 ĐÃ BẬT CHECKPOINTING -> NÂNG BATCH TỪ 8 LÊN 16
        n_classes_batch, n_samples = 8, 2  # Physical Batch = 16 ảnh
        accumulation_steps = 8             # Effective Batch = 128 (Giữ nguyên)
        lr_backbone, lr_head = 3e-5, 3e-4  # Tăng LR lên một chút do Batch to ra
    elif any(x in backbone_name for x in medium_backbones):
        n_classes_batch, n_samples = 8, 2  # Physical Batch = 16 ảnh
        accumulation_steps = 8             # Effective Batch = 128
        lr_backbone, lr_head = 3e-5, 3e-4
    elif any(x in backbone_name for x in small_backbones):
        n_classes_batch, n_samples = 16, 2 # Physical Batch = 32 ảnh
        accumulation_steps = 4             # Effective Batch = 128
        lr_backbone, lr_head = 5e-5, 5e-4  
    else: 
        raise Exception(f'Please assign {backbone_name} to category (Support large - Lager - Medium - Small) to continue...')
    
    if is_pure_vit:
        lr_backbone = lr_backbone * 0.3  # Ép LR backbone xuống (ví dụ 2e-5 -> 6e-6)
        print(f"📉 Đã tự động giảm LR Backbone cho Pure Transformer xuống: {lr_backbone}")

    # -----------------------------------------------------
    # 🛡️ 2. TỰ ĐỘNG CHỈNH SIZE CHO DINOv2 / ViT (Chống lỗi Patch)
    # -----------------------------------------------------
    img_size = 384
    if 'patch14' in backbone_name:
        img_size = 392 # 392 chia hết cho 14 (14 * 28)
        print(f"⚠️ Phát hiện ViT Patch14: Tự động nâng Image Size lên {img_size}x{img_size}")
    elif 'patch16' in backbone_name:
        img_size = 384 # 384 chia hết cho 16, giữ nguyên
    resize_scale = int(img_size * 1.15) 
    
    print(f"⚙️ Config {args.backbone}: Size={img_size}x{img_size} | Batch={n_classes_batch*n_samples} | Accum={accumulation_steps}")
    
    # DUAL TRANSFORM
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

    model = PillRetrievalModel(num_classes=num_classes, backbone_type=args.backbone, pooling_type=args.pooling).to(device)

    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
    criterion_shape = nn.MSELoss()

    backbone_params = [p for n, p in model.named_parameters() if 'features' in n]
    head_params = [p for n, p in model.named_parameters() if 'features' not in n]

    # TRỪNG PHẠT TẠ ĐỘNG (Dynamic Weight Decay) ĐỂ CHỐNG OVERFITTING CHO MẠNG LỚN
    wd_value = 0.1 if any(x in backbone_name for x in ['large', 'xlarge']) else 5e-2

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone}, 
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=wd_value)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(TOTAL_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_map, r1 = 0.0, 0.0
    best_metrics = {
        'Rank-1': 0.0,
        'mAP': 0.0,
        'mAP(Cons)': 0.0,
        'mAP(Ref)': 0.0,
        'mAP(Delta)': 0.0,
        'Rank-1(Ref)': 0.0
    }


    for epoch in range(1, TOTAL_EPOCHS + 1):
        # ==========================================
        # 🛡️ HEAD WARM-UP (BẢO VỆ MẠNG YẾU & MẠNG QUÁ TO)
        # ==========================================
        if is_fragile or is_massive:
            freeze_epochs = 2 if is_fragile else 4 
            if epoch <= freeze_epochs:
                print(f"🔒 Epoch {epoch}: ĐÓNG BĂNG Backbone, chỉ train Head!")
                for param in model.features.parameters():
                    param.requires_grad = False
            elif epoch == freeze_epochs + 1:
                print(f"🔓 Epoch {epoch}: MỞ KHÓA Backbone, train toàn bộ mạng!")
                for param in model.features.parameters():
                    param.requires_grad = True

        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"{args.backbone} | Fold {f_idx} | Ep {epoch}")
        
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            sub_labels = sub_labels.to(device).long()
            
            imgs_gray = TF.rgb_to_grayscale(imgs, num_output_channels=3)

            with torch.amp.autocast('cuda'):
                logits_sce, logits_csce, norm_embedding_rgb = model(imgs, labels=sub_labels)
                loss_sce = criterion_sce(logits_sce, sub_labels)
                loss_csce = criterion_sce(logits_csce, sub_labels)
                # hard_pairs = miner(norm_embedding_rgb, sub_labels)
                # loss_triplet = criterion_triplet(norm_embedding_rgb, sub_labels, hard_pairs)
                # loss_contrastive = criterion_contrastive(norm_embedding_rgb, sub_labels)

                if L_TRIPLET > 0 or L_CONTRASTIVE > 0:
                    hard_pairs = miner(norm_embedding_rgb, sub_labels)
                    loss_triplet = criterion_triplet(norm_embedding_rgb, sub_labels, hard_pairs)
                    loss_contrastive = criterion_contrastive(norm_embedding_rgb, sub_labels)
                else:
                    loss_triplet = torch.tensor(0.0, device=device)
                    loss_contrastive = torch.tensor(0.0, device=device)

                # ==================================================
                # 🛡️ SHAPE-CONSISTENCY LOSS (AN TOÀN VRAM Tuyệt Đối)
                # ==================================================
                if USE_SHAPE_LOSS:
                    with torch.no_grad(): # CHẶN TÍNH TOÁN GRADIENT NHÁNH XÁM
                        _, _, norm_embedding_gray = model(imgs_gray, labels=sub_labels) 
                    # BẮT BUỘC dùng .detach()
                    loss_shape = criterion_shape(norm_embedding_rgb, norm_embedding_gray.detach())
                else:
                    loss_shape = 0.0
                
                loss = (L_SCE * loss_sce + L_CSCE * loss_csce + L_TRIPLET * loss_triplet + L_CONTRASTIVE * loss_contrastive + L_SHAPE * loss_shape) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                clip_val = 1.0 if (is_fragile or is_pure_vit) else 5.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            current_avg_loss = total_loss / (i + 1)

            pbar.set_postfix({
                'Avg_L': f"{current_avg_loss:.4f}", 
                'mAp': f"{best_val_map:.4f}", 
                'R1': f"{r1:.4f}"
            })

        scheduler.step()

        if (epoch % 5 == 0) or (epoch > TOTAL_EPOCHS - 10):
            val_metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=True, flag_eval_delta=True)
            val_map = val_metrics['mAP']
            val_r1 = val_metrics['Rank-1']
            print(f"📊 Epoch {epoch} mAP: {val_map:.4f} | (Rank-1: {val_r1:.4f}) | Extra: {val_metrics}")
            
            if val_map > best_val_map:
                best_val_map = val_map
                r1 = val_r1
                best_metrics = val_metrics
                os.makedirs(folder_weight, exist_ok=True)
                torch.save(model.state_dict(), f"{folder_weight}/best_{args.backbone}_{args.pooling}_fold{f_idx}_{args.w_sce}_{args.w_csce}_{args.w_triplet}_{args.w_cont}.pth")
                
    return best_val_map, r1, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='convnext_base,resnet50,mobilenet_v3_large')
    parser.add_argument('--pooling', type=str, default='gem', choices=['gem', 'mpncov'])
    parser.add_argument('--pipeline_name', type=str, default='baseline')

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--w_sce', type=float, default=1.0)
    parser.add_argument('--w_csce', type=float, default=0.2)
    parser.add_argument('--w_triplet', type=float, default=1.0)
    parser.add_argument('--w_cont', type=float, default=1.0)

    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        device = torch.device('mps')
    else: 
        device = torch.device('cpu')
    
    df_all = load_epill_full_data()
    total_sub_classes = df_all['sub_label_idx'].nunique()

    fold = 0
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    cond_train = (df_all['fold'] != fold) & (df_all['is_ref'] == 0)
    train_labels = df_all[cond_train]['label_idx'].unique()
    cond_ref_train = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
    df_train = df_all[cond_train | cond_ref_train].reset_index(drop=True)
    
    backbone_list = [b.strip() for b in args.backbone.split(',')]
    print(f"🚀 BẮT ĐẦU CHIẾN DỊCH VỚI: {backbone_list}")

    L_SCE = args.w_sce
    L_CSCE = args.w_csce
    L_TRIPLET = args.w_triplet
    L_CONTRASTIVE = args.w_cont
    TOTAL_EPOCHS = args.epochs 
    
    results_summary = {}
    pooling = args.pooling
    pipeline_name = args.pipeline_name

    # ==========================================
    # 📝 TẠO FILE LOG VÀ GHI HEADER TRƯỚC
    # ==========================================
    os.makedirs("reports", exist_ok=True)
    log_file_path = f"reports/{pipeline_name}_batch_experiment_{pooling}_summary.txt"
    log_error_file_path = f"reports/{pipeline_name}_batch_experiment_{pooling}_error.txt"

    # CHỈ tạo mới và ghi Header nếu file CHƯA CÓ.
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write(f"Batch Experiment {pipeline_name} Results for pooling: {pooling}\n")
            f.write("="*50 + "\n")
            f.write(f"{'Backbone'.ljust(35)} | {'mAP'.ljust(6)} | {'Rank-1'.ljust(6)} | {'SCE'.ljust(6)} | {'CSCE'.ljust(6)} | {'TRIP'.ljust(6)} | {'CONT'.ljust(6)} | EXTRA INFO\n")
            f.write("-"*50 + "\n")
    
    if not os.path.exists(log_error_file_path):
        with open(log_error_file_path, "w") as f:
            f.write(f"Batch Experiment {pipeline_name} Error for pooling: {pooling}\n")
            f.write("="*50 + "\n")
            f.write(f"{'Backbone'.ljust(35)} | {'Error reason'.ljust(6)}\n")
            f.write("-"*50 + "\n")

    for current_backbone in backbone_list:
        print(f"\n{'='*60}")
        print(f"🔥 ĐANG HUẤN LUYỆN: {current_backbone.upper()} - pooling: {pooling}")
        print(f"{'='*60}")
        
        args.backbone = current_backbone
        try:
            best_map, r1, best_metrics = train_one_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)
            mApCons = best_metrics.get('mAP(Cons)') or 0.0
            mApRef = best_metrics.get('mAP(Ref)') or 0.0
            mApDelta = best_metrics.get('mAP(Delta)') or 0.0
            r1Ref = best_metrics.get('Rank-1(Ref)') or 0.0
            results_summary[current_backbone] = (best_map, r1, best_metrics)
            status = "SUCCESS"
        except Exception as e:
            print(f"❌ Đã có sự cố khi huấn luyện {current_backbone}")
            print(f"Chi tiết lỗi: {e}") # In chi tiết lỗi ra màn hình
            traceback.print_exc()     # In stack trace để dễ debug


            with open(log_error_file_path, "a") as f:
                f.write(f"{current_backbone.ljust(35)} | f{e}\n")
            
            best_map, r1, mApCons, mApRef, mApDelta, r1Ref  = 0.0, 0.0, 0.0,0.0, 0.0, 0.0
            results_summary[current_backbone] = (best_map, r1, {})
            status = "FAILED"
        
        # ==========================================
        # 📝 GHI APPEND NGAY LẬP TỨC SAU MỖI MODEL
        # ==========================================
        # Mở file chế độ "a" (append) để ghi nối tiếp vào cuối file
        with open(log_file_path, "a") as f:
            if status == "SUCCESS":
                f.write(f"{current_backbone.ljust(35)} | {best_map:.4f} | {r1:.4f} | {L_SCE} | {L_CSCE} | {L_TRIPLET} | {L_CONTRASTIVE} | mAP(Cons):{mApCons:.4f} - mAP(Ref):{mApRef:.4f} - mAP(Delta):{mApDelta:.4f} - Rank-1(Ref):{r1Ref:.4f}\n")
            else:
                f.write(f"{current_backbone.ljust(35)} | 0.0000 | 0.0000 (ERROR) | {L_SCE} | {L_CSCE} | {L_TRIPLET} | {L_CONTRASTIVE} | mAP(Cons):{mApCons:.4f} - mAP(Ref):{mApRef:.4f} - mAP(Delta):{mApDelta:.4f} - Rank-1(Ref):{r1Ref:.4f}\n")
        
        # Dọn dẹp RAM/VRAM sau mỗi model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    

    # ==========================================
    # 🏆 IN BẢNG TỔNG KẾT RA MÀN HÌNH SAU CÙNG
    # ==========================================
    print("\n" + "="*50)
    print("🏆 BÁO CÁO TỔNG KẾT (FOLD 0)")
    print("="*70)
    print(f"{'Backbone'.ljust(35)} | {'mAP'.ljust(6)} | {'Rank-1'.ljust(6)} | {'SCE'.ljust(6)} | {'CSCE'.ljust(6)} | {'TRIP'.ljust(6)} | {'CONT'.ljust(6)} | Extra info")
    for bb, (mAP, r1, best_metrics) in results_summary.items():
        mApCons = best_metrics.get('mAP(Cons)') or 0.0
        mApRef = best_metrics.get('mAP(Ref)') or 0.0
        mApDelta = best_metrics.get('mAP(Delta)') or 0.0
        r1Ref = best_metrics.get('Rank-1(Ref)') or 0.0
        print(f"{bb.ljust(35)} | {mAP:.4f} | {r1:.4f} | {L_SCE} | {L_CSCE} | {L_TRIPLET} | {L_CONTRASTIVE} | mAP(Cons):{mApCons:.4f} - mAP(Ref):{mApRef:.4f} - mAP(Delta):{mApDelta:.4f} - Rank-1(Ref):{r1Ref:.4f}")
    print("="*70)
    print(f"📁 Toàn bộ kết quả đã được lưu an toàn tại: {log_file_path}")

if __name__ == '__main__':
    main()