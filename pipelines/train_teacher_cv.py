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
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    L_SCE, L_CSCE, L_TRIPLET, L_CONTRASTIVE = 1.0, 0.2, 1.0, 1.0
    USE_SHAPE_LOSS, L_SHAPE = False, 1.0 # Enable need more GPU cause 2 model embedding 
    # debug
    L_CONTRASTIVE = 0.0

    # XÓA ĐOẠN IF/ELIF CŨ VÀ DÒNG img_size = 384 GHI ĐÈ ĐI, THAY BẰNG ĐOẠN NÀY:
    
    backbone_name = args.backbone.lower()
    
    # 1. NHÓM SIÊU NẶNG (XLARGE): > 100 Triệu Tham số (OOM Risk: High)
    if any(x in backbone_name for x in ['xlarge', 'large', 'b6', 'b7', 'v2_l']):
        img_size = 384
        n_classes_batch, n_samples = 2, 2  # Physical Batch = 4 ảnh
        accumulation_steps = 32            # Effective Batch = 4 * 32 = 128
        lr_backbone, lr_head = 1e-5, 1e-4  # Mạng to cần LR nhỏ để tránh sốc Gradient

    # 2. NHÓM NẶNG (LARGE/BASE): 40M - 100M Tham số
    elif any(x in backbone_name for x in ['base', 'b5', 'resnet101', 'resnet152', 'resnext101', 'densenet161', 'seresnext101_32x4d']):
        img_size = 384
        n_classes_batch, n_samples = 4, 2  # Physical Batch = 8 ảnh
        accumulation_steps = 16            # Effective Batch = 8 * 16 = 128
        lr_backbone, lr_head = 2e-5, 2e-4

    # 3. NHÓM TRUNG BÌNH (MEDIUM): 15M - 40M Tham số
    elif any(x in backbone_name for x in ['resnet50', 'b3', 'b4', 'nfnet']):
        img_size = 384
        n_classes_batch, n_samples = 8, 2  # Physical Batch = 16 ảnh
        accumulation_steps = 8             # Effective Batch = 16 * 8 = 128
        lr_backbone, lr_head = 3e-5, 3e-4

    # 4. NHÓM SIÊU NHẸ (LIGHTWEIGHT/TINY): < 15M Tham số (Student Models)
    else: # Bao gồm: resnet18, mobilenet, shufflenet, ghostnet, b0, b1, atto, femto, pico...
        img_size = 384
        # Mạng nhỏ VRAM dư dả, ta nhồi Batch to vào để train cực nhanh
        n_classes_batch, n_samples = 16, 2 # Physical Batch = 32 ảnh
        accumulation_steps = 4             # Effective Batch = 32 * 4 = 128
        lr_backbone, lr_head = 5e-5, 5e-4  # Mạng nhỏ cần LR to hơn một chút để học nhanh

    print(f"⚙️ Config {args.backbone}: Size={img_size}x{img_size} | Physical Batch={n_classes_batch*n_samples} | Accum={accumulation_steps} | Eff Batch={n_classes_batch*n_samples*accumulation_steps}")



    
    # # ĐỘNG: QUẢN LÝ VRAM THEO BACKBONE
    # if 'convnext_large' in args.backbone:
    #     img_size = 384
    #     n_classes_batch, n_samples = 4, 2  
    #     accumulation_steps = 16            
    #     lr_backbone, lr_head = 2e-5, 2e-4
    # elif 'convnext_base' in args.backbone:
    #     img_size = 384
    #     n_classes_batch, n_samples = 8, 2  
    #     accumulation_steps = 8             
    #     lr_backbone, lr_head = 3e-5, 3e-4
    # elif 'resnet101' in args.backbone:
    #     img_size = 448
    #     n_classes_batch, n_samples = 8, 2  
    #     accumulation_steps = 8
    #     lr_backbone, lr_head = 3e-5, 3e-4
    # elif 'resnet18' in args.backbone:
    #     img_size = 384
    #     n_classes_batch, n_samples = 16, 2 
    #     accumulation_steps = 4
    #     lr_backbone, lr_head = 4e-5, 4e-4
    # else: 
    #     img_size = 448
    #     n_classes_batch, n_samples = 16, 2 
    #     accumulation_steps = 4
    #     lr_backbone, lr_head = 4e-5, 4e-4

    # img_size = 384 # hard size as limit of hardward
    # print(f"⚙️ Config {args.backbone}: Size={img_size}x{img_size} | Batch={n_classes_batch*n_samples} | Accum={accumulation_steps}")

    resize_scale = int(img_size * 1.15) 

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

    model = PillRetrievalModel(num_classes=num_classes, backbone_type=args.backbone,pooling_type=args.pooling).to(device)

    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")

    # Use for shape
    criterion_shape = nn.MSELoss()

    backbone_params = [p for n, p in model.named_parameters() if 'features' in n]
    head_params = [p for n, p in model.named_parameters() if 'features' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone}, 
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=5e-2)
    
    TOTAL_EPOCHS = 60 
    WARMUP_EPOCHS = 5
    # 1. Warm-up từ từ trong 5 Epoch đầu
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    
    # 2. Cosine Annealing cho 55 Epoch còn lại
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(TOTAL_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6
    )
    
    # 3. Nối 2 cái lại với nhau
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[WARMUP_EPOCHS]
    )

    scaler = torch.amp.GradScaler('cuda')

    best_val_map,r1 = 0.0, 0.0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"{args.backbone} | Fold {f_idx} | Ep {epoch}")
        
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            sub_labels = sub_labels.to(device).long()
            # 1. Tự động tạo ra phiên bản ảnh Xám ngay trong vòng lặp
            # Chú ý: Ảnh xám chỉ có 1 kênh màu, ta phải repeat nó lên 3 kênh để đưa vào ResNet
            imgs_gray = TF.rgb_to_grayscale(imgs, num_output_channels=3)

            with torch.amp.autocast('cuda'):
                logits_sce, logits_csce, norm_embedding_rgb = model(imgs, labels=sub_labels)
                loss_sce = criterion_sce(logits_sce, sub_labels)
                loss_csce = criterion_sce(logits_csce, sub_labels)
                hard_pairs = miner(norm_embedding_rgb, sub_labels)
                loss_triplet = criterion_triplet(norm_embedding_rgb, sub_labels, hard_pairs)
                loss_contrastive = criterion_contrastive(norm_embedding_rgb, sub_labels)

                # ==================================================
                # 4. THÊM SHAPE-CONSISTENCY LOSS (Ý TƯỞNG CỦA BẠN)
                # ==================================================
                # Ép vector ảnh màu phải giống hệt vector ảnh xám
                if USE_SHAPE_LOSS:
                    _, _, norm_embedding_gray = model(imgs_gray, labels=sub_labels) # Chỉ lấy embedding của ảnh xám
                    loss_shape = criterion_shape(norm_embedding_rgb, norm_embedding_gray)
                else:
                    loss_shape = 0.0
                
                # 5. Tổng hợp Loss (Thêm hệ số L_SHAPE, ví dụ 1.0)
                loss = (L_SCE * loss_sce + L_CSCE * loss_csce + L_TRIPLET * loss_triplet + L_CONTRASTIVE * loss_contrastive + L_SHAPE * loss_shape) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'L': f"{loss.item()*accumulation_steps:.4f}", 'mAp': f"{best_val_map:.4f}", 'R1':f"{r1:.4f}"})

        scheduler.step()

        if (epoch % 5 == 0) or (epoch > TOTAL_EPOCHS - 10):
            # Mặc định sử dụng Both Sides để lưu Model tốt nhất
            val_metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=True)
            val_map = val_metrics['mAP']
            val_r1 = val_metrics['Rank-1']
            print(f"📊 Epoch {epoch} mAP: {val_map:.4f} (Rank-1: {val_r1:.4f})")
            
            if val_map > best_val_map:
                best_val_map = val_map
                r1 = val_r1
                folder_weight = 'weights/phase2'
                os.makedirs(folder_weight, exist_ok=True)
                torch.save(model.state_dict(), f"{folder_weight}/best_{args.backbone}_{args.pooling}_fold{f_idx}.pth")
                
    return best_val_map, r1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='convnext_base,resnet50,mobilenet_v3_large')
    parser.add_argument('--pooling', type=str, default='gem', choices=['gem', 'mpncov'])
    parser.add_argument('--pipeline_name', type=str, default='baseline')
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
    elif torch.backends.mps.is_available(): 
        device = torch.device('mps')
        device_type = 'mps'
    else: 
        device = torch.device('cpu')
        device_type = 'cpu'
    
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
    
    results_summary = {}
    pooling = args.pooling
    pipeline_name = args.pipeline_name

    # ==========================================
    # 📝 TẠO FILE LOG VÀ GHI HEADER TRƯỚC
    # ==========================================
    os.makedirs("reports", exist_ok=True)
    log_file_path = f"reports/{pipeline_name}_batch_experiment_{pooling}_summary.txt"
    
    # Mở file chế độ "w" (write) lần đầu để tạo file mới và ghi tiêu đề
    with open(log_file_path, "w") as f:
        f.write(f"Batch Experiment {pipeline_name} Results for pooling: {pooling}\n")
        f.write("="*50 + "\n")
        f.write(f"{'Backbone'.ljust(25)} | {'mAP'.ljust(6)} | {'Rank-1'.ljust(6)}\n")
        f.write("-"*50 + "\n")

    for current_backbone in backbone_list:
        print(f"\n{'='*60}")
        print(f"🔥 ĐANG HUẤN LUYỆN: {current_backbone.upper()} - pooling: {pooling}")
        print(f"{'='*60}")
        
        args.backbone = current_backbone
        try:
            best_map, r1 = train_one_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)
            results_summary[current_backbone] = (best_map, r1)
            status = "SUCCESS"
        except Exception as e:
            print(f"❌ Đã có sự cố khi huấn luyện {current_backbone}")
            print(f"Chi tiết lỗi: {e}") # In chi tiết lỗi ra màn hình
            traceback.print_exc()     # In stack trace để dễ debug
            
            best_map, r1 = 0.0, 0.0
            results_summary[current_backbone] = (best_map, r1)
            status = "FAILED"
        
        # ==========================================
        # 📝 GHI APPEND NGAY LẬP TỨC SAU MỖI MODEL
        # ==========================================
        # Mở file chế độ "a" (append) để ghi nối tiếp vào cuối file
        with open(log_file_path, "a") as f:
            if status == "SUCCESS":
                f.write(f"{current_backbone.ljust(25)} | {best_map:.4f} | {r1:.4f}\n")
            else:
                f.write(f"{current_backbone.ljust(25)} | 0.0000 | 0.0000 (ERROR)\n")
        
        # Dọn dẹp RAM/VRAM sau mỗi model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================
    # 🏆 IN BẢNG TỔNG KẾT RA MÀN HÌNH SAU CÙNG
    # ==========================================
    print("\n" + "="*50)
    print("🏆 BÁO CÁO TỔNG KẾT (FOLD 0)")
    print("="*50)
    print(f"{'Backbone'.ljust(25)} | {'mAP'.ljust(6)} | {'Rank-1'.ljust(6)}")
    for bb, (mAP, r1) in results_summary.items():
        print(f"{bb.ljust(25)} | {mAP:.4f} | {r1:.4f} ")
    print("="*50)
    print(f"📁 Toàn bộ kết quả đã được lưu an toàn tại: {log_file_path}")

if __name__ == '__main__':
    main()