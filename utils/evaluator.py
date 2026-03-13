import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def evaluate_retrieval(model, dataloader, device, flag_both_side=True):
    model.eval()
    all_feats, all_labels, all_sub_labels, all_is_ref = [], [], [], []

    # 1. Trích xuất đặc trưng
    with torch.no_grad():
        for imgs, sub_labels, labels, is_ref in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
            all_sub_labels.append(sub_labels.cpu())
            all_is_ref.append(is_ref.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    all_labels = torch.cat(all_labels, dim=0)
    all_sub_labels = torch.cat(all_sub_labels, dim=0)
    all_is_ref = torch.cat(all_is_ref, dim=0)

    # 2. Phân tách Query (is_ref=0) và Gallery (is_ref=1)
    g_mask = all_is_ref == 1
    q_mask = all_is_ref == 0

    g_feats, g_labels, g_sub_labels = all_feats[g_mask], all_labels[g_mask], all_sub_labels[g_mask]
    q_feats, q_labels, q_sub_labels = all_feats[q_mask], all_labels[q_mask], all_sub_labels[q_mask]

    unique_classes = torch.unique(g_labels)
    n_classes = len(unique_classes)
    
    # =====================================================================
    # 3. SET-TO-SET MAX-POOLING (Giải quyết triệt để Gallery có nhiều ảnh)
    # =====================================================================
    # S có kích thước (N_queries, N_gallery) - Tính điểm 1 lần cho mọi tổ hợp
    S = torch.mm(q_feats, g_feats.T)
    
    # Ma trận lưu điểm MAX của Query tới Mặt A và Mặt B của từng Class
    sim_A = torch.zeros((len(q_feats), n_classes))
    sim_B = torch.zeros((len(q_feats), n_classes))
    
    class_list = []
    
    for c_idx, lbl in enumerate(unique_classes):
        lbl = lbl.item()
        class_list.append(lbl)
        
        mask_c = (g_labels == lbl)
        mask_front = mask_c & (g_sub_labels % 2 == 0)
        mask_back = mask_c & (g_sub_labels % 2 != 0)
        
        # Lấy điểm cao nhất của Query so với TẤT CẢ các ảnh Mặt A trong Gallery
        if mask_front.any():
            sim_A[:, c_idx] = S[:, mask_front].max(dim=1).values
        else:
            sim_A[:, c_idx] = -1.0 
            
        # Lấy điểm cao nhất của Query so với TẤT CẢ các ảnh Mặt B trong Gallery
        if mask_back.any():
            sim_B[:, c_idx] = S[:, mask_back].max(dim=1).values
        else:
            sim_B[:, c_idx] = -1.0

    class_tensor = torch.tensor(class_list)
    ranks = []

    # =====================================================================
    # 4. CHẤM ĐIỂM
    # =====================================================================
    if not flag_both_side:
        # --- A. SINGLE SIDE ---
        sim_scores = torch.max(sim_A, sim_B) # Lấy mặt nào giống nhất làm điểm
        for i in range(len(q_feats)):
            q_lbl = q_labels[i].item()
            scores = sim_scores[i]
            sorted_indices = torch.argsort(scores, descending=True)
            sorted_classes = class_tensor[sorted_indices]
            rank = (sorted_classes == q_lbl).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            
    else:
        # --- B. BOTH SIDES LATE FUSION (CỘNG ĐỒNG THỜI - CROSS MAX) ---
        unique_q_labels = torch.unique(q_labels)
        
        for lbl in unique_q_labels:
            lbl = lbl.item()
            mask_lbl = (q_labels == lbl)
            
            q_indices = mask_lbl.nonzero(as_tuple=True)[0]
            q_subs = q_sub_labels[q_indices]
            
            front_idx = q_indices[q_subs % 2 == 0]
            back_idx = q_indices[q_subs % 2 != 0]
            
            n_pairs = min(len(front_idx), len(back_idx))
            
            # Chấm điểm cho các cặp ảnh (Pairs)
            for i in range(n_pairs):
                idx1 = front_idx[i]
                idx2 = back_idx[i]
                
                # CÔNG THỨC CHUẨN MỰC: Phép CỘNG (SUM) để 2 mặt gánh nhau
                # Tổ hợp 1: Q1 khớp Mặt A, Q2 khớp Mặt B
                fuse_1 = sim_A[idx1] + sim_B[idx2] 
                # Tổ hợp 2: Q1 khớp Mặt B, Q2 khớp Mặt A
                fuse_2 = sim_B[idx1] + sim_A[idx2] 
                
                # Lấy kịch bản xoay chiều tốt nhất
                sim_scores = torch.max(fuse_1, fuse_2)
                
                sorted_indices = torch.argsort(sim_scores, descending=True)
                sorted_classes = class_tensor[sorted_indices]
                rank = (sorted_classes == lbl).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
                
            # Chấm điểm cho các ảnh lẻ (Leftovers) - Đánh giá như Single Side
            leftover_idx = torch.cat([front_idx[n_pairs:], back_idx[n_pairs:]])
            for idx in leftover_idx:
                sim_scores = torch.max(sim_A[idx], sim_B[idx])
                sorted_indices = torch.argsort(sim_scores, descending=True)
                sorted_classes = class_tensor[sorted_indices]
                rank = (sorted_classes == lbl).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

    ranks = np.array(ranks)
    rank1 = (ranks == 1).mean()
    mAP = (1.0 / ranks).mean()

    return {
        'Rank-1': rank1,
        'mAP': mAP
    }