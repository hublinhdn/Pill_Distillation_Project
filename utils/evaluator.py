import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_retrieval(model, loader, device):
    """
    Đánh giá mô hình theo tiêu chuẩn Max-Matching + Flip-Augmentation.
    Trích xuất vector đặc trưng của ảnh gốc và ảnh lật ngang, sau đó cộng trung bình.
    """
    model.eval()
    all_embs, all_labels, all_is_refs = [], [], []

    with torch.no_grad():
        for imgs, sub_labels, labels, is_refs in loader:
            imgs = imgs.to(device)
            
            # --- 1. BỔ SUNG FLIP-AUGMENTATION (TTA) ---
            # Trích xuất ảnh gốc
            emb_orig = model(imgs)
            
            # Lật ảnh theo chiều ngang (chiều rộng = dim 3) và trích xuất
            imgs_flipped = torch.flip(imgs, dims=[3])
            emb_flipped = model(imgs_flipped)
            
            # Cộng trung bình và chuẩn hóa lại (L2 Norm)
            emb_fused = (emb_orig + emb_flipped) / 2.0
            emb_fused = F.normalize(emb_fused, p=2, dim=1)
            
            all_embs.append(emb_fused.cpu())
            all_labels.append(labels) 
            all_is_refs.append(is_refs)

    all_embs = torch.cat(all_embs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_is_refs = torch.cat(all_is_refs).numpy()

    q_embs = all_embs[all_is_refs == 0]
    q_labels = all_labels[all_is_refs == 0]
    g_embs = all_embs[all_is_refs == 1]
    g_labels = all_labels[all_is_refs == 1]

    sim_matrix = np.dot(q_embs, g_embs.T)

    unique_g_labels = np.unique(g_labels)
    aps = []
    r1 = 0

    # --- 2. TỐI ƯU TỐC ĐỘ MAX-SCORE ---
    # Tìm trước index của các mặt thuốc cho mỗi loại (Chỉ chạy 1 lần)
    g_lab_indices = [np.where(g_labels == g_lab)[0] for g_lab in unique_g_labels]

    for i in range(len(q_labels)):
        scores = sim_matrix[i]
        
        # Lấy max score cực nhanh dựa trên list index đã tính sẵn
        pill_scores = np.array([np.max(scores[idx]) for idx in g_lab_indices])

        sorted_indices = np.argsort(-pill_scores)
        pred_labels = unique_g_labels[sorted_indices]
        
        if pred_labels[0] == q_labels[i]:
            r1 += 1
            
        rank = np.where(pred_labels == q_labels[i])[0][0] + 1
        aps.append(1.0 / rank)
        
    return {'mAP': np.mean(aps), 'Rank-1': r1 / len(q_labels)}