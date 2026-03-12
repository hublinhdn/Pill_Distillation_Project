import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_retrieval(model, loader, device):
    """
    Đánh giá mô hình theo tiêu chuẩn Max-Matching.
    Đã loại bỏ Flip-Augmentation để bảo toàn đặc trưng chữ (imprint) của thuốc.
    """
    model.eval()
    all_embs, all_labels, all_is_refs = [], [], []

    with torch.no_grad():
        for imgs, sub_labels, labels, is_refs in loader: # Vẫn nhận sub_labels từ loader nhưng bỏ qua
            imgs = imgs.to(device)
            
            # --- CHỈ LẤY ĐẶC TRƯNG GỐC ---
            emb_orig = model(imgs)
            emb_norm = F.normalize(emb_orig, p=2, dim=1) # Chuẩn hóa L2
            
            all_embs.append(emb_norm.cpu())
            all_labels.append(labels) 
            all_is_refs.append(is_refs)

    all_embs = torch.cat(all_embs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_is_refs = torch.cat(all_is_refs).numpy()

    q_embs = all_embs[all_is_refs == 0]
    q_labels = all_labels[all_is_refs == 0]
    g_embs = all_embs[all_is_refs == 1]
    g_labels = all_labels[all_is_refs == 1]

    # Tính ma trận độ tương đồng (Cosine Similarity)
    sim_matrix = np.dot(q_embs, g_embs.T)

    unique_g_labels = np.unique(g_labels)
    aps = []
    r1 = 0

    # --- TỐI ƯU TỐC ĐỘ MAX-SCORE ---
    # Tìm trước index của các mặt thuốc cho mỗi loại (Chỉ chạy 1 lần)
    g_lab_indices = [np.where(g_labels == g_lab)[0] for g_lab in unique_g_labels]

    for i in range(len(q_labels)):
        scores = sim_matrix[i]
        
        # Lấy max score: Giả sử Gallery có mặt trước score 0.9, mặt sau score 0.2 -> Lấy 0.9
        pill_scores = np.array([np.max(scores[idx]) for idx in g_lab_indices])

        sorted_indices = np.argsort(-pill_scores)
        pred_labels = unique_g_labels[sorted_indices]
        
        if pred_labels[0] == q_labels[i]:
            r1 += 1
            
        rank = np.where(pred_labels == q_labels[i])[0][0] + 1
        aps.append(1.0 / rank)
        
    return {'mAP': np.mean(aps), 'Rank-1': r1 / len(q_labels)}