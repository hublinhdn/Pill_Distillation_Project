import torch
import numpy as np
from tqdm import tqdm

def evaluate_retrieval(model, dataloader, device):
    """
    Đánh giá mô hình theo tiêu chuẩn Image Retrieval của bài báo ePillID.
    Query: Consumer Images | Gallery: Reference Images
    Metrics: mAP, Rank-1, Rank-5, Rank-10
    """
    model.eval()
    all_embs = []
    all_labels = []
    all_is_refs = []

    # 1. Trích xuất đặc trưng (Embedding Extraction)
    with torch.no_grad():
        for imgs, labels, is_refs in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            # Model trả về norm_emb (vector đã L2 normalize)
            # Lưu ý: PillTeacher bản mới nhất trả về (logits_sce, logits_csce, norm_emb)
            outputs = model(imgs)
            emb = outputs[-1] if isinstance(outputs, tuple) else outputs
            
            all_embs.append(emb.cpu())
            all_labels.append(labels)
            all_is_refs.append(is_refs)

    all_embs = torch.cat(all_embs)
    all_labels = torch.cat(all_labels).numpy()
    all_is_refs = torch.cat(all_is_refs).numpy()

    # 2. Tách Query (Consumer) và Gallery (Reference)
    q_indices = np.where(all_is_refs == 0)[0]
    g_indices = np.where(all_is_refs == 1)[0]

    q_embs = all_embs[q_indices]
    q_labels = all_labels[q_indices]
    g_embs = all_embs[g_indices]
    g_labels = all_labels[g_indices]

    # 3. Tính độ tương đồng Cosine
    # Vì vector đã chuẩn hóa L2, tích vô hướng chính là Cosine Similarity
    sims = torch.mm(q_embs, g_embs.t()) # [N_query, N_gallery]

    aps = []
    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0

    # 4. Tính toán Metrics cho từng Query
    for i in range(len(q_labels)):
        query_label = q_labels[i]
        query_sims = sims[i].numpy()
        
        # Sắp xếp Gallery theo độ tương đồng giảm dần
        sorted_indices = np.argsort(query_sims)[::-1]
        sorted_labels = g_labels[sorted_indices]
        
        # Mảng nhị phân: 1 nếu khớp nhãn (Appearance Class), 0 nếu sai
        binary_hits = (sorted_labels == query_label).astype(int)
        
        # Tính Rank-k Accuracy
        if binary_hits[0] == 1: rank1_correct += 1
        if np.any(binary_hits[:5]): rank5_correct += 1
        if np.any(binary_hits[:10]): rank10_correct += 1
        
        # Tính Average Precision (AP)
        num_correct = np.sum(binary_hits)
        if num_correct == 0:
            aps.append(0)
            continue
            
        relevant_ranks = np.where(binary_hits == 1)[0] + 1
        ap = np.sum(np.arange(1, num_correct + 1) / relevant_ranks) / num_correct
        aps.append(ap)

    # 5. Tổng hợp kết quả
    metrics = {
        'mAP': np.mean(aps),
        'Rank-1': rank1_correct / len(q_labels),
        'Rank-5': rank5_correct / len(q_labels),
        'Rank-10': rank10_correct / len(q_labels)
    }
    
    return metrics