import torch
import numpy as np
from tqdm import tqdm

def evaluate_retrieval(model, dataloader, device):
    """
    Đánh giá mô hình theo tiêu chuẩn Max-Matching của bài báo ePillID.
    Query: Consumer Images (1 mặt)
    Gallery: Reference Images (Gộp các mặt lại, lấy Max Similarity)
    """
    model.eval()
    all_embs = []
    all_labels = []
    all_is_refs = []

    # 1. Trích xuất đặc trưng (Embedding Extraction)
    with torch.no_grad():
        for imgs, labels, is_refs in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            # Lấy norm_emb từ tuple (logits_sce, logits_csce, norm_emb) hoặc tensor đơn lẻ
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

    # 3. Gom nhóm Gallery theo nhãn (Appearance Class)
    # Vì một loại thuốc (Label) có thể có nhiều ảnh Reference (mặt trước/mặt sau)
    unique_g_labels = np.unique(g_labels)
    gallery_by_label = {}
    for label in unique_g_labels:
        idx = np.where(g_labels == label)[0]
        gallery_by_label[label] = g_embs[idx]

    # 4. Tính toán Max-Matching Similarity
    # Thay vì ma trận [Q x G_ảnh], ta tính ma trận [Q x G_loại_thuốc]
    num_queries = len(q_labels)
    num_gallery_labels = len(unique_g_labels)
    
    # Ma trận chứa độ tương đồng lớn nhất của mỗi Query với mỗi loại thuốc
    max_sims = torch.zeros((num_queries, num_gallery_labels))

    for j, label in enumerate(unique_g_labels):
        label_embs = gallery_by_label[label] # [N_mặt, D]
        # Tính similarity của tất cả Query với các mặt của label này
        sim_matrix = torch.mm(q_embs, label_embs.t()) # [Q, N_mặt]
        # Lấy Max theo hàng (mặt khớp nhất)
        max_sims[:, j] = torch.max(sim_matrix, dim=1)[0]

    # 5. Tính toán Metrics (mAP, Rank-k) dựa trên nhãn loại thuốc
    aps = []
    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0

    for i in range(num_queries):
        query_label = q_labels[i]
        # Điểm tương đồng của query i với tất cả các loại thuốc trong gallery
        scores = max_sims[i].numpy()
        
        # Sắp xếp các loại thuốc theo điểm số giảm dần
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = unique_g_labels[sorted_indices]
        
        # Kiểm tra khớp nhãn
        binary_hits = (sorted_labels == query_label).astype(int)
        
        # Rank-k
        if binary_hits[0] == 1: rank1_correct += 1
        if np.any(binary_hits[:5]): rank5_correct += 1
        if np.any(binary_hits[:10]): rank10_correct += 1
        
        # Average Precision (AP)
        num_relevant = np.sum(binary_hits) 
        if num_relevant == 0:
            aps.append(0)
            continue
            
        relevant_ranks = np.where(binary_hits == 1)[0] + 1
        # Trong retrieval chuẩn, mỗi label chỉ có 1 "vị trí đúng" sau khi đã Max-Pooling
        # nên AP = 1 / rank_của_nhãn_đúng
        ap = 1.0 / relevant_ranks[0]
        aps.append(ap)

    metrics = {
        'mAP': np.mean(aps),
        'Rank-1': rank1_correct / num_queries,
        'Rank-5': rank5_correct / num_queries,
        'Rank-10': rank10_correct / num_queries
    }

    return metrics