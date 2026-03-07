import torch
import numpy as np

def calculate_cosine_similarity(query_features, gallery_features):
    """
    Tính toán ma trận độ tương đồng Cosine giữa Query và Gallery.
    Query: (M, D), Gallery: (N, D) -> Kết quả: (M, N)
    """
    query_norm = torch.nn.functional.normalize(query_features, p=2, dim=1)
    gallery_norm = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
    return torch.mm(query_norm, gallery_norm.t())

def evaluate_retrieval(similarity_matrix, query_labels, gallery_labels, topk=[1, 5, 10]):
    """
    Tính toán Rank-k và mAP.
    Args:
        similarity_matrix (Tensor): (M, N) Ma trận tương đồng
        query_labels (Tensor): (M,) Nhãn của tập Query
        gallery_labels (Tensor): (N,) Nhãn của tập Gallery
        topk (list): Các mốc Rank cần tính
    """
    # Sắp xếp index theo độ tương đồng giảm dần (M, N)
    indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # Chuyển index thành nhãn tương ứng (M, N)
    matched_labels = gallery_labels[indices]
    
    # Tạo ma trận boolean: True nếu khớp nhãn (M, N)
    correct_mask = (matched_labels == query_labels.view(-1, 1))
    
    results = {}

    # 1. Tính Rank-k Accuracy
    for k in topk:
        # Nếu có ít nhất 1 ảnh đúng trong k ảnh đầu tiên
        hits = torch.any(correct_mask[:, :k], dim=1).float()
        results[f'Rank-{k}'] = torch.mean(hits).item() * 100

    # 2. Tính mean Average Precision (mAP)
    # Tính Precision tại mỗi vị trí có ảnh đúng
    aps = []
    for i in range(correct_mask.size(0)):
        row = correct_mask[i]
        if not torch.any(row):
            aps.append(0.0)
            continue
            
        # Vị trí của các ảnh đúng (1-based indexing)
        p_indices = torch.where(row)[0] + 1
        
        # Số lượng ảnh đúng tích lũy tại mỗi vị trí đúng
        relevant_counts = torch.arange(1, len(p_indices) + 1).to(p_indices.device)
        
        # Average Precision = Mean of (Relevant_Counts / Position_Index)
        ap = torch.mean(relevant_counts.float() / p_indices.float())
        aps.append(ap.item())
        
    results['mAP'] = np.mean(aps) * 100
    
    return results

# --- Ví dụ cách sử dụng trong Pipeline đánh giá ---
if __name__ == "__main__":
    # Giả lập 10 ảnh Query và 50 ảnh Gallery, mỗi ảnh có vector 512 chiều
    q_feat = torch.randn(10, 512)
    g_feat = torch.randn(50, 512)
    
    # Giả lập nhãn (ví dụ có 5 loại thuốc)
    q_labels = torch.randint(0, 5, (10,))
    g_labels = torch.randint(0, 5, (50,))
    
    sim_mat = calculate_cosine_similarity(q_feat, g_feat)
    metrics = evaluate_retrieval(sim_mat, q_labels, g_labels)
    
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")