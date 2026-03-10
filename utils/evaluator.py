import torch
import numpy as np
from tqdm import tqdm

def evaluate_retrieval(model, dataloader, device):
    """
    Đánh giá mô hình theo tiêu chuẩn Max-Matching + Flip-Augmentation (TTA).
    Phù hợp với chiến lược 'ép' mAP 0.79.
    """
    model.eval()
    all_embs = []
    all_labels = []
    all_is_refs = []

    # Sử dụng no_grad và autocast để tăng tốc eval và tiết kiệm VRAM
    with torch.no_grad():
        for imgs, labels, is_refs in tqdm(dataloader, desc="Eval (Flip-Match)", leave=False):
            imgs = imgs.to(device)
            
            with torch.amp.autocast('cuda'):
                # 1. Trích xuất ảnh gốc - Lấy norm_emb (phần tử cuối cùng trong tuple)
                out_orig = model(imgs)
                emb_orig = out_orig[-1] if isinstance(out_orig, tuple) else out_orig
                
                # 2. Trích xuất ảnh lật ngang (TTA - Test Time Augmentation)
                out_flip = model(torch.flip(imgs, dims=[3]))
                emb_flip = out_flip[-1] if isinstance(out_flip, tuple) else out_flip
                
                # 3. Trung bình cộng và chuẩn hóa L2 (Fusion embedding)
                # Đây là lý do mAP của bạn cao hơn bình thường
                emb_final = (emb_orig + emb_flip) / 2.0
                emb_final = torch.nn.functional.normalize(emb_final, p=2, dim=1)
            
            # Chuyển về CPU và half precision để tránh tràn RAM hệ thống
            all_embs.append(emb_final.cpu().half())
            all_labels.append(labels)
            all_is_refs.append(is_refs)

    # Tổng hợp dữ liệu
    all_embs = torch.cat(all_embs).float()
    all_labels = torch.cat(all_labels).numpy()
    all_is_refs = torch.cat(all_is_refs).numpy()

    # Tách tập Query (Consumer) và tập Gallery (Reference)
    q_indices = np.where(all_is_refs == 0)[0]
    g_indices = np.where(all_is_refs == 1)[0]
    
    q_embs, q_labels = all_embs[q_indices], all_labels[q_indices]
    g_embs, g_labels = all_embs[g_indices], all_labels[g_indices]

    # Max-Matching logic: Một Query đối chiếu với TẤT CẢ các ảnh cùng nhãn trong Gallery
    unique_g_labels = np.unique(g_labels)
    # Gom nhóm Gallery theo nhãn để tính toán nhanh hơn
    gallery_by_label = {l: g_embs[g_labels == l] for l in unique_g_labels}
    
    max_sims = torch.zeros((len(q_labels), len(unique_g_labels)))
    
    # Tính toán ma trận tương đồng cực đại
    for j, label in enumerate(unique_g_labels):
        label_embs = gallery_by_label[label]
        # Similarity giữa toàn bộ Query với các ảnh của nhãn hiện tại
        sim_matrix = torch.mm(q_embs, label_embs.t())
        # Lấy giá trị tương đồng cao nhất (Max-matching)
        max_sims[:, j] = torch.max(sim_matrix, dim=1)[0]

    aps = []
    rank1_correct = 0
    
    # Tính mAP và Rank-1
    for i in range(len(q_labels)):
        scores = max_sims[i].numpy()
        # Sắp xếp các nhãn theo điểm số giảm dần
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = unique_g_labels[sorted_indices]
        
        # Kiểm tra Rank-1
        if sorted_labels[0] == q_labels[i]:
            rank1_correct += 1
        
        # Tìm vị trí của nhãn đúng để tính AP
        binary_hits = (sorted_labels == q_labels[i]).astype(int)
        relevant_rank = np.where(binary_hits == 1)[0] + 1
        # Với Retrieval 1-đối-nhiều nhưng tính theo nhãn, AP = 1/rank
        aps.append(1.0 / relevant_rank[0])

    return {
        'mAP': np.mean(aps), 
        'Rank-1': rank1_correct / len(q_labels)
    }