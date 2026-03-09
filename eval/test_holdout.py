import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data

def get_ensemble_embeddings(models, dataloader, device):
    """Trích xuất đặc trưng kết hợp Ensemble 4-fold và Flip-Matching"""
    for m in models: m.eval()
    all_embs, all_labels, all_is_refs, all_instances = [], [], [], []

    with torch.no_grad():
        for imgs, labels, is_refs, instance_ids in tqdm(dataloader, desc="🚀 Trích xuất đặc trưng"):
            imgs = imgs.to(device)
            imgs_flip = torch.flip(imgs, dims=[3])
            
            batch_embs = []
            for model in models:
                with torch.amp.autocast('cuda'):
                    e_orig = model(imgs)
                    if isinstance(e_orig, tuple): e_orig = e_orig[-1]
                    e_flip = model(imgs_flip)
                    if isinstance(e_flip, tuple): e_flip = e_flip[-1]
                    batch_embs.append((e_orig + e_flip) / 2.0)
            
            ensemble_emb = torch.stack(batch_embs).mean(dim=0)
            ensemble_emb = F.normalize(ensemble_emb, p=2, dim=1)
            
            all_embs.append(ensemble_emb.cpu())
            all_labels.append(labels)
            all_is_refs.append(is_refs)
            all_instances.append(instance_ids)

    return torch.cat(all_embs), torch.cat(all_labels).numpy(), \
           torch.cat(all_is_refs).numpy(), torch.cat(all_instances).numpy()

def compute_metrics(sim_matrix, query_labels, gallery_labels):
    """Hàm lõi tính mAP và Rank-1"""
    aps = []
    rank1 = 0
    for i in range(len(query_labels)):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = gallery_labels[sorted_indices]
        
        if sorted_labels[0] == query_labels[i]:
            rank1 += 1
        
        hit_rank = np.where(sorted_labels == query_labels[i])[0][0] + 1
        aps.append(1.0 / hit_rank)
    
    return np.mean(aps), rank1 / len(query_labels)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)

    # 1. Chuẩn bị tập Hold-out (Fold 4)
    # Lấy thêm cột pill_instance_id để phục vụ Both-side
    df_query = df_all[(df_all['fold'] == 4) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    df_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    test_df = pd.concat([df_query, df_gallery]).reset_index(drop=True)
    
    # Custom Dataset cần trả về thêm instance_id (giả định bạn đã update Dataset)
    # Nếu chưa update PillDataset, bạn có thể map instance_id thủ công sau.
    test_loader = DataLoader(PillDataset(test_df, transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), batch_size=16, shuffle=False)

    # 2. Load Models
    models = []
    for i in range(4):
        path = f'weights/teacher_cv_f{i}.pth'
        if os.path.exists(path):
            m = PillTeacher(num_classes=num_classes).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            models.append(m)

    # 3. Trích xuất (Kết quả thô cho từng ảnh)
    embs, labels, is_refs, instance_ids = get_ensemble_embeddings(models, test_loader, device)
    
    q_embs, q_labels, q_instances = embs[is_refs==0], labels[is_refs==0], instance_ids[is_refs==0]
    g_embs, g_labels = embs[is_refs==1], labels[is_refs==1]
    unique_g_labels = np.unique(g_labels)

    # 4. Tính Raw Similarity (Query Images vs Gallery Labels)
    raw_sims = torch.zeros((len(q_embs), len(unique_g_labels)))
    for j, label in enumerate(tqdm(unique_g_labels, desc="🔍 So khớp Gallery")):
        label_embs = g_embs[g_labels == label]
        sims = torch.mm(q_embs, label_embs.t())
        raw_sims[:, j] = torch.max(sims, dim=1)[0]

    # ==========================================
    # PHẦN BÁO CÁO SO SÁNH
    # ==========================================
    
    # A. SINGLE-SIDE EVALUATION
    mAP_s, r1_s = compute_metrics(raw_sims.numpy(), q_labels, unique_g_labels)
    
    # B. BOTH-SIDE EVALUATION (Grouping by Instance)
    unique_pills = np.unique(q_instances)
    both_sims, both_labels = [], []
    for p_id in unique_pills:
        idx = np.where(q_instances == p_id)[0]
        # Max-Fusion giữa các mặt của cùng 1 viên thuốc
        both_sims.append(torch.max(raw_sims[idx], dim=0)[0])
        both_labels.append(q_labels[idx[0]])
    
    mAP_b, r1_b = compute_metrics(torch.stack(both_sims).numpy(), np.array(both_labels), unique_g_labels)

    print("\n" + "="*45)
    print(f"{'Hạng mục':<20} | {'Single-side':<12} | {'Both-side':<12}")
    print("-" * 45)
    print(f"{'mAP':<20} | {mAP_s:<12.4f} | {mAP_b:<12.4f}")
    print(f"{'Rank-1 Accuracy':<20} | {r1_s:<12.4f} | {r1_b:<12.4f}")
    print(f"{'Số lượng mẫu':<20} | {len(q_labels):<12} | {len(unique_pills):<12}")
    print("="*45)
    
    # Lưu kết quả ra file csv phục vụ viết báo cáo
    report = pd.DataFrame({
        'Metric': ['mAP', 'Rank-1', 'Samples'],
        'Single-side': [mAP_s, r1_s, len(q_labels)],
        'Both-side': [mAP_b, r1_b, len(unique_pills)]
    })
    report.to_csv('analysis/final_benchmark_comparison.csv', index=False)

if __name__ == '__main__':
    main()