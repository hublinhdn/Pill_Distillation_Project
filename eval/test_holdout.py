import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os, sys
import argparse
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data
from utils.metrics import calculate_cosine_similarity, evaluate_retrieval_metrics

def extract_ensemble_features(models, dataloader, device):
    """
    Trích xuất đặc trưng kết hợp:
    1. Ensemble: Trung bình cộng embedding từ N models (4 folds).
    2. Test-time Augmentation (TTA): Trung bình cộng ảnh gốc và ảnh lật.
    """
    for m in models:
        m.eval()
    
    all_embs, all_labels, all_instances = [], [], []
    
    with torch.no_grad():
        for imgs, labels, _, instance_ids in tqdm(dataloader, desc="🚀 Ensemble Extraction"):
            imgs = imgs.to(device)
            imgs_flip = torch.flip(imgs, dims=[3])
            
            # Danh sách chứa embedding của từng model trong ensemble
            model_outputs = []
            
            for model in models:
                with torch.cuda.amp.autocast():
                    # TTA cho từng model
                    e_orig = model(imgs)["embedding"]
                    e_flip = model(imgs_flip)["embedding"]
                    avg_tta = (e_orig + e_flip) / 2.0
                    model_outputs.append(avg_tta)
            
            # Trung bình cộng embedding của tất cả models trong ensemble
            ensemble_emb = torch.stack(model_outputs).mean(dim=0)
            ensemble_emb = F.normalize(ensemble_emb, p=2, dim=1) # Chuẩn hóa lại sau khi cộng
            
            all_embs.append(ensemble_emb.cpu())
            all_labels.append(labels)
            all_instances.append(instance_ids)
            
    return torch.cat(all_embs), torch.cat(all_labels), torch.cat(all_instances)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)

    # 1. Tự động tìm tất cả các file weights của backbone tương ứng
    # TỰ ĐỘNG QUÉT: Tìm trong weights/{backbone}/fold*_best.pth
    weight_pattern = os.path.join("weights", args.backbone, "fold*_best.pth")
    weight_files = sorted(glob.glob(weight_pattern))
    
    if not weight_files:
        print(f"❌ Không tìm thấy trọng số tại: {weight_pattern}")
        return

    print(f"📦 Đang Ensemble {len(weight_files)} mô hình từ thư mục weights/{args.backbone}/")


    # 2. Load tất cả models vào list
    models_list = []
    for wf in weight_files:
        model = PillTeacher(backbone_name=args.backbone, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(wf, map_location=device))
        models_list.append(model)

    # 3. Chuẩn bị Data
    img_size = 300 if 'efficientnet' in args.backbone else 224
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    df_query = df_all[(df_all['fold'] == 4) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    df_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    
    query_loader = DataLoader(PillDataset(df_query, transform=test_transform), batch_size=16, shuffle=False)
    gallery_loader = DataLoader(PillDataset(df_gallery, transform=test_transform), batch_size=16, shuffle=False)

    # 4. Trích xuất đặc trưng Ensemble
    print("\n--- Trích xuất tập Query (Hold-out) ---")
    q_embs, q_labels, q_instances = extract_ensemble_features(models_list, query_loader, device)
    
    print("\n--- Trích xuất tập Gallery (Reference) ---")
    # Gallery thường chỉ dùng 1 model mạnh nhất hoặc ensemble tùy bạn, ở đây dùng ensemble cho đồng bộ
    g_embs, g_labels, _ = extract_ensemble_features(models_list, gallery_loader, device)

    # 5. Tính toán Similarity & Max-Matching
    unique_g_labels = torch.unique(g_labels)
    sim_matrix = calculate_cosine_similarity(q_embs, g_embs)
    
    # Rút gọn ma trận về (Query x Unique_Labels) bằng Max-matching
    max_sims = torch.zeros((len(q_labels), len(unique_g_labels)))
    for i, lb in enumerate(tqdm(unique_g_labels, desc="📊 Max-Matching")):
        mask = (g_labels == lb)
        max_sims[:, i] = torch.max(sim_matrix[:, mask], dim=1)[0]

    # 6. Đánh giá
    # A. Single-side
    res_s = evaluate_retrieval_metrics(max_sims, q_labels, unique_g_labels)
    
    # B. Both-side (Gộp theo Instance viên thuốc)
    unique_pills = torch.unique(q_instances)
    both_sims, both_labels = [], []
    for p_id in unique_pills:
        idx = (q_instances == p_id)
        both_sims.append(torch.max(max_sims[idx], dim=0)[0])
        both_labels.append(q_labels[idx][0])
    
    res_b = evaluate_retrieval_metrics(torch.stack(both_sims), torch.tensor(both_labels), unique_g_labels)

    # 7. Xuất kết quả
    print("\n" + "⭐" * 30)
    print(f" KẾT QUẢ FINAL ENSEMBLE ({len(models_list)} FOLDS)")
    print(f" Backbone: {args.backbone.upper()}")
    print("⭐" * 30)
    print(f"{'Metric':<15} | {'Single-side':<15} | {'Both-side (MAX)':<15}")
    print("-" * 50)
    for m in ['mAP', 'Rank-1', 'Rank-5']:
        print(f"{m:<15} | {res_s[m]:<15.4f} | {res_b[m]:<15.4f}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, required=True, help="resnet50, convnext_base, hoặc efficientnet_v2_s")
    parser.add_argument('--weights_dir', type=str, default='weights/', help="Thư mục chứa các file .pth")
    args = parser.parse_args()
    main(args)