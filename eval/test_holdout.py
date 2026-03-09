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
    """
    Trích xuất embedding từ 4 models, kết hợp Flip-Matching và Ensemble trung bình.
    """
    for m in models: m.eval()
        
    all_embs = []
    all_labels = []
    all_is_refs = []
    all_paths = []

    with torch.no_grad():
        for imgs, labels, is_refs in tqdm(dataloader, desc="🚀 Đang trích xuất đặc trưng Ensemble"):
            imgs = imgs.to(device)
            imgs_flip = torch.flip(imgs, dims=[3])
            
            batch_embs = []
            for model in models:
                with torch.amp.autocast('cuda'):
                    # Forward ảnh gốc
                    emb_orig = model(imgs)
                    if isinstance(emb_orig, tuple): emb_orig = emb_orig[-1]
                    # Forward ảnh lật
                    emb_flip = model(imgs_flip)
                    if isinstance(emb_flip, tuple): emb_flip = emb_flip[-1]
                    
                    # Trung bình Flip của model này
                    batch_embs.append((emb_orig + emb_flip) / 2.0)
            
            # Trung bình cộng tất cả các Model (Ensemble)
            ensemble_emb = torch.stack(batch_embs).mean(dim=0)
            ensemble_emb = F.normalize(ensemble_emb, p=2, dim=1)
            
            all_embs.append(ensemble_emb.cpu())
            all_labels.append(labels)
            all_is_refs.append(is_refs)

    return torch.cat(all_embs), torch.cat(all_labels).numpy(), torch.cat(all_is_refs).numpy()

def visualize_failures(q_embs, q_labels, g_embs, g_labels, df_query, df_gallery, top_k=5, num_samples=5):
    """
    Vẽ các trường hợp nhận diện sai để phân tích giới hạn mô hình.
    """
    os.makedirs('analysis', exist_ok=True)
    root_img_path = 'data/raw/ePillID/classification_data'
    
    # Tính ma trận tương đồng giữa Query và Gallery
    sim_matrix = torch.mm(q_embs, g_embs.t())
    
    failure_indices = []
    for i in range(len(q_labels)):
        scores = sim_matrix[i].numpy()
        best_match_idx = np.argmax(scores)
        if g_labels[best_match_idx] != q_labels[i]:
            failure_indices.append(i)

    if not failure_indices:
        print("🎉 Tuyệt vời! Không tìm thấy lỗi nào trong số các mẫu kiểm tra.")
        return

    num_samples = min(num_samples, len(failure_indices))
    fig, axes = plt.subplots(num_samples, top_k + 1, figsize=(20, 4 * num_samples))
    
    for idx, q_idx in enumerate(failure_indices[:num_samples]):
        # Hiển thị ảnh Query
        q_row = df_query.iloc[q_idx]
        q_img = Image.open(os.path.join(root_img_path, q_row['image_path']))
        axes[idx, 0].imshow(q_img)
        axes[idx, 0].set_title(f"QUERY\nID: {q_labels[q_idx]}", color='blue', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Lấy Top-K Gallery tương ứng
        scores = sim_matrix[q_idx].numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        for k in range(top_k):
            g_idx = top_indices[k]
            g_row = df_gallery.iloc[g_idx]
            g_img = Image.open(os.path.join(root_img_path, g_row['image_path']))
            
            is_correct = (g_labels[g_idx] == q_labels[q_idx])
            color = 'green' if is_correct else 'red'
            
            axes[idx, k+1].imshow(g_img)
            axes[idx, k+1].set_title(f"Rank-{k+1}\nID: {g_labels[g_idx]}", color=color, fontsize=9)
            axes[idx, k+1].axis('off')

    plt.tight_layout()
    plt.savefig('analysis/failure_analysis.png')
    print("📸 Đã lưu ảnh phân tích lỗi tại: analysis/failure_analysis.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_all = load_epill_full_data()
    num_classes = int(df_all['label_idx'].max() + 1)

    # 1. Chuẩn bị tập Hold-out (Fold 4)
    df_query = df_all[(df_all['fold'] == 4) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    df_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    test_df = pd.concat([df_query, df_gallery]).reset_index(drop=True)
    
    print(f"📊 Hold-out: {len(df_query)} queries | Gallery: {len(df_gallery)} images")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_loader = DataLoader(PillDataset(test_df, test_transform), batch_size=16, shuffle=False, num_workers=4)

    # 2. Load Models
    models = []
    for i in range(4):
        path = f'weights/teacher_benchmark_f{i}.pth'
        if os.path.exists(path):
            m = PillTeacher(num_classes=num_classes).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            models.append(m)
            print(f"✅ Đã load Fold {i}")

    if not models: return print("❌ Không thấy file weights nào!")

    # 3. Trích xuất đặc trưng
    embs, labels, is_refs = get_ensemble_embeddings(models, test_loader, device)
    
    q_embs = embs[is_refs == 0]
    q_labels = labels[is_refs == 0]
    g_embs = embs[is_refs == 1]
    g_labels = labels[is_refs == 1]

    # 4. Tính Max-Matching (Mỗi nhãn lấy điểm cao nhất từ các ảnh Reference của nó)
    unique_g_labels = np.unique(g_labels)
    max_sims = torch.zeros((len(q_labels), len(unique_g_labels)))
    
    for j, label in enumerate(tqdm(unique_g_labels, desc="🔍 Đang so khớp Max-Matching")):
        label_mask = (g_labels == label)
        label_embs = g_embs[label_mask]
        sims = torch.mm(q_embs, label_embs.t())
        max_sims[:, j] = torch.max(sims, dim=1)[0]

    # 5. Tính toán Metrics
    aps = []
    rank1 = 0
    for i in range(len(q_labels)):
        scores = max_sims[i].numpy()
        sorted_labels = unique_g_labels[np.argsort(scores)[::-1]]
        
        if sorted_labels[0] == q_labels[i]: rank1 += 1
        
        hit_rank = np.where(sorted_labels == q_labels[i])[0][0] + 1
        aps.append(1.0 / hit_rank)

    # 6. Thống kê và Visualization
    mAP = np.mean(aps)
    r1_acc = rank1 / len(q_labels)
    
    print("\n" + "="*40)
    print(f"🏆 KẾT QUẢ CUỐI CÙNG TRÊN FOLD 4")
    print(f"🔹 mAP: {mAP:.4f}")
    print(f"🔹 Rank-1: {r1_acc:.4f}")
    print("="*40)

    # Lưu thống kê chi tiết loại khó nhất
    results_df = pd.DataFrame({'label': q_labels, 'ap': aps})
    hard_classes = results_df.groupby('label')['ap'].mean().sort_values().head(10)
    print("\n🧐 Top 10 nhãn khó nhận diện nhất (AP thấp nhất):")
    print(hard_classes)

    visualize_failures(q_embs, q_labels, g_embs, g_labels, df_query, df_gallery)

if __name__ == '__main__':
    main()