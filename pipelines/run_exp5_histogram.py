import torch
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các module hiện có của bạn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data

# ==========================================
# ⚙️ CẤU HÌNH THỰC NGHIỆM 5: DISTANCE HISTOGRAM
# ==========================================
MODELS_TO_TEST = [
    {
        "title": "Teacher (ResNeSt101e)",
        "backbone": "resnest101e_timm",
        "weight_path": "weights/phase2/best_resnest101e_timm_gem_fold0.pth"
    },
    {
        "title": "Student Baseline (ResNet18)",
        "backbone": "resnet18_tv",
        "weight_path": "weights/phase2/best_resnet18_tv_gem_fold0.pth"
    },
    {
        "title": "KD Student (ResNet18)",
        "backbone": "resnet18_tv",
        "weight_path": "weights/kd_models/best_resnest101e_timm_kd_resnet18_tv_kd_typecosine_fold0.pth"
    }
]

NUM_CLASSES_TO_SAMPLE = 50  # Chọn 50 class để tính toán cho nhanh gọn và đủ thống kê
IMAGE_SIZE = 384
BATCH_SIZE = 64

def extract_embeddings(model, dataloader, device):
    """Trích xuất vector 512D đã được L2-normalized"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, sub_labels, _, _ in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            _, _, norm_emb = model(imgs, labels=sub_labels.to(device))
            all_embeddings.append(norm_emb.cpu().numpy())
            all_labels.append(sub_labels.numpy())
            
    return np.vstack(all_embeddings), np.concatenate(all_labels)

def compute_distances(embeddings, labels):
    """Tính toán khoảng cách Intra-class và Inter-class"""
    # Vì embeddings đã L2-normalized, tích vô hướng (dot product) chính là Cosine Similarity
    # Cosine Distance = 1 - Cosine Similarity
    sim_matrix = np.dot(embeddings, embeddings.T)
    dist_matrix = 1.0 - sim_matrix
    
    # Ép các giá trị siêu nhỏ do sai số float về 0 để tránh nhiễu
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    
    # Tạo ma trận mask cho cùng class và khác class
    n = len(labels)
    label_matrix = np.expand_dims(labels, 1) == np.expand_dims(labels, 0)
    
    # Bỏ đường chéo chính (khoảng cách của 1 ảnh với chính nó luôn = 0)
    np.fill_diagonal(label_matrix, False)
    
    # Lọc lấy danh sách khoảng cách
    intra_distances = dist_matrix[label_matrix]  # Cùng class
    inter_distances = dist_matrix[~label_matrix & ~np.eye(n, dtype=bool)] # Khác class
    
    return intra_distances, inter_distances

def main():
    print("="*60)
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM 5: DISTANCE DISTRIBUTION HISTOGRAM")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print("📦 Đang chuẩn bị dữ liệu...")
    df_all = load_epill_full_data()
    num_total_classes = df_all['sub_label_idx'].nunique()
    
    # Lấy dữ liệu validation/gallery
    df_eval = df_all[(df_all['fold'] == 0) | (df_all['is_ref'] == 1)].reset_index(drop=True)
    
    # Chọn ngẫu nhiên 50 classes có nhiều ảnh để làm thống kê
    top_classes = df_eval['sub_label_idx'].value_counts().nlargest(NUM_CLASSES_TO_SAMPLE).index.tolist()
    df_subset = df_eval[df_eval['sub_label_idx'].isin(top_classes)].reset_index(drop=True)
    print(f"✅ Đã chọn {NUM_CLASSES_TO_SAMPLE} classes (Tổng: {len(df_subset)} ảnh)")

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    subset_loader = DataLoader(
        PillDataset(df_subset, transform=val_transform), 
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Dictionary lưu trữ khoảng cách
    dist_results = {}

    # 2. Trích xuất đặc trưng và tính khoảng cách
    for model_cfg in MODELS_TO_TEST:
        title = model_cfg["title"]
        print(f"\n⏳ Đang xử lý: {title}")
        
        if not os.path.exists(model_cfg["weight_path"]):
            print(f"❌ KHÔNG TÌM THẤY WEIGHT: {model_cfg['weight_path']}")
            continue

        model = PillRetrievalModel(num_classes=num_total_classes, backbone_type=model_cfg["backbone"], pooling_type='gem').to(device)
        model.load_state_dict(torch.load(model_cfg["weight_path"], map_location=device))
        
        embeddings, labels = extract_embeddings(model, subset_loader, device)
        
        print("   📐 Đang tính toán ma trận khoảng cách Pairwise...")
        intra_dist, inter_dist = compute_distances(embeddings, labels)
        
        dist_results[title] = {"intra": intra_dist, "inter": inter_dist}
        
        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 🎨 3. VẼ BIỂU ĐỒ HISTOGRAM
    # ==========================================
    print("\n🎨 Đang vẽ biểu đồ phân phối...")
    os.makedirs("reports_kd", exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for idx, (title, dist_data) in enumerate(dist_results.items()):
        ax = axes[idx]
        
        # Vẽ KDE (Kernel Density Estimate) cho mượt mà thay vì cột cứng nhắc
        sns.kdeplot(dist_data["intra"], fill=True, color="blue", alpha=0.5, label="Intra-class (Same Pill)", ax=ax)
        sns.kdeplot(dist_data["inter"], fill=True, color="red", alpha=0.5, label="Inter-class (Different Pills)", ax=ax)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Cosine Distance")
        if idx == 0:
            ax.set_ylabel("Density")
        if idx == 2:
            ax.legend(loc="upper right", frameon=True)

    plt.xlim(0, 1.5) # Giới hạn trục X từ 0 đến 1.5 để nhìn rõ hình chuông
    plt.tight_layout()
    
    save_path = "reports_kd/distance_histogram.jpg"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("="*60)
    print(f"🎉 HOÀN TẤT! Ảnh Histogram đã được lưu tại: {save_path}")
    print("="*60)

if __name__ == "__main__":
    main()