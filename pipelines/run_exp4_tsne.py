import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các module hiện có của bạn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data

# ==========================================
# ⚙️ CẤU HÌNH THỰC NGHIỆM 4: t-SNE VISUALIZATION
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

NUM_CLASSES_TO_VISUALIZE = 10
IMAGE_SIZE = 384
BATCH_SIZE = 64

def extract_embeddings(model, dataloader, device):
    """Hàm chạy model để lấy vector norm_embedding 512 chiều"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, sub_labels, _, _ in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            # Hàm forward của bạn trả về: logits_sce, logits_csce, norm_embedding
            _, _, norm_emb = model(imgs, labels=sub_labels.to(device))
            all_embeddings.append(norm_emb.cpu().numpy())
            all_labels.append(sub_labels.numpy())
            
    return np.vstack(all_embeddings), np.concatenate(all_labels)

def main():
    print("="*60)
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM 4: t-SNE EMBEDDING VISUALIZATION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data và Chọn ra 10 class phổ biến nhất để vẽ cho đẹp
    print("📦 Đang chuẩn bị dữ liệu...")
    df_all = load_epill_full_data()
    num_total_classes = df_all['sub_label_idx'].nunique()
    
    # Gom cả ảnh Train và Gallery để thấy được độ phân tán
    df_eval = df_all[(df_all['fold'] == 0) | (df_all['is_ref'] == 1)].reset_index(drop=True)
    
    # Tìm 10 class (sub_label_idx) có nhiều ảnh nhất để vẽ
    top_10_classes = df_eval['sub_label_idx'].value_counts().nlargest(NUM_CLASSES_TO_VISUALIZE).index.tolist()
    df_subset = df_eval[df_eval['sub_label_idx'].isin(top_10_classes)].reset_index(drop=True)
    
    print(f"✅ Đã chọn 10 classes: {top_10_classes} (Tổng cộng {len(df_subset)} ảnh)")

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    subset_loader = DataLoader(
        PillDataset(df_subset, transform=val_transform), 
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Dictionary lưu trữ tọa độ 2D của từng model
    tsne_results = {}
    labels_arr = None

    # 2. Vòng lặp trích xuất và giảm chiều (t-SNE) cho từng model
    for model_cfg in MODELS_TO_TEST:
        title = model_cfg["title"]
        print(f"\n⏳ Đang xử lý: {title}")
        
        if not os.path.exists(model_cfg["weight_path"]):
            print(f"❌ KHÔNG TÌM THẤY WEIGHT: {model_cfg['weight_path']}")
            continue

        # Load model
        model = PillRetrievalModel(num_classes=num_total_classes, backbone_type=model_cfg["backbone"], pooling_type='gem').to(device)
        model.load_state_dict(torch.load(model_cfg["weight_path"], map_location=device))
        
        # Lấy vector 512D
        embeddings, labels = extract_embeddings(model, subset_loader, device)
        labels_arr = labels # Nhãn của tất cả model đều giống nhau vì dùng chung 1 dataloader
        
        # Chạy t-SNE giảm từ 512D xuống 2D
        print(f"   📉 Đang chạy t-SNE (giảm chiều 512D -> 2D)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        embeddings_2d = tsne.fit_transform(embeddings)
        
        tsne_results[title] = embeddings_2d
        
        # Dọn dẹp VRAM
        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 🎨 3. VẼ BIỂU ĐỒ (VISUALIZATION)
    # ==========================================
    print("\n🎨 Đang vẽ biểu đồ...")
    os.makedirs("reports_kd", exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    palette = sns.color_palette("tab10", NUM_CLASSES_TO_VISUALIZE) # Lấy 10 màu phân biệt rõ ràng

    for idx, (title, emb_2d) in enumerate(tsne_results.items()):
        ax = axes[idx]
        # Vẽ các chấm bi (scatter)
        sns.scatterplot(
            x=emb_2d[:, 0], y=emb_2d[:, 1],
            hue=labels_arr,
            palette=palette,
            s=60, alpha=0.8,
            legend=False, # Tắt legend ở từng ô để tránh rối mắt
            ax=ax
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([]) # Tắt số trên trục toạ độ cho đẹp
        
    plt.tight_layout()
    
    # Lưu ảnh độ phân giải cao (chuẩn 300 dpi cho báo cáo IEEE)
    save_path = "reports_kd/tsne_ablation.jpg"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("="*60)
    print(f"🎉 HOÀN TẤT! Ảnh t-SNE đã được lưu tại: {save_path}")
    print("="*60)

if __name__ == "__main__":
    main()