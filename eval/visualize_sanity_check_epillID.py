import os
import sys
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data

# Tùy chỉnh tham số
FOLD_T0_EVALUATE = 0
NUM_CLASSES = 9804
root_dataset_path = 'data/raw/ePillID/classification_data'

def get_pil_image(df, idx):
    """Hàm hỗ trợ đọc ảnh từ DataFrame của ePillID"""
    # Xử lý linh hoạt tên cột đường dẫn (có thể là 'img_path' hoặc 'image_path')
    path_col = 'img_path' if 'img_path' in df.columns else 'image_path'
    img_path = os.path.join(root_dataset_path, df.iloc[idx][path_col])
    return Image.open(img_path).convert('RGB')

def main():
    # 1. Cấu hình Device
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        DEVICE = torch.device('mps')
    else: 
        DEVICE = torch.device('cpu')
        
    MODEL_PATH = "weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth" # Thay bằng model bạn muốn test
    OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'epillid_eval')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Tải và phân chia Dữ liệu ePillID (Chỉ dùng Fold 0)
    print("📦 Đang tải dữ liệu ePillID...")
    df_all = load_epill_full_data()
    
    # Lọc Gallery (is_ref = 1) và Query (is_ref = 0 thuộc Fold 0)
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val_query = df_all[(df_all['fold'] == FOLD_T0_EVALUATE) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    # Gộp chung để đưa vào DataLoader
    df_eval = pd.concat([df_val_query, df_ref_gallery]).reset_index(drop=True)
    
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Shuffle=False để Index của DataLoader khớp chính xác 100% với df_eval
    dataset = PillDataset(df_eval, transform=val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. Tải Mô hình
    print("\n🚀 Đang trích xuất đặc trưng với Model trên tập ePillID...")
    model = PillRetrievalModel(num_classes=NUM_CLASSES, backbone_type='resnet18', pooling_type='gem').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 4. Trích xuất Đặc trưng (Feature Extraction)
    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            # Lấy ảnh từ batch (Phụ thuộc vào output của PillDataset, thường element 0 là imgs)
            imgs = batch[0].to(DEVICE)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                feats = model(imgs)
            all_feats.append(feats.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    
    # Lấy nhãn is_ref trực tiếp từ DataFrame vì ta set shuffle=False
    all_is_ref = torch.tensor(df_eval['is_ref'].values)

    # Tách Features
    g_feats = all_feats[all_is_ref == 1]
    q_feats = all_feats[all_is_ref == 0]
    
    # Lấy Index gốc từ DataFrame df_eval để phục hồi ảnh PIL
    g_indices = df_eval.index[df_eval['is_ref'] == 1].tolist()
    q_indices = df_eval.index[df_eval['is_ref'] == 0].tolist()

    # Tính ma trận độ tương đồng
    S = torch.mm(q_feats, g_feats.T)

    # 5. Lấy mẫu Ngẫu nhiên và Trực quan hóa (Sanity Check)
    random.seed(42) # Giữ seed để biểu đồ luôn giống nhau qua các lần chạy
    selected_q_idx_in_matrix = random.sample(range(len(q_feats)), min(10, len(q_feats)))

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle(f"Visual Sanity Check: Query vs Top-1 Gallery (ePillID - Fold {FOLD_T0_EVALUATE})", fontsize=20, fontweight='bold', y=0.98)

    for i, q_mat_idx in enumerate(selected_q_idx_in_matrix):
        # Thông tin Query
        global_q_idx = q_indices[q_mat_idx]
        q_lbl = df_eval.iloc[global_q_idx]['label_idx'] # ePillID dùng label_idx
        img_q = get_pil_image(df_eval, global_q_idx)
        
        # Tìm Top-1 Gallery
        top1_g_mat_idx = torch.argmax(S[q_mat_idx]).item()
        global_g_idx = g_indices[top1_g_mat_idx]
        g_lbl = df_eval.iloc[global_g_idx]['label_idx']
        img_g = get_pil_image(df_eval, global_g_idx)
        
        is_correct = (q_lbl == g_lbl)
        color = 'green' if is_correct else 'red'
        sim_score = S[q_mat_idx][top1_g_mat_idx].item()
        
        row, col_q = i // 2, (i % 2) * 2
        
        # Vẽ Query
        axes[row, col_q].imshow(img_q)
        axes[row, col_q].set_title(f"QUERY\nLabel_idx: {q_lbl}", fontweight='bold')
        axes[row, col_q].axis('off')
        
        # Vẽ Gallery
        ax_g = axes[row, col_q + 1]
        ax_g.imshow(img_g)
        ax_g.set_title(f"TOP-1\nLabel_idx: {g_lbl} | Sim: {sim_score:.2f}", color=color, fontweight='bold')
        for spine in ax_g.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax_g.set_xticks([]); ax_g.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, 'sanity_check_visual_epillid.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu ảnh Visual Sanity Check tại: {out_path}")

if __name__ == "__main__":
    main()