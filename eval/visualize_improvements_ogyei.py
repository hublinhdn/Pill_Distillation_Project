# Use to draw what is improve from baseline to KD

import os
import sys
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import module nội bộ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_ogyei import build_ogyei_df_strict_split, OGYEICropDataset, LetterboxResize

# ==========================================
# ⚙️ TÙY CHỈNH THÔNG SỐ Ở ĐÂY
# ==========================================
NUM_CLASSES = 9804
USE_TRAIN_AS_GALLERY = True 
gallery_split = 'train' if USE_TRAIN_AS_GALLERY else 'valid' 
OGYEI_ROOT = os.path.join('data/raw/OGYEIv2/ogyeiv2', 'ogyeiv2')
OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'error_analysis')

# Cấu hình 2 mô hình cần so sánh
NAME = 'Resnet18'
BACKBONE = 'resnet18_tv'
BASELINE_WEIGHT = "weights/phase2/best_resnet18_tv_gem_fold0.pth"
KD_WEIGHT = "weights/kd_models/best_resnest101e_timm_kd_resnet18_tv_kd_typecosine_fold0.pth"

# NAME = 'EffB2'
# BACKBONE = 'efficientnet_b2_timm'
# BASELINE_WEIGHT = "weights/phase2/best_efficientnet_b2_timm_gem_fold0.pth"
# KD_WEIGHT = "weights/kd_models/best_efficientnet_b5_timm_kd_efficientnet_b2_timm_kd_typecosine_fold0.pth"


def extract_features(model, dataloader, device):
    """Hàm hỗ trợ trích xuất đặc trưng cho một model"""
    model.eval()
    all_feats, all_is_ref = [], []
    with torch.no_grad():
        for imgs, _, _, is_ref, _ in tqdm(dataloader, desc="Extracting Features"):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                feats = model(imgs) # Output đã được cập nhật trả về 1 tensor
            all_feats.append(feats.cpu())
            all_is_ref.append(is_ref.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    all_is_ref = torch.cat(all_is_ref, dim=0)
    return all_feats, all_is_ref


def main():
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        DEVICE = torch.device('mps')
    else: 
        DEVICE = torch.device('cpu')
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Chuẩn bị Dữ liệu
    print("📦 Đang tải dữ liệu OGYEIv2...")
    df = build_ogyei_df_strict_split(OGYEI_ROOT, gallery_split=gallery_split)
    if len(df) == 0:
        print("❌ Lỗi: Không tìm thấy dữ liệu OGYEIv2.")
        sys.exit(1)
        
    transform = transforms.Compose([
        LetterboxResize(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = OGYEICropDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Khởi tạo Mô hình
    model = PillRetrievalModel(num_classes=NUM_CLASSES, backbone_type=BACKBONE, pooling_type='gem').to(DEVICE)

    # 3. Trích xuất đặc trưng cho Baseline
    print("\n🏃 Bước 1/2: Trích xuất đặc trưng với STUDENT BASELINE...")
    model.load_state_dict(torch.load(BASELINE_WEIGHT, map_location=DEVICE, weights_only=True))
    feats_base, is_ref_base = extract_features(model, loader, DEVICE)

    # 4. Trích xuất đặc trưng cho KD
    print("\n🏃 Bước 2/2: Trích xuất đặc trưng với STUDENT KD...")
    model.load_state_dict(torch.load(KD_WEIGHT, map_location=DEVICE, weights_only=True))
    feats_kd, is_ref_kd = extract_features(model, loader, DEVICE)

    # 5. Phân tách Query và Gallery
    g_indices = df.index[df['is_ref'] == 1].tolist()
    q_indices = df.index[df['is_ref'] == 0].tolist()

    # Tính ma trận Similarity
    S_base = torch.mm(feats_base[is_ref_base == 0], feats_base[is_ref_base == 1].T)
    S_kd = torch.mm(feats_kd[is_ref_kd == 0], feats_kd[is_ref_kd == 1].T)

    # ==========================================
    # 🕵️ LỌC RA CÁC "CA LỘT XÁC" (BASELINE SAI, KD ĐÚNG)
    # ==========================================
    improvements = []
    print("\n🧮 Đang truy vết các trường hợp được KD cải thiện...")
    
    for q_mat_idx in range(len(q_indices)):
        top1_g_mat_idx_base = torch.argmax(S_base[q_mat_idx]).item()
        top1_g_mat_idx_kd = torch.argmax(S_kd[q_mat_idx]).item()
        
        global_q_idx = q_indices[q_mat_idx]
        global_g_idx_base = g_indices[top1_g_mat_idx_base]
        global_g_idx_kd = g_indices[top1_g_mat_idx_kd]
        
        lbl_q = df.iloc[global_q_idx]['label_name']
        lbl_g_base = df.iloc[global_g_idx_base]['label_name']
        lbl_g_kd = df.iloc[global_g_idx_kd]['label_name']
        
        # LOGIC LỌC TỐI THƯỢNG:
        if (lbl_q != lbl_g_base) and (lbl_q == lbl_g_kd):
            sim_base = S_base[q_mat_idx][top1_g_mat_idx_base].item()
            sim_kd = S_kd[q_mat_idx][top1_g_mat_idx_kd].item()
            
            improvements.append({
                'q_idx': global_q_idx,
                'g_base_idx': global_g_idx_base,
                'g_kd_idx': global_g_idx_kd,
                'lbl_q': lbl_q,
                'lbl_g_base': lbl_g_base,
                'lbl_g_kd': lbl_g_kd,
                'sim_base': sim_base,
                'sim_kd': sim_kd
            })

    print(f"🌟 KẾT QUẢ: Nhờ KD, mô hình đã sửa sai thành công {len(improvements)} viên thuốc trên môi trường OGYEIv2!")
    
    if len(improvements) == 0:
        print("Không có ca nào Baseline sai mà KD đúng. Dừng vẽ.")
        return

    # ==========================================
    # 🎨 VẼ BIỂU ĐỒ 3 CỘT (QUERY | BASELINE | KD)
    # ==========================================
    random.seed(42)
    num_plots = min(4, len(improvements)) # Vẽ 4 ca là đẹp nhất cho 1 trang A4
    selected_cases = random.sample(improvements, num_plots)

    fig, axes = plt.subplots(num_plots, 3, figsize=(9, 3 * num_plots))
    # fig.suptitle(f"{NAME} with Knowledge Distillation Improvements on OGYEIv2", fontsize=18, fontweight='bold', y=0.98, color='darkgreen')

    # Đảm bảo axes luôn là mảng 2 chiều
    if num_plots == 1: axes = axes.reshape(1, -1)

    for i, case in enumerate(selected_cases):
        # Lấy ảnh
        img_q = dataset.get_pil_image(case['q_idx'])
        img_base = dataset.get_pil_image(case['g_base_idx'])
        img_kd = dataset.get_pil_image(case['g_kd_idx'])
        
        # Lấy tên file để in cho dễ soi
        name_q = dataset.df.iloc[case['q_idx']]['img_name']
        name_base = dataset.df.iloc[case['g_base_idx']]['img_name']
        name_kd = dataset.df.iloc[case['g_kd_idx']]['img_name']

        # [Cột 1]: Ảnh Query (Thực tế)
        ax_q = axes[i, 0]
        ax_q.imshow(img_q)
        ax_q.set_title(f"[1] QUERY (OGYEIv2)\nClass: {case['lbl_q']}", fontweight='bold')
        ax_q.axis('off')

        # [Cột 2]: Ảnh Baseline đoán sai
        ax_base = axes[i, 1]
        ax_base.imshow(img_base)
        ax_base.set_title(f"[2] {NAME} BASELINE (False)\nClass: {case['lbl_g_base']} | Sim: {case['sim_base']:.2f}.", color='red', fontweight='bold')
        for spine in ax_base.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(4)
        ax_base.set_xticks([]); ax_base.set_yticks([])

        # [Cột 3]: Ảnh KD đoán đúng
        ax_kd = axes[i, 2]
        ax_kd.imshow(img_kd)
        ax_kd.set_title(f"[3] {NAME} KD (True)\nClass: {case['lbl_g_kd']} | Sim: {case['sim_kd']:.2f}", color='green', fontweight='bold')
        for spine in ax_kd.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(6) # Viền dày hơn để nhấn mạnh chiến thắng
        ax_kd.set_xticks([]); ax_kd.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_img_path = os.path.join(OUTPUT_DIR, f'kd_improvements_{BACKBONE}_ogyei.jpg')
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Đã lưu ảnh truy vết tại: {out_img_path}")
    print("👉 Hãy mở ảnh này lên và đưa vào báo cáo để chứng minh KD đã giúp mô hình học được gì nhé!")

if __name__ == "__main__":
    main()