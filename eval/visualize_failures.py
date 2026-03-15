import os
import sys
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillRetrievalModel
from utils.dataset_ogyei import build_ogyei_df_strict_split, OGYEICropDataset, LetterboxResize

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ⚠️ HÃY ĐẢM BẢO ĐƯỜNG DẪN NÀY CHÍNH XÁC VỚI SERVER CỦA BẠN
    OGYEI_ROOT = os.path.join('data/raw/OGYEIv2/ogyeiv2', 'ogyeiv2')
    # BACKBONE = "resnet18"
    # MODEL_PATH = "weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth"
    BACKBONE = "convnext_base"
    MODEL_PATH = "weights/best_teacher_convnext_base_fold0.pth"
    OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'error_analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = build_ogyei_df_strict_split(OGYEI_ROOT)
    if len(df) == 0:
        print("❌ Lỗi: Không tìm thấy dữ liệu. Dừng chương trình.")
        sys.exit(1)
        
    transform = transforms.Compose([
        LetterboxResize(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = OGYEICropDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("\n🚀 Đang trích xuất đặc trưng với Student KD...")
    model = PillRetrievalModel(num_classes=9804, backbone_type=BACKBONE, pooling_type='gem').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    all_feats, all_is_ref = [], []
    with torch.no_grad():
        for imgs, _, _, is_ref, _ in tqdm(loader, desc="Extracting"):
            feats = model(imgs.to(DEVICE))
            all_feats.append((feats[-1] if isinstance(feats, tuple) else feats).cpu())
            all_is_ref.append(is_ref.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    all_is_ref = torch.cat(all_is_ref, dim=0)

    g_feats = all_feats[all_is_ref == 1]
    q_feats = all_feats[all_is_ref == 0]
    
    g_indices = df.index[df['is_ref'] == 1].tolist()
    q_indices = df.index[df['is_ref'] == 0].tolist()

    S = torch.mm(q_feats, g_feats.T)

    # ==========================================
    # LỌC RA CÁC TRƯỜNG HỢP NHẬN DẠNG SAI
    # ==========================================
    error_cases = []
    print("🧮 Đang truy vết các trường hợp lỗi...")
    for q_mat_idx in range(len(q_feats)):
        top1_g_mat_idx = torch.argmax(S[q_mat_idx]).item()
        
        global_q_idx = q_indices[q_mat_idx]
        global_g_idx = g_indices[top1_g_mat_idx]
        
        q_lbl_name = df.iloc[global_q_idx]['label_name']
        g_lbl_name = df.iloc[global_g_idx]['label_name']
        
        # NẾU NHÃN KHÁC NHAU => LƯU LẠI
        if q_lbl_name != g_lbl_name:
            sim_score = S[q_mat_idx][top1_g_mat_idx].item()
            error_cases.append((global_q_idx, global_g_idx, sim_score, q_lbl_name, g_lbl_name))

    print(f"⚠️ Tìm thấy {len(error_cases)} ca nhận diện sai trên tổng số {len(q_feats)} Queries.")
    
    if len(error_cases) == 0:
        print("🎉 Chúc mừng! Mô hình không sai ca nào. Không có gì để vẽ.")
        sys.exit(0)

    # Chọn ngẫu nhiên tối đa 10 lỗi để vẽ
    random.seed(42)
    num_plots = min(10, len(error_cases))
    selected_errors = random.sample(error_cases, num_plots)

    # Cấu hình lưới vẽ (Grid)
    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    fig.suptitle("Failure Analysis: Teacher Bị Nhầm Lẫn (OGYEIv2)", fontsize=20, fontweight='bold', y=0.98, color='red')

    # Đảm bảo axes luôn là mảng 2 chiều để dễ index
    if rows == 1: axes = axes.reshape(1, -1)

    for i, (q_idx, g_idx, sim, q_lbl, g_lbl) in enumerate(selected_errors):
        img_q = dataset.get_pil_image(q_idx)
        img_g = dataset.get_pil_image(g_idx)
        
        row, col_q = i // 2, (i % 2) * 2
        
        # Vẽ Query (Sự thật)
        axes[row, col_q].imshow(img_q)
        axes[row, col_q].set_title(f"QUERY (Thực tế)\nClass: {q_lbl}", fontweight='bold')
        axes[row, col_q].axis('off')
        
        # Vẽ Gallery (Mô hình đoán sai)
        ax_g = axes[row, col_q + 1]
        ax_g.imshow(img_g)
        ax_g.set_title(f"TOP-1 (Đoán sai)\nClass: {g_lbl} | Sim: {sim:.2f}", color='red', fontweight='bold')
        for spine in ax_g.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(4)
        ax_g.set_xticks([]); ax_g.set_yticks([])

    # Xóa các ô trống nếu số lượng ảnh lẻ
    for j in range(num_plots * 2, rows * 4):
        axes[j // 4, j % 4].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_img_path = os.path.join(OUTPUT_DIR, f'{BACKBONE}_ogyei_failures_visual.png')
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu ảnh Phân tích lỗi tại: {out_img_path}")

if __name__ == "__main__":
    main()