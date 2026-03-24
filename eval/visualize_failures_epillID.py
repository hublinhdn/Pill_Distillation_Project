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

# Import module nội bộ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data

# Tùy chỉnh tham số
FOLD_T0_EVALUATE = 0
NUM_CLASSES = 9804
OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'epillid_eval')

def get_pil_image(df, idx):
    """Hàm hỗ trợ đọc ảnh từ DataFrame kết hợp với root_path của ePillID"""
    root_path = 'data/raw/ePillID/classification_data'
    img_path = os.path.join(root_path, df.iloc[idx]['image_path'])
    return Image.open(img_path).convert('RGB')

def get_image_name(df, idx):
    """Lấy tên file ảnh gốc (để in lên biểu đồ)"""
    return df.iloc[idx]['image_path']

def do_visualize_plan(name='Student KD', backbone='resnet18', weight=''):
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        device_type = 'cuda'
    elif torch.backends.mps.is_available(): 
        DEVICE = torch.device('mps')
        device_type = 'cpu' # autocast mps chưa ổn định, fallback về cpu
    else: 
        DEVICE = torch.device('cpu')
        device_type = 'cpu'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tải và phân chia Dữ liệu ePillID (Chỉ dùng Fold 0)
    print("📦 Đang tải dữ liệu ePillID...")
    df_all = load_epill_full_data()
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val_query = df_all[(df_all['fold'] == FOLD_T0_EVALUATE) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    if len(df_val_query) == 0 or len(df_ref_gallery) == 0:
        print("❌ Lỗi: Không tìm thấy dữ liệu Query hoặc Gallery cho Fold 0. Dừng chương trình.")
        sys.exit(1)
        
    df_eval = pd.concat([df_val_query, df_ref_gallery]).reset_index(drop=True)
        
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 2. Khởi tạo Dataset chuẩn
    dataset = PillDataset(df_eval, transform=val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. Load Mô hình
    print(f"\n🚀 Đang trích xuất đặc trưng với {name}...")
    model = PillRetrievalModel(num_classes=NUM_CLASSES, backbone_type=backbone, pooling_type='gem').to(DEVICE)
    model.load_state_dict(torch.load(weight, map_location=DEVICE, weights_only=True))
    model.eval()

    all_feats = []
    with torch.no_grad():
        # CẬP NHẬT LUỒNG DỮ LIỆU TỪ PillDataset: (image, sub_label, label, is_ref)
        for imgs, sub_labels, labels, is_refs in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast(device_type):
                feats = model(imgs) # Model trả về Tensor duy nhất khi label=None
            all_feats.append(feats.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    
    # Lấy nhãn is_ref trực tiếp từ DataFrame (Do đặt shuffle=False)
    all_is_ref = torch.tensor(df_eval['is_ref'].values)

    g_feats = all_feats[all_is_ref == 1]
    q_feats = all_feats[all_is_ref == 0]
    
    g_indices = df_eval.index[df_eval['is_ref'] == 1].tolist()
    q_indices = df_eval.index[df_eval['is_ref'] == 0].tolist()

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
        
        # So sánh nhãn (label_idx) giữa Query và Gallery
        q_lbl = df_eval.iloc[global_q_idx]['label_idx']
        g_lbl = df_eval.iloc[global_g_idx]['label_idx']
        
        if q_lbl != g_lbl:
            sim_score = S[q_mat_idx][top1_g_mat_idx].item()
            error_cases.append((global_q_idx, global_g_idx, sim_score, q_lbl, g_lbl))

    print(f"⚠️ Tìm thấy {len(error_cases)} ca nhận diện sai trên tổng số {len(q_feats)} Queries.")
    
    if len(error_cases) == 0:
        print("🎉 Chúc mừng! Mô hình không sai ca nào. Không có gì để vẽ.")
        return len(error_cases)

    # Vẽ ngẫu nhiên tối đa 10 lỗi
    random.seed(42)
    num_plots = min(10, len(error_cases))
    selected_errors = random.sample(error_cases, num_plots)

    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    fig.suptitle(f"Analysis: {name} on (ePillID - Fold {FOLD_T0_EVALUATE}) - Failures: {len(error_cases)}", fontsize=20, fontweight='bold', y=0.98, color='red')

    if rows == 1: axes = axes.reshape(1, -1)

    for i, (q_idx, g_idx, sim, q_lbl, g_lbl) in enumerate(selected_errors):
        img_q = get_pil_image(df_eval, q_idx)
        img_g = get_pil_image(df_eval, g_idx)
        img_name_q = get_image_name(df_eval, q_idx)
        img_name_g = get_image_name(df_eval, g_idx)
        
        row, col_q = i // 2, (i % 2) * 2
        
        # Vẽ Query
        axes[row, col_q].imshow(img_q)
        axes[row, col_q].set_title(f"QUERY (Thực tế)\nL_idx: {q_lbl}\n{img_name_q[:18]}", fontweight='bold')
        axes[row, col_q].axis('off')
        
        # Vẽ Gallery đoán sai
        ax_g = axes[row, col_q + 1]
        ax_g.imshow(img_g)
        ax_g.set_title(f"TOP-1 (Đoán sai)\nL_idx: {g_lbl} | Sim: {sim:.2f}\n{img_name_g[:18]}", color='red', fontweight='bold')
        for spine in ax_g.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(4)
        ax_g.set_xticks([]); ax_g.set_yticks([])

    for j in range(num_plots * 2, rows * 4):
        axes[j // 4, j % 4].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_name = name.replace(" ", "_").lower()
    out_img_path = os.path.join(OUTPUT_DIR, f'{safe_name}_epillid_failures_visual.png')
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close() # Giải phóng bộ nhớ RAM
    print(f"✅ Đã lưu ảnh Phân tích lỗi tại: {out_img_path}")
    return len(error_cases)

def generate_report(results, output_dir, output_img_path='epillid_failure_comparison.png', output_txt_report='epillid_failure_comparison_report.txt'):
    os.makedirs(output_dir, exist_ok=True)
    names = [r['name'] for r in results]
    error_count = [r['error_count'] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, error_count, color='lightcoral', edgecolor='darkred')

    plt.title(f'Failure Comparison (ePillID - Fold {FOLD_T0_EVALUATE})', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Number of Failures', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_img_path), dpi=300)

    with open(os.path.join(output_dir, output_txt_report), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\nBÁO CÁO SỐ LƯỢNG LỖI TRÊN EPILLID (FOLD 0)\n" + "="*60 + "\n")
        f.write(f"{'Tên':<25} | {'Mô hình':<15} | {'Số lỗi':<10}\n" + "-" * 55 + "\n")
        for r in results:
            f.write(f"{r['name']:<25} | {r['backbone']:<15} | {r['error_count']}\n")


if __name__ == "__main__":
    # Cập nhật danh sách Models của bạn tại đây
    visualize_plan = [
        {
            "name": "Student KD",
            "backbone":"resnet18",
            "weight":"weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth"
        },
        {
            "name": "Student Baseline",
            "backbone":"resnet18",
            "weight":"weights/best_resnet18_gem_fold0.pth"
        }
    ]
    
    results = []
    for plan in visualize_plan:
        error_count = do_visualize_plan(plan["name"], plan["backbone"], plan["weight"])
        results.append({'name': plan["name"],'backbone':plan["backbone"], 'error_count': error_count})
        torch.cuda.empty_cache() # Dọn dẹp VRAM
    
    generate_report(results, OUTPUT_DIR)