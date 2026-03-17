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
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_ogyei import build_ogyei_df_strict_split, OGYEICropDataset, LetterboxResize

USE_TRAIN_AS_GALLERY = True # use valid (6) or train(28) as gallery for query
gallery_split = 'train' if USE_TRAIN_AS_GALLERY else 'valid' 
# ⚠️ HÃY ĐẢM BẢO ĐƯỜNG DẪN NÀY CHÍNH XÁC VỚI SERVER CỦA BẠN
OGYEI_ROOT = os.path.join('data/raw/OGYEIv2/ogyeiv2', 'ogyeiv2')
OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'error_analysis')
CLASS_TRAINNING = 9804 # train by ePillID as 2 side 9804

def do_visualize_plan(name='student_kd_resnet18', backbone='resnet18', weight='weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth'):
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        device_type = 'cuda'
    elif torch.backends.mps.is_available(): # Dành cho MacBook chip M1/M2/M3
        DEVICE = torch.device('mps')
        device_type = 'mps'
    else: # Dành cho MacBook chip Intel
        DEVICE = torch.device('cpu')
        device_type = 'cpu'

    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Student KD
    NAME_TEST = name
    BACKBONE = backbone
    MODEL_PATH = weight

    df = build_ogyei_df_strict_split(OGYEI_ROOT, gallery_split = gallery_split)
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

    print(f"\n🚀 Đang trích xuất đặc trưng với {NAME_TEST}...")
    model = PillRetrievalModel(num_classes=CLASS_TRAINNING, backbone_type=BACKBONE, pooling_type='gem').to(DEVICE)
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
        return len(error_cases)
        # sys.exit(0)

    # Chọn ngẫu nhiên tối đa 10 lỗi để vẽ
    random.seed(42)
    num_plots = min(10, len(error_cases))
    selected_errors = random.sample(error_cases, num_plots)

    # Cấu hình lưới vẽ (Grid)
    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    fig.suptitle(f"Analysis: {NAME_TEST} on (OGYEIv2) with gallery:{gallery_split} - Failure: {len(error_cases)}", fontsize=20, fontweight='bold', y=0.98, color='red')

    # Đảm bảo axes luôn là mảng 2 chiều để dễ index
    if rows == 1: axes = axes.reshape(1, -1)

    for i, (q_idx, g_idx, sim, q_lbl, g_lbl) in enumerate(selected_errors):
        img_q = dataset.get_pil_image(q_idx)
        img_g = dataset.get_pil_image(g_idx)
        img_name_g = dataset.df.iloc[g_idx]['img_name']

        img_name_q = dataset.df.iloc[q_idx]['img_name']
        
        row, col_q = i // 2, (i % 2) * 2
        
        # Vẽ Query (Sự thật)
        axes[row, col_q].imshow(img_q)
        axes[row, col_q].set_title(f"QUERY (Thực tế)\nClass: {q_lbl}\nFile {img_name_q}", fontweight='bold')
        axes[row, col_q].axis('off')
        
        # Vẽ Gallery (Mô hình đoán sai)
        ax_g = axes[row, col_q + 1]
        ax_g.imshow(img_g)
        ax_g.set_title(f"TOP-1 (Đoán sai)\nClass: {g_lbl} | Sim: {sim:.2f}| \nFile: {img_name_g}", color='red', fontweight='bold')
        for spine in ax_g.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(4)
        ax_g.set_xticks([]); ax_g.set_yticks([])

    # Xóa các ô trống nếu số lượng ảnh lẻ
    for j in range(num_plots * 2, rows * 4):
        axes[j // 4, j % 4].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_img_path = os.path.join(OUTPUT_DIR, f'{NAME_TEST}_ogyei_failures_visual.png')
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu ảnh Phân tích lỗi tại: {out_img_path}")
    return len(error_cases)

def generate_report(results, output_dir, output_img_path='failure_comparison.png', output_txt_report='failure_comparison_report.txt'):
    os.makedirs(output_dir, exist_ok=True)
    names = [r['name'] for r in results]
    backbone = [r['backbone'] for r in results]
    error_count = [r['error_count'] for r in results]

    # --- Vẽ biểu đồ ---
    plt.figure(figsize=(8, 5)) # Chỉnh kích thước khung hình
    plt.bar(names, error_count, color='skyblue', edgecolor='navy')

    # 3. Thêm chi tiết (Tiêu đề, nhãn trục)
    plt.title('Failure Comparison', fontsize=14)
    plt.xlabel('Num of failure', fontsize=12)
    plt.ylabel('Backbone', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_img_path), dpi=300)

    # --- Xuất báo cáo text ---
    with open(os.path.join(output_dir, output_txt_report), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\nBÁO CÁO LỖI OGYEIv2 \n" + "="*60 + "\n")
        f.write(f"{'Tên':<25} | {'Mô hình':<12} | {'Số lỗi':<10}\n" + "-" * 55 + "\n")
        for r in results:
            f.write(f"{r['name']:<25} | {r['backbone']:<25} % | {r['error_count']}\n")


if __name__ == "__main__":
    visualize_plan = [
        {
            "name": "student_kd_resnet18",
            "backbone":"resnet18",
            "weight":"weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth"
        },
        {
            "name": "student_baseline_resnet18",
            "backbone":"resnet18",
            "weight":"weights/best_resnet18_gem_fold0.pth"
        },
        {
            "name": "teacher_convnext_base",
            "backbone":"convnext_base",
            "weight":"weights/best_teacher_convnext_base_fold0.pth"
        }
    ]
    results = []
    for plan in visualize_plan:
        error_count = do_visualize_plan(plan["name"], plan["backbone"], plan["weight"])
        results.append({'name': plan["name"],'backbone':plan["backbone"], 'error_count': error_count})
    
    output_img_path=f'failure_comparison.png'
    output_txt_report=f'failure_comparison_report.txt'
    generate_report(results, OUTPUT_DIR, output_img_path=output_img_path, output_txt_report=output_txt_report)
