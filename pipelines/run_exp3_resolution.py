import torch
import pandas as pd
import os
import sys
import json
import csv
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

# Import các module hiện có của bạn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

# ==========================================
# ⚙️ CẤU HÌNH THỰC NGHIỆM 3: ĐỘ PHÂN GIẢI
# ==========================================
# ⚠️ QUAN TRỌNG: Hãy đảm bảo đường dẫn tới 3 file weights này là chính xác!
MODELS_TO_TEST = [
    {
        "model_role": "Teacher (ResNeSt101e)",
        "backbone": "resnest101e_timm",
        "weight_path": "weights/phase2/best_resnest101e_timm_gem_fold0.pth"
    },
    {
        "model_role": "Student Baseline (ResNet18)",
        "backbone": "resnet18_tv",
        "weight_path": "weights/phase2/best_resnet18_tv_gem_fold0.pth"
    },
    {
        "model_role": "Student KD (ResNet18)",
        "backbone": "resnet18_tv",
        "weight_path": "weights/kd_models/best_resnest101e_timm_kd_resnet18_tv_kd_typecosine_fold0.pth"
    }
]

RESOLUTIONS = [384, 256, 128]
BATCH_SIZE = 128  # Để batch lớn cho inference nhanh

def main():
    print("="*60)
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM 3: ROBUSTNESS TO RESOLUTION DEGRADATION")
    print("="*60)

    # 1. Setup Device & Data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_all = load_epill_full_data()
    num_classes = df_all['sub_label_idx'].nunique()
    
    fold = 0
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    df_eval_combined = pd.concat([df_val, df_ref_gallery]).reset_index(drop=True)

    results = []

    # 2. Vòng lặp thay đổi kích thước ảnh (Resolution Loop)
    for res in RESOLUTIONS:
        print(f"\n[🖼️ ĐỘ PHÂN GIẢI: {res}x{res}] Đang chuẩn bị DataLoader...")
        
        # 🛡️ Thay đổi Transform ở đây
        val_transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_loader = DataLoader(
            PillDataset(df_eval_combined, transform=val_transform), 
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )

        # 3. Vòng lặp test từng mô hình (Model Loop)
        for model_cfg in MODELS_TO_TEST:
            role = model_cfg["model_role"]
            backbone = model_cfg["backbone"]
            w_path = model_cfg["weight_path"]

            print(f"⏳ Đang đánh giá {role}...")
            
            if not os.path.exists(w_path):
                print(f"❌ KHÔNG TÌM THẤY WEIGHT: {w_path} -> Bỏ qua!")
                continue

            # Khởi tạo model và load weight
            model = PillRetrievalModel(num_classes=num_classes, backbone_type=backbone, pooling_type='gem').to(device)
            model.load_state_dict(torch.load(w_path, map_location=device))
            model.eval()

            # Chạy Evaluator
            with torch.no_grad():
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    val_metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=True)
            
            map_val = val_metrics['mAP']
            r1_val = val_metrics['Rank-1']
            
            results.append({
                "Resolution": f"{res}x{res}",
                "Model_Role": role,
                "mAP": map_val,
                "Rank-1": r1_val
            })
            
            print(f"✅ {role} @ {res}px -> mAP: {map_val:.4f} | Rank-1: {r1_val:.4f}")
            
            # Xóa model khỏi VRAM để nhường chỗ cho model tiếp theo
            del model
            torch.cuda.empty_cache()

    # ==========================================
    # 💾 4. XUẤT KẾT QUẢ RA CSV
    # ==========================================
    os.makedirs("reports_kd", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"reports_kd/Exp3_Resolution_Robustness_{timestamp}.csv"

    if results:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Resolution", "Model_Role", "mAP", "Rank-1"])
            writer.writeheader()
            writer.writerows(results)
            
        print("\n" + "="*60)
        print("🎉 THỰC NGHIỆM 3 ĐÃ HOÀN TẤT!")
        print(f"📊 Kết quả lưu tại: {csv_file}")
        print("="*60)
    else:
        print("\n⚠️ Không có kết quả nào được ghi nhận.")

if __name__ == "__main__":
    main()