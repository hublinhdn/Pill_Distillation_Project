"""
tmux new -s recalcaulte -d "bash -lc '
python pipelines_selections_teacher/recalculate_metrics.py \
|& tee -a logs/recalcaulte_15_models_$(date +%F_%H%M%S).log
'"

"""
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

# Thêm root path để import module local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data
from utils.evaluator import evaluate_retrieval

def get_weight_path(backbone, pooling='gem', fold=0, w_sce=1.0, w_csce=0.2, w_triplet=1.0, w_cont=1.0):
    folder_weight = 'weights/phase3'
    return f"{folder_weight}/best_{backbone}_{pooling}_fold{fold}_{w_sce}_{w_csce}_{w_triplet}_{w_cont}.pth"

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Đang chạy đánh giá lại trên: {device}")

    # Danh sách 15 ứng viên
    students = ['resnet18_tv', 'mobilenetv3_large_100_tv', 'ghostnet_100_timm', 'efficientnet_b1_timm']
    teachers = [
        'efficientnet_b5_timm', 'convnextv2_base.fcmae_ft_in22k_in1k_384_timm',
        'convnext_base_timm', 'resnest101e_timm', 'convnext_large_timm', 
        'seresnext101_32x4d_timm', 'densenet161_tv', 'tresnet_l_timm', 
        'resnet101_tv', 'tf_efficientnetv2_l.in21k_ft_in1k', 'maxvit_base_tf_384_timm'
    ]

    all_backbones = teachers + students

    # 1. Chuẩn bị dữ liệu Validation dùng chung
    print("⏳ Đang chuẩn bị tập dữ liệu Validation...")
    df_all = load_epill_full_data()
    total_sub_classes = df_all['sub_label_idx'].nunique()
    
    fold = 0
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)

    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Gộp Consumer Query và Reference Gallery vào chung 1 loader (1-pass inference)
    val_loader = DataLoader(
        PillDataset(pd.concat([df_val, df_ref_gallery]).reset_index(drop=True), transform=val_transform), 
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # Khởi tạo danh sách lưu kết quả
    recalc_results = []

    print(f"🚀 BẮT ĐẦU CHẤM ĐIỂM LẠI CHO {len(all_backbones)} MÔ HÌNH")
    print("-" * 70)

    for backbone in all_backbones:
        weight_path = get_weight_path(backbone)
        if not os.path.exists(weight_path):
            print(f"⚠️ BỎ QUA: Không tìm thấy trọng số cho {backbone}")
            continue

        print(f"🟢 Đang đánh giá: {backbone}")
        try:
            # Khởi tạo và nạp trọng số
            model = PillRetrievalModel(num_classes=total_sub_classes, backbone_type=backbone, pooling_type='gem').to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            
            # Chạy Evaluator với cờ đánh giá Delta
            metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=True, flag_eval_delta=True)
            
            # Trích xuất dữ liệu
            recalc_results.append({
                'Backbone': backbone,
                'mAP(Cons)': round(metrics.get('mAP', 0.0), 4),
                'Rank-1(Cons)': round(metrics.get('Rank-1', 0.0), 4),
                'mAP(Ref)': round(metrics.get('mAP(Ref)', 0.0), 4),
                'mAP(Delta)': round(metrics.get('mAP(Delta)', 0.0), 4)
            })
            
            print(f"   => mAP(Cons): {metrics.get('mAP'):.4f} | mAP(Ref): {metrics.get('mAP(Ref)'):.4f} | Delta: {metrics.get('mAP(Delta)'):.4f}")
            
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {backbone}: {e}")

    # ==========================================
    # LƯU VÀ IN BÁO CÁO TỔNG KẾT
    # ==========================================
    df_results = pd.DataFrame(recalc_results)
    
    # Sắp xếp theo mAP Consumer giảm dần để dễ nhìn
    df_results = df_results.sort_values(by='mAP(Cons)', ascending=False).reset_index(drop=True)
    
    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/Recalculated_Metrics_Summary.csv"
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print("🏆 BẢNG ĐIỂM CHUẨN XÁC ĐÃ ĐƯỢC CẬP NHẬT")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    print(f"📁 Dữ liệu chi tiết đã được lưu tại: {csv_path}")

if __name__ == "__main__":
    main()