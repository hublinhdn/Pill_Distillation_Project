import subprocess
import time
"""
tmux new -s baseline_full_15_models -d "bash -lc '
python pipelines_selections_teacher/run_all_baselines.py \
|& tee -a logs/baseline_full_15_models_$(date +%F_%H%M%S).log
'"

"""
def main():
    students = ['resnet18_tv', 'mobilenetv3_large_100_tv', 'ghostnet_100_timm', 'efficientnet_b1_timm']
    teachers = [
        'efficientnet_b5_timm', 'convnextv2_base.fcmae_ft_in22k_in1k_384_timm',
        'convnext_base_timm', 'resnest101e_timm', 'convnext_large_timm', 
        'seresnext101_32x4d_timm', 'densenet161_tv', 'tresnet_l_timm', 
        'resnet101_tv', 'tf_efficientnetv2_l.in21k_ft_in1k', 'maxvit_base_tf_384_timm'
    ]

    # Gộp chung vào một list để chạy tuần tự
    all_backbones = teachers + students
    
    pipeline_name = "baseline_full_15_models"
    pooling_type = "gem"
    
    print(f"🚀 BẮT ĐẦU CHIẾN DỊCH HUẤN LUYỆN {len(all_backbones)} MÔ HÌNH")
    print("="*60)

    for idx, backbone in enumerate(all_backbones, 1):
        print(f"\n⏳ [{idx}/{len(all_backbones)}] Đang khởi động tiến trình cho: {backbone.upper()}")
        
        # Cấu trúc câu lệnh gọi terminal
        command = [
            "python", "pipelines/train_teacher_cv.py",
            "--backbone", backbone,
            "--pooling", pooling_type,
            "--pipeline_name", pipeline_name
        ]
        
        try:
            # Chạy tiến trình con, hiển thị trực tiếp log ra màn hình
            start_time = time.time()
            process = subprocess.run(command, check=True)
            end_time = time.time()
            
            hours, rem = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"✅ Hoàn thành {backbone} trong {hours:0>2}:{minutes:0>2}:{seconds:05.2f}")
            
        except subprocess.CalledProcessError as e:
            # Nếu script con bị lỗi (OOM, NaN,...), vòng lặp vẫn đi tiếp sang model sau
            print(f"❌ LỖI NGHIÊM TRỌNG KHI CHẠY {backbone}.")
            print(f"Tiếp tục chuyển sang mô hình tiếp theo...")
            
        print("-" * 60)
        # Nghỉ 5 giây giữa các mô hình để hệ điều hành dọn dẹp RAM/VRAM
        time.sleep(5) 

    print("\n🎉 CHIẾN DỊCH ĐÃ HOÀN TẤT!")
    print(f"Bạn có thể kiểm tra file báo cáo: reports/{pipeline_name}_batch_experiment_{pooling_type}_summary.txt")

if __name__ == "__main__":
    main()