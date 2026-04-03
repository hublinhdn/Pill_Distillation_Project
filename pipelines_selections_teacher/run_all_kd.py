"""
Train student with KD, require all baselines finish
4 students x 11 teachers => 44 run

tmux new -s run_all_kd -d "bash -lc '
python pipelines_selections_teacher/run_all_kd.py \
|& tee -a logs/run_all_kd_$(date +%F_%H%M%S).log
'"

"""
import os
import subprocess
import time

def main():
    # Danh sách 4 Students và 11 Teachers
    students = ['resnet18_tv', 'mobilenetv3_large_100_tv', 'ghostnet_100_timm', 'efficientnet_b1_timm']
    teachers = [
        'efficientnet_b5_timm', 'convnextv2_base.fcmae_ft_in22k_in1k_384_timm',
        'convnext_base_timm', 'resnest101e_timm', 'convnext_large_timm', 
        'seresnext101_32x4d_timm', 'densenet161_tv', 'tresnet_l_timm', 
        'resnet101_tv', 'tf_efficientnetv2_l.in21k_ft_in1k', 'maxvit_base_tf_384_timm'
    ]

    # Các siêu tham số KD mặc định (Bạn có thể tinh chỉnh tại đây)
    kd_type = "cosine"
    alpha = "10.0"
    temperature = "4.0"
    
    # Các trọng số loss đã dùng để train baseline (Dùng để tìm đúng file weight của Teacher)
    # Nếu baseline bạn chạy với số khác, hãy sửa lại ở đây
    t_sce, t_csce, t_trip, t_cont = "1.0", "0.2", "1.0", "1.0"
    
    pipeline_name = "KD_Matrix_Experiment"
    
    total_runs = len(students) * len(teachers)
    current_run = 0

    print(f"🚀 BẮT ĐẦU CHIẾN DỊCH CHƯNG CẤT TRI THỨC ({total_runs} TỔ HỢP)")
    print("="*70)

    for student in students:
        for teacher in teachers:
            current_run += 1
            print(f"\n⏳ [{current_run}/{total_runs}] Đang thiết lập: Thầy [{teacher}] -> Trò [{student}]")
            
            # 1. Xác định đường dẫn trọng số của Teacher
            # Giả định vòng Baseline sử dụng pooling='gem' và fold=0
            teacher_weight = f"weights/phase3/best_{teacher}_gem_fold0_{t_sce}_{t_csce}_{t_trip}_{t_cont}.pth"
            
            if not os.path.exists(teacher_weight):
                print(f"   ⚠️ CẢNH BÁO: Không tìm thấy trọng số Teacher tại {teacher_weight}")
                print(f"   ⏭️ BỎ QUA tổ hợp này!")
                continue

            # 2. Xây dựng câu lệnh khởi chạy
            command = [
                "python", "pipelines/train_student_kd.py",
                "--teacher", teacher,
                "--teacher_weight", teacher_weight,
                "--student", student,
                "--kd_type", kd_type,
                "--alpha", alpha,
                "--temperature", temperature,
                "--pipeline_name", pipeline_name
            ]
            
            try:
                # 3. Thực thi tiến trình độc lập
                start_time = time.time()
                subprocess.run(command, check=True)
                end_time = time.time()
                
                hours, rem = divmod(end_time - start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"✅ Hoàn thành tổ hợp trong {hours:0>2}:{minutes:0>2}:{seconds:05.2f}")
                
            except subprocess.CalledProcessError as e:
                # Nếu xảy ra lỗi OOM, NaN, hoặc Crash, bỏ qua và chạy tổ hợp tiếp theo
                print(f"❌ LỖI NGHIÊM TRỌNG KHI CHẠY: {teacher} -> {student}.")
                print("Tiếp tục chuyển sang tổ hợp tiếp theo...")
                
            print("-" * 70)
            # 4. Giải phóng hoàn toàn bộ nhớ giữa các lần chạy
            time.sleep(5) 

    print("\n🎉 CHIẾN DỊCH CHƯNG CẤT ĐÃ HOÀN TẤT!")
    print(f"Báo cáo tổng hợp được lưu tại: reports_kd/{pipeline_name}_Summary.txt")

if __name__ == "__main__":
    main()