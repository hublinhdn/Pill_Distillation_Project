import subprocess
import os
import time
from datetime import datetime

# =====================================================================
# 📋 DANH SÁCH 9 CẶP TEACHER - STUDENT CHIẾN LƯỢC
# =====================================================================
# Cấu trúc: (Tên_Teacher, Tên_Student, Trọng_số_Alpha, KD_Type)
alpha = 10.0
KD_EXPERIMENTS = [
    # 1. Đồng hệ gen (Homogeneous)
    ("efficientnet_b5_timm", "efficientnet_b2_timm", alpha, "cosine"),
    ("convnextv2_base.fcmae_ft_in22k_in1k_384_timm", "convnext_femto_timm", alpha, "cosine"),
    ("efficientnet_b5_timm", "efficientnet_b0_tv", alpha, "cosine"), # Bước nhảy siêu hạng
    
    # 2. Lai hệ gen (Heterogeneous) - Tối ưu Deploy
    ("efficientnet_b5_timm", "repvgg_a0_timm", alpha, "cosine"),
    ("resnest101e_timm", "mobilenetv3_large_100_tv", alpha, "cosine"),
    ("convnextv2_base.fcmae_ft_in22k_in1k_384_timm", "shufflenet_v2_x1_0_tv", alpha, "cosine"),
    
    # 3. Tiến hóa gen & Chú ý (Evolutionary & Attention)
    ("resnest101e_timm", "resnet18_tv", alpha, "cosine"),
    ("seresnext101_32x4d_timm", "resnet18_tv", alpha, "cosine"),
    ("resnest101e_timm", "edgenext_small_timm", alpha, "cosine"),
]

# Thư mục chứa tạ của Teacher (Bạn hãy chỉnh lại nếu đường dẫn khác)
TEACHER_WEIGHTS_DIR = "weights/phase2"
# Tiền tố và hậu tố của file tạ (Ví dụ: best_convnext_base_gem_fold0.pth)
WEIGHT_PREFIX = "best_"
WEIGHT_SUFFIX = "_gem_fold0.pth"

def main():
    os.makedirs("reports_kd", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_summary_file = f"reports_kd/KD_Benchmark_Summary_{timestamp}.txt"
    log_error_file = f"reports_kd/KD_Benchmark_Errors_{timestamp}.txt"

    print(f"🚀 BẮT ĐẦU CHIẾN DỊCH CHƯNG CẤT: {len(KD_EXPERIMENTS)} CẶP")
    print(f"Báo cáo sẽ được lưu tại: {log_summary_file}\n")

    # Tạo Header cho file Log
    with open(log_summary_file, "w") as f:
        f.write(f"KNOWLEDGE DISTILLATION OVERNIGHT BENCHMARK ({timestamp})\n")
        f.write("="*80 + "\n")
        f.write(f"{'TEACHER'.ljust(45)} | {'STUDENT'.ljust(25)} | {'STATUS'}\n")
        f.write("-" * 80 + "\n")

    for i, (teacher, student, alpha, kd_type) in enumerate(KD_EXPERIMENTS, 1):
        print("="*80)
        print(f"🔥 ĐANG CHẠY THỬ NGHIỆM {i}/{len(KD_EXPERIMENTS)}")
        print(f"👨‍🏫 Teacher: {teacher}")
        print(f"🧑‍🎓 Student: {student}")
        print("="*80)

        # 1. Lắp ráp đường dẫn file tạ Sư phụ
        weight_path = os.path.join(TEACHER_WEIGHTS_DIR, f"{WEIGHT_PREFIX}{teacher}{WEIGHT_SUFFIX}")
        
        # Kiểm tra xem tạ có tồn tại không trước khi chạy
        if not os.path.exists(weight_path):
            error_msg = f"❌ LỖI: Không tìm thấy tạ Teacher tại {weight_path}"
            print(error_msg)
            with open(log_error_file, "a") as f:
                f.write(f"[{teacher} -> {student}] {error_msg}\n")
            with open(log_summary_file, "a") as f:
                f.write(f"{teacher[:43].ljust(45)} | {student[:23].ljust(25)} | FAILED (NO WEIGHT)\n")
            continue

        # 2. Xây dựng lệnh Terminal
        cmd = [
            "python", "pipelines/train_student_kd.py",
            "--teacher", teacher,
            "--teacher_weight", weight_path,
            "--student", student,
            "--kd_type", kd_type,
            "--alpha", str(alpha),
            "--epochs", "60" # Bạn có thể đổi số epoch ở đây
        ]

        # 3. Chạy lệnh thông qua Subprocess (Đảm bảo cách ly VRAM hoàn toàn)
        start_time = time.time()
        try:
            # Lệnh này sẽ in thẳng log của quá trình train ra màn hình terminal của bạn
            process = subprocess.run(cmd, check=True)
            status = "SUCCESS"
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ LỖI CRASH KHI HUẤN LUYỆN CẶP: {teacher} -> {student}")
            status = "FAILED (CRASHED)"
            with open(log_error_file, "a") as f:
                f.write(f"[{teacher} -> {student}] Subprocess exited with code {e.returncode}\n")
                
        except KeyboardInterrupt:
            print("\n🛑 BẠN ĐÃ DỪNG TIẾN TRÌNH THỦ CÔNG!")
            break

        # 4. Ghi Log và tính thời gian
        elapsed_time = (time.time() - start_time) / 60
        print(f"⏱️ Hoàn thành trong {elapsed_time:.1f} phút. Trạng thái: {status}\n")
        
        with open(log_summary_file, "a") as f:
            f.write(f"{teacher[:43].ljust(45)} | {student[:23].ljust(25)} | {status}\n")

    print("\n" + "="*80)
    print("🎉 CHIẾN DỊCH CHƯNG CẤT XUYÊN ĐÊM ĐÃ HOÀN TẤT!")
    print(f"Hãy kiểm tra file log: {log_summary_file}")
    print("="*80)

if __name__ == "__main__":
    main()