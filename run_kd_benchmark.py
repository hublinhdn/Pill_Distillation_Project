import os
import subprocess
import re
import time

def main():
    print("🚀 KÍCH HOẠT HỆ THỐNG BENCHMARK KNOWLEDGE DISTILLATION")
    
    # --- CẤU HÌNH CƠ BẢN ---
    TEACHER = "convnext_base"
    TEACHER_WEIGHT = "weights/best_teacher_convnext_base_fold0.pth"
    STUDENT = "resnet18"
    EPOCHS = 60 # Bạn có thể giảm xuống 30 nếu muốn test nhanh sơ bộ
    
    # Tạo thư mục chứa log báo cáo
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    summary_file = "reports/kd_benchmark_summary.txt"
    
    # --- DANH SÁCH CÁC TRẬN ĐÁNH (EXPERIMENTS) ---
    experiments = [
        # Giai đoạn 1: Tìm Tâm Pháp (Alpha tiêu chuẩn)
        {"name": "Phase 1 - MSE", "kd_type": "mse", "alpha": 50.0, "temp": 4.0},
        {"name": "Phase 1 - Cosine", "kd_type": "cosine", "alpha": 50.0, "temp": 4.0},
        {"name": "Phase 1 - KL Div", "kd_type": "kl", "alpha": 1.0, "temp": 4.0},
        
        # Giai đoạn 2: Tìm Alpha "Điểm Ngọt" (Giả định Cosine hoặc MSE đang tốt, ta test thêm hệ số)
        {"name": "Phase 2 - Cosine Low Alpha", "kd_type": "cosine", "alpha": 10.0, "temp": 4.0},
        {"name": "Phase 2 - Cosine High Alpha", "kd_type": "cosine", "alpha": 100.0, "temp": 4.0},
        
        # Giai đoạn 3: Cuộc tấn công tổng lực (Hybrid)
        {"name": "Phase 3 - Hybrid Cocktail", "kd_type": "hybrid", "alpha": 30.0, "temp": 4.0},
    ]

    results = []

    # Khởi tạo file Báo cáo
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"🏆 BÁO CÁO BENCHMARK KD: {TEACHER.upper()} -> {STUDENT.upper()}\n")
        f.write("="*60 + "\n")

    for exp in experiments:
        name = exp["name"]
        kd_type = exp["kd_type"]
        alpha = exp["alpha"]
        temp = exp["temp"]
        
        print(f"\n{'='*50}")
        print(f"🔥 ĐANG CHẠY: {name}")
        print(f"   ⚙️  Loại KD: {kd_type.upper()} | Alpha: {alpha} | Temp: {temp}")
        print(f"{'='*50}")
        
        log_filename = f"logs/kd_{kd_type}_alpha{alpha}_temp{temp}.log"
        
        # Lệnh chạy script huấn luyện
        cmd = [
            "python", "pipelines/train_student_kd.py",
            "--teacher", TEACHER,
            "--teacher_weight", TEACHER_WEIGHT,
            "--student", STUDENT,
            "--epochs", str(EPOCHS),
            "--kd_type", kd_type,
            "--alpha", str(alpha),
            "--temperature", str(temp)
        ]
        
        best_map = 0.0
        
        # Thực thi và ghi log trực tiếp
        with open(log_filename, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            for line in process.stdout:
                print(line, end="") # Vẫn in ra màn hình để bạn theo dõi
                log_file.write(line)
                
                # Quét dòng output để tìm giá trị mAP
                # Dựa vào format in ra của train_student_kd.py: "📊 Epoch X mAP: 0.YYYY"
                match = re.search(r"mAP:\s*(0\.\d+)", line)
                if match:
                    current_map = float(match.group(1))
                    if current_map > best_map:
                        best_map = current_map
                        
            process.wait() # Đợi chạy xong epoch cuối
            
        print(f"\n✅ HOÀN THÀNH {name} - Best mAP: {best_map:.4f}")
        results.append({"name": name, "kd_type": kd_type, "alpha": alpha, "best_map": best_map})
        
        # Ghi ngay vào file summary để backup
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"{name.ljust(30)} | Type: {kd_type.ljust(6)} | Alpha: {str(alpha).ljust(5)} => Best mAP: {best_map:.4f}\n")
            
        time.sleep(5) # Nghỉ 5 giây cho GPU xả RAM trước khi chạy trận mới

    # --- IN TỔNG KẾT CUỐI CÙNG ---
    print("\n" + "🌟"*25)
    print("🏆 BẢNG VÀNG KẾT QUẢ KNOWLEDGE DISTILLATION")
    print("🌟"*25)
    for res in results:
        print(f"- {res['name'].ljust(30)}: mAP = {res['best_map']:.4f}")
    print("🌟"*25)
    print(f"📁 Toàn bộ chi tiết đã được lưu tại: {summary_file}")

if __name__ == "__main__":
    main()