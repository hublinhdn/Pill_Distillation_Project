import subprocess
import csv
import os
import datetime
import numpy as np

# ==========================================
# ⚙️ CẤU HÌNH THỰC NGHIỆM 7: STATISTICAL VALIDATION (3 SEEDS)
# ==========================================
TEACHER = "resnest101e_timm"
STUDENT = "resnet18_tv"

# ⚠️ QUAN TRỌNG: Đảm bảo đường dẫn này đúng trên máy bạn
TEACHER_WEIGHT_PATH = "weights/phase2/best_resnest101e_timm_gem_fold0.pth" 

KD_TYPE = "cosine"
ALPHA_KD = 10.0
SEEDS = [42, 1234, 2024]  # 3 random seeds chuẩn mực trong nghiên cứu

TEMP_SUMMARY_FILE = "temp_exp7_summary.txt"

def main():
    print("="*60)
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM 7: STATISTICAL VALIDATION (3 SEEDS)")
    print(f"👨‍🏫 Teacher: {TEACHER} | 👨‍🎓 Student: {STUDENT}")
    print(f"⚖️ Cấu hình tốt nhất: KD {KD_TYPE.upper()}, Alpha {ALPHA_KD}")
    print(f"🌱 Các hạt giống (Seeds) sẽ chạy: {SEEDS}")
    print("="*60)

    results = []
    map_list = []
    r1_list = []
    
    os.makedirs("reports_kd", exist_ok=True)

    for seed in SEEDS:
        print(f"\n[⏳ {datetime.datetime.now().strftime('%H:%M:%S')}] ĐANG HUẤN LUYỆN VỚI RANDOM SEED = {seed} ...")
        
        # 1. Gọi file train, truyền tham số --seed
        cmd = [
            "python", "pipelines/train_student_kd.py",
            "--teacher", TEACHER,
            "--teacher_weight", TEACHER_WEIGHT_PATH,
            "--student", STUDENT,
            "--kd_type", KD_TYPE,
            "--alpha", str(ALPHA_KD),
            "--seed", str(seed),  # Truyền seed vào đây
            "--summary_file", TEMP_SUMMARY_FILE
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy Seed = {seed}. Bỏ qua...")
            continue

        # 2. Đọc kết quả từ file log tạm
        try:
            temp_log_path = os.path.join("reports_kd", TEMP_SUMMARY_FILE)
            with open(temp_log_path, "r") as f:
                lines = f.readlines()
                last_line = lines[-1].strip()
                
                parts = last_line.split("|")
                if len(parts) >= 5:
                    map_val = float(parts[3].strip())
                    r1_val = float(parts[4].strip())
                    
                    results.append({
                        "Seed": seed,
                        "mAP": map_val,
                        "Rank-1": r1_val
                    })
                    map_list.append(map_val)
                    r1_list.append(r1_val)
                    
                    print(f"✅ Đã ghi nhận Seed={seed}: mAP={map_val:.4f}, Rank-1={r1_val:.4f}")
        except Exception as e:
            print(f"⚠️ Không thể đọc kết quả cho Seed = {seed}. Chi tiết lỗi: {e}")

    # ==========================================
    # 🧮 3. TÍNH TOÁN THỐNG KÊ (MEAN ± STD)
    # ==========================================
    if len(map_list) == len(SEEDS):
        mean_map = np.mean(map_list)
        std_map = np.std(map_list)
        
        mean_r1 = np.mean(r1_list)
        std_r1 = np.std(r1_list)
        
        print("\n" + "="*60)
        print("🎉 TỔNG HỢP KẾT QUẢ STATISTICAL VALIDATION:")
        print(f"📈 mAP (Mean ± Std):    {mean_map:.4f} ± {std_map:.4f}")
        print(f"🥇 Rank-1 (Mean ± Std): {mean_r1:.4f} ± {std_r1:.4f}")
        print("="*60)
        
        # Ghi thêm dòng tổng hợp vào danh sách để lưu CSV
        results.append({
            "Seed": "MEAN ± STD",
            "mAP": f"{mean_map:.4f} ± {std_map:.4f}",
            "Rank-1": f"{mean_r1:.4f} ± {std_r1:.4f}"
        })

    # 4. Xuất CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"reports_kd/Exp7_Statistical_Validation_{timestamp}.csv"

    if results:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Seed", "mAP", "Rank-1"])
            writer.writeheader()
            writer.writerows(results)
        print(f"📊 Báo cáo chi tiết lưu tại: {csv_file}\n")
    else:
        print("\n⚠️ Không có kết quả nào để tính toán.")

if __name__ == "__main__":
    main()