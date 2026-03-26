import subprocess
import csv
import json
import os
import datetime

# ==========================================
# ⚙️ CẤU HÌNH THỰC NGHIỆM 2: ABLATION KD LOSS
# ==========================================
TEACHER = "resnest101e_timm"
STUDENT = "resnet18_tv"

# ⚠️ QUAN TRỌNG: Sửa lại đường dẫn file weights của Teacher trên máy bạn
TEACHER_WEIGHT_PATH = "weights/phase2/best_resnest101e_timm_gem_fold0.pth" 

ALPHA_KD = 10.0  # Fix cứng Alpha theo yêu cầu
KD_TYPES = ["mse", "kl"] # Bạn có thể thêm "cosine" vào list này nếu muốn chạy lại cho trọn bộ

# File log tạm thời cho script con
TEMP_SUMMARY_FILE = "temp_exp2_summary.txt"

def main():
    print("="*60)
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM 2: ABLATION KNOWLEDGE DISTILLATION LOSS")
    print(f"👨‍🏫 Teacher: {TEACHER} | 👨‍🎓 Student: {STUDENT}")
    print(f"⚖️ Alpha cố định: {ALPHA_KD} | Các hàm Loss cần test: {KD_TYPES}")
    print("="*60)

    results = []
    
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs("reports", exist_ok=True)

    for kd_type in KD_TYPES:
        print(f"\n[⏳ {datetime.datetime.now().strftime('%H:%M:%S')}] ĐANG CHẠY KD_TYPE = '{kd_type.upper()}' ...")
        
        # 1. Tạo command để gọi file train_student_kd.py
        cmd = [
            "python", "pipelines/train_student_kd.py",
            "--teacher", TEACHER,
            "--teacher_weight", TEACHER_WEIGHT_PATH,
            "--student", STUDENT,
            "--kd_type", kd_type,
            "--alpha", str(ALPHA_KD),
            "--summary_file", TEMP_SUMMARY_FILE
        ]
        
        # 2. Chạy subprocess (Chờ cho đến khi train xong 1 hàm loss)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi xảy ra khi chạy KD Type = {kd_type}. Bỏ qua và chạy tiếp...")
            continue

        # 3. Đọc kết quả từ file log tạm để trích xuất mAP và Rank-1
        try:
            temp_log_path = os.path.join("reports", TEMP_SUMMARY_FILE)
            with open(temp_log_path, "r") as f:
                lines = f.readlines()
                last_line = lines[-1].strip()
                
                # Format dòng log: teacher --> student | alpha | kd_type | mAP | r1
                parts = last_line.split("|")
                if len(parts) >= 5:
                    map_val = float(parts[3].strip())
                    r1_val = float(parts[4].strip())
                    
                    results.append({
                        "Teacher": TEACHER,
                        "Student": STUDENT,
                        "Alpha": ALPHA_KD,
                        "KD_Type": kd_type.upper(),
                        "mAP": map_val,
                        "Rank-1": r1_val
                    })
                    print(f"✅ Đã ghi nhận Loss={kd_type.upper()}: mAP={map_val:.4f}, Rank-1={r1_val:.4f}")
        except Exception as e:
            print(f"⚠️ Không thể đọc kết quả cho Loss = {kd_type}. Chi tiết lỗi: {e}")

    # ==========================================
    # 💾 4. XUẤT KẾT QUẢ RA CSV VÀ JSON
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"reports/Exp2_Loss_Ablation_{timestamp}.csv"
    json_file = f"reports/Exp2_Loss_Ablation_{timestamp}.json"

    if results:
        # Ghi CSV
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Teacher", "Student", "Alpha", "KD_Type", "mAP", "Rank-1"])
            writer.writeheader()
            writer.writerows(results)
            
        # Ghi JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        print("\n" + "="*60)
        print("🎉 THỰC NGHIỆM 2 ĐÃ HOÀN TẤT!")
        print(f"📊 Kết quả CSV lưu tại: {csv_file}")
        print(f"📋 Kết quả JSON lưu tại: {json_file}")
        print("="*60)
    else:
        print("\n⚠️ Không có kết quả nào được ghi nhận. Vui lòng kiểm tra lại log.")

if __name__ == "__main__":
    main()