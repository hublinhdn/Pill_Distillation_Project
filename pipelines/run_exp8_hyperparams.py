import subprocess
import csv
import os
import datetime

# ==========================================
# ⚙️ CẤU HÌNH THỰC NGHIỆM 8: LOSS & POOLING ABLATION
# ==========================================
BACKBONE = "resnet18_tv"
EPOCHS = 60  # Chạy 30 epoch là đủ đánh giá xu hướng

# 6 Cấu hình chiến lược để giải phẫu hoàn toàn kiến trúc
# ==========================================
# ⚙️ 7 CẤU HÌNH CHIẾN LƯỢC ĐỂ GIẢI PHẪU KIẾN TRÚC
# ==========================================
CONFIGS = [
    # --- VỊ VUA (The Chosen One) ---
    {"name": "Config A (Chosen - Full & GEM)", "w_sce": 1.0, "w_csce": 0.2, "w_triplet": 1.0, "w_cont": 1.0, "pooling": "gem"},
    
    # --- CÂU CHUYỆN 1: GLOBAL vs LOCAL METRIC (Tại sao phải kết hợp?) ---
    {"name": "Config B (Global Loss Only)",    "w_sce": 1.0, "w_csce": 0.2, "w_triplet": 0.0, "w_cont": 0.0, "pooling": "gem"},
    {"name": "Config C (Local Loss Only)",     "w_sce": 0.0, "w_csce": 0.2, "w_triplet": 1.0, "w_cont": 1.0, "pooling": "gem"},
    
    # --- CÂU CHUYỆN 2: TÌM ĐIỂM CÂN BẰNG TRỌNG SỐ (Sự tinh tế của 0.2 và 1.0) ---
    {"name": "Config D (No Aux CE)",           "w_sce": 1.0, "w_csce": 0.0, "w_triplet": 1.0, "w_cont": 1.0, "pooling": "gem"},
    {"name": "Config E (High Aux CE)",         "w_sce": 1.0, "w_csce": 1.0, "w_triplet": 1.0, "w_cont": 1.0, "pooling": "gem"},
    {"name": "Config F (Over-Penalty Metric)", "w_sce": 1.0, "w_csce": 0.2, "w_triplet": 3.0, "w_cont": 3.0, "pooling": "gem"},
    
    # --- CÂU CHUYỆN 3: POOLING MECHANISM (Tại sao lại là GEM?) ---
    {"name": "Config G (Alternative Pool)",    "w_sce": 1.0, "w_csce": 0.2, "w_triplet": 1.0, "w_cont": 1.0, "pooling": "mpncov"},
]

TEMP_LOG_FILE = "CV_Summary.txt"

def main():
    print("="*80)
    print("🚀 BẮT ĐẦU THỰC NGHIỆM 8: SENSITIVITY ANALYSIS (LOSS & POOLING)")
    print("="*80)

    results = []
    os.makedirs("reports_kd", exist_ok=True)

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        w_sce, w_csce = cfg["w_sce"], cfg["w_csce"]
        w_triplet, w_cont = cfg["w_triplet"], cfg["w_cont"]
        pooling = cfg["pooling"]
        
        print(f"\n[⏳] ĐANG HUẤN LUYỆN: {cfg_name}")
        print(f"     ArcFace: {w_sce} | Aux CE: {w_csce} | Triplet: {w_triplet} | Contrastive: {w_cont} | Pooling: {pooling.upper()}")
        
        if os.path.exists(TEMP_LOG_FILE):
            os.remove(TEMP_LOG_FILE)

        cmd = [
            "python", "pipelines/train_teacher_cv.py",
            "--backbone", BACKBONE,
            "--epochs", str(EPOCHS),
            "--w_sce", str(w_sce),
            "--w_csce", str(w_csce),
            "--w_triplet", str(w_triplet),
            "--w_cont", str(w_cont),
            "--pooling", pooling
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Lỗi khi chạy {cfg_name} (Mạng sụp đổ/DNC).")
            results.append({
                "Config": cfg_name, "Pooling": pooling.upper(),
                "W_ArcFace": w_sce, "W_AuxCE": w_csce, 
                "W_Triplet": w_triplet, "W_Cont": w_cont, 
                "mAP": "0.0000 (DNC)", "Rank-1": "0.0000"
            })
            continue

        try:
            with open(TEMP_LOG_FILE, "r") as f:
                lines = f.readlines()
                last_line = lines[-1].strip()
                
                parts = last_line.split("|")
                if len(parts) >= 3:
                    map_val = parts[1].strip()
                    r1_val = parts[2].strip()
                    
                    results.append({
                        "Config": cfg_name, "Pooling": pooling.upper(),
                        "W_ArcFace": w_sce, "W_AuxCE": w_csce, 
                        "W_Triplet": w_triplet, "W_Cont": w_cont, 
                        "mAP": map_val, "Rank-1": r1_val
                    })
                    print(f"✅ Đã ghi nhận: mAP={map_val}, Rank-1={r1_val}")
        except Exception as e:
            print(f"⚠️ Lỗi đọc kết quả: {e}")

    # ==========================================
    # 💾 XUẤT CSV 
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"reports_kd/Exp8_Hyperparams_Pooling_{timestamp}.csv"

    if results:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Config", "Pooling", "W_ArcFace", "W_AuxCE", "W_Triplet", "W_Cont", "mAP", "Rank-1"])
            writer.writeheader()
            writer.writerows(results)
        print("\n" + "="*80)
        print(f"🎉 HOÀN TẤT! Báo cáo Phụ lục lưu tại: {csv_file}")
        print("="*80)

if __name__ == "__main__":
    main()