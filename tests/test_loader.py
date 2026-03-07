import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Thêm thư mục gốc vào path để import được utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_loader import PillDataset, get_transforms

def run_test_suite():
    print("🚀 Bắt đầu Run_All Unit Test cho Dataset Loader...\n")
    
    # Cấu hình đường dẫn (Hãy điều chỉnh cho khớp với máy bạn)
    EPILL_CSV = 'data/raw/ePillID/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv'
    EPILL_ROOT = 'data/raw/ePillID/classification_data' # Thư mục chứa folder images của ePillID
    OGYEI_CSV = 'data/processed/ogyei_manifest.csv'

    tests = [
        {"name": "ePillID_Lab", "csv": EPILL_CSV, "root": EPILL_ROOT, "mask": False, "filter": "is_ref == 1"},
        {"name": "ePillID_Consumer", "csv": EPILL_CSV, "root": EPILL_ROOT, "mask": False, "filter": "is_ref == 0"},
        {"name": "OGYEIv2_Masked", "csv": OGYEI_CSV, "root": "", "mask": True, "filter": None}
    ]

    for test in tests:
        try:
            print(f"Testing {test['name']}...")
            df = pd.read_csv(test['csv'])
            if test['filter']:
                df = df.query(test['filter'])
            
            dataset = PillDataset(
                df=df.head(5), 
                root_dir=test['root'], 
                transform=get_transforms(is_train=False),
                use_mask=test['mask']
            )
            
            img, label = dataset[0]
            print(f"   ✅ [PASS] Load thành công. Image size: {img.shape}, Label: {label}")
            
        except Exception as e:
            print(f"   ❌ [FAIL] Lỗi tại {test['name']}: {str(e)}")

    print("\n🏁 Hoàn tất kiểm tra.")

if __name__ == "__main__":
    run_test_suite()