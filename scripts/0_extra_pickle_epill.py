import pickle
import sys
import os

# Xử lý lỗi tương thích module của sklearn cũ
try:
    import sklearn.preprocessing._label
    sys.modules['sklearn.preprocessing.label'] = sys.modules['sklearn.preprocessing._label']
except ImportError:
    pass

def export_labels(pickle_path, output_txt):
    with open(pickle_path, 'rb') as f:
        le = pickle.load(f)
    
    classes = le.classes_
    with open(output_txt, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"✅ Đã xuất {len(classes)} nhãn ra file {output_txt}")

if __name__ == "__main__":
    root_path = 'data/raw/ePillID'
    folds_dir = os.path.join(root_path, 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base')
    pickle_file = os.path.join(folds_dir, 'label_encoder_pytorch131.pickle')
    output = os.path.join(folds_dir, 'pill_classes.txt') 
    export_labels(pickle_file, output)