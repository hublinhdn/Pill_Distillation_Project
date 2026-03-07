import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillTeacher
from utils.dataset_loader import PillDataset, get_transforms
from utils.data_utils import load_epill_full_data
from utils.evaluator import evaluate_retrieval

def eval_best_teacher(weight_path='weights/teacher_best.pth'):
    EPILL_ROOT = 'data/raw/ePillID/classification_data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_all = load_epill_full_data()
    df_ref_all = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_cons_all = df_all[df_all['is_ref'] == 0].reset_index(drop=True)
    num_classes = df_all['label_idx'].nunique()

    model = PillTeacher(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    results = []
    for i in range(5):
        df_query = df_cons_all[df_cons_all['fold'] == i]
        df_eval = pd.concat([df_query, df_ref_all]).drop_duplicates().reset_index(drop=True)
        loader = DataLoader(PillDataset(df_eval, get_transforms(False), EPILL_ROOT), batch_size=64)
        
        m = evaluate_retrieval(model, loader, device)
        results.append(m)
        print(f"Fold {i}: mAP {m['mAP']:.4f}")

    print("\nFinal Average mAP:", pd.DataFrame(results)['mAP'].mean())

if __name__ == "__main__":
    eval_best_teacher()