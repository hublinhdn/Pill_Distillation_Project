import torch
import numpy as np
from tqdm import tqdm

def evaluate_retrieval(model, query_loader, gallery_loader, device):
    model.eval()
    
    def extract(loader, desc):
        all_embs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, _ in tqdm(loader, desc=desc, leave=False):
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda'):
                    # Lấy output cuối cùng (norm_embedding)
                    out = model(imgs)
                    emb = out[0] if isinstance(out, tuple) else out
                    
                    # Flip Augmentation trong lúc Eval (Chiến lược ép mAP)
                    out_f = model(torch.flip(imgs, dims=[3]))
                    emb_f = out_f[0] if isinstance(out_f, tuple) else out_f
                    
                    emb_final = torch.nn.functional.normalize((emb + emb_f) / 2.0, p=2, dim=1)
                
                all_embs.append(emb_final.cpu())
                all_labels.append(labels)
        return torch.cat(all_embs), torch.cat(all_labels)

    # Trích xuất đặc trưng Query và Gallery riêng biệt
    q_embs, q_labels = extract(query_loader, "Extract Query")
    g_embs, g_labels = extract(gallery_loader, "Extract Gallery")

    # Tính ma trận tương đồng Cosine
    sim_matrix = torch.mm(q_embs, g_embs.t())
    
    # Tính mAP và Rank-1 (Dựa trên logic metrics.py của bạn)
    unique_g_labels = torch.unique(g_labels)
    # Max-matching logic
    max_sims = torch.zeros((len(q_labels), len(unique_g_labels)))
    for j, lb in enumerate(unique_g_labels):
        mask = (g_labels == lb)
        max_sims[:, j] = torch.max(sim_matrix[:, mask], dim=1)[0]

    # Tính toán mAP dựa trên max_sims
    indices = torch.argsort(max_sims, dim=1, descending=True)
    matched_labels = unique_g_labels[indices]
    correct_mask = (matched_labels == q_labels.view(-1, 1))
    
    # Rank-1
    rank1 = torch.mean(correct_mask[:, 0].float()).item()
    
    # mAP
    aps = []
    for i in range(len(q_labels)):
        pos = torch.where(correct_mask[i])[0] + 1
        if len(pos) > 0:
            aps.append(1.0 / pos[0].item())
        else:
            aps.append(0.0)
            
    return {'mAP': np.mean(aps), 'Rank-1': rank1}