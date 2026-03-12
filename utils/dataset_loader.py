import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class PillDataset(Dataset):
    # Thêm tham số transform_ref
    def __init__(self, df, transform=None, transform_ref=None):
        self.df = df
        self.transform = transform
        # Nếu không truyền transform_ref, mặc định dùng chung transform
        self.transform_ref = transform_ref if transform_ref is not None else transform
        self.root_path = 'data/raw/ePillID/classification_data'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_path, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = row['label_idx']
        sub_label = row['sub_label_idx']
        is_ref = row['is_ref']
        
        # PHÂN LUỒNG AUGMENTATION THÔNG MINH
        if is_ref == 1 and self.transform_ref:
            image = self.transform_ref(image)
        elif self.transform:
            image = self.transform(image)

        # Trả về sub_label để train, label gốc để eval
        return image, sub_label, label, is_ref

def get_transforms(is_train=True, size=448): # Cập nhật size mặc định lên 448
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # Giảm góc xoay xuống 15 để tránh lật ngược chữ
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.label_to_indices = {l: np.where(labels == l)[0] for l in np.unique(labels)}
        self.labels_list = list(self.label_to_indices.keys())

    def __iter__(self):
        for _ in range(len(self.labels) // (self.n_classes * self.n_samples)):
            batch = []
            classes = np.random.choice(self.labels_list, self.n_classes, replace=False)
            for cls in classes:
                indices = self.label_to_indices[cls]
                batch.extend(np.random.choice(indices, self.n_samples, replace=True))
            yield batch

    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_samples)