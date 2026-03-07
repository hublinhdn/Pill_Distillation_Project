import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class PillDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.root_path = 'data/raw/ePillID/classification_data'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.df.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label_idx']
        is_ref = self.df.iloc[idx]['is_ref']
        
        if self.transform:
            image = self.transform(image)
        return image, label, is_ref

def get_transforms(is_train=True, size=224):
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
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