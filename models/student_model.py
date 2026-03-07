import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PillStudent(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(PillStudent, self).__init__()
        # Sử dụng MobileNetV3-Large để cân bằng giữa tốc độ và độ chính xác
        self.backbone = models.mobilenet_v3_large(weights='DEFAULT')
        
        # Lấy số feature đầu vào của lớp classifier cuối cùng
        in_features = self.backbone.classifier[0].in_features
        # Thay thế classifier bằng Identity để lấy feature map
        self.backbone.classifier = nn.Identity()
        
        # Nhánh Embedding (Neck) - Khớp với kích thước của Teacher (512)
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # Head cho CrossEntropy
        self.fc_ce = nn.Linear(embedding_size, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        
        # L2 Normalize
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            logits_ce = self.fc_ce(embedding)
            return logits_ce, norm_embedding
        
        return norm_embedding