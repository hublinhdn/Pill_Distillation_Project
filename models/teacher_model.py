import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PillTeacher(nn.Module):
    def __init__(self, num_classes, embedding_size=512, backbone_type='resnet50'):
        super(PillTeacher, self).__init__()
        
        # 1. Khởi tạo Backbone (Features Extractor)
        if backbone_type == 'resnet50':
            base = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        elif backbone_type == 'resnet101':
            base = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone_type} chưa được hỗ trợ.")

        # 2. Bilinear Pooling Head
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.fc_bilinear = nn.Linear(512 * 512, embedding_size)
        self.bn_head = nn.BatchNorm1d(embedding_size)

        # 3. Classification & Metric Heads
        self.fc_ce = nn.Linear(embedding_size, num_classes)
        self.proxy_cos = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxy_cos)

    def forward(self, x, labels=None):
        # Trích xuất đặc trưng spatial: [B, 2048, 14, 14] khi size=448
        feat = self.features(x) 
        feat = self.reduce_conv(feat) # [B, 512, 14, 14]
        
        # Bilinear Operation
        batch_size, channels, h, w = feat.size()
        feat = feat.view(batch_size, channels, h * w)
        bilinear_feat = torch.bmm(feat, feat.transpose(1, 2)) / (h * w)
        bilinear_feat = bilinear_feat.view(batch_size, -1)
        
        # Power-Normalization & L2
        bilinear_feat = torch.sign(bilinear_feat) * torch.sqrt(torch.abs(bilinear_feat) + 1e-8)
        bilinear_feat = F.normalize(bilinear_feat, p=2, dim=1)
        
        # Projection
        embedding = self.bn_head(self.fc_bilinear(bilinear_feat))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            logits_sce = self.fc_ce(embedding)
            norm_proxy = F.normalize(self.proxy_cos, p=2, dim=1)
            logits_csce = F.linear(norm_embedding, norm_proxy) * 16.0
            return logits_sce, logits_csce, norm_embedding
        
        return norm_embedding