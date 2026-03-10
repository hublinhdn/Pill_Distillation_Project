import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MPNCOV(nn.Module):
    def __init__(self, iterNum=5): 
        super(MPNCOV, self).__init__()
        self.iterNum = iterNum

    def forward(self, x):
        batchSize, channels, h, w = x.data.shape
        M = h * w
        x = x.view(batchSize, channels, M)
        I_hat = (-1.0/M) * torch.ones(M, M, device=x.device) + torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2)) / M
        trY = y.diagonal(dim1=-2, dim2=-1).sum(1)
        y = y / trY.view(batchSize, 1, 1)
        I = torch.eye(channels, channels, device=x.device).view(1, channels, channels).repeat(batchSize, 1, 1)
        Y, Z = y, I
        for i in range(self.iterNum):
            ZY = Z.bmm(Y)
            Y = 0.5 * Y.bmm(3.0 * I - ZY)
            Z = 0.5 * (3.0 * I - ZY).bmm(Z)
        return Y * torch.sqrt(trY).view(batchSize, 1, 1)

class PillTeacher(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=4902, embedding_size=1024):
        super(PillTeacher, self).__init__()
        
        # 1. Backbone Selection
        if 'resnet50' in backbone_name:
            base = models.resnet50(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        elif 'convnext' in backbone_name:
            base = models.convnext_base(weights='IMAGENET1K_V1')
            self.features = base.features
            in_channels = 1024
        elif 'efficientnet' in backbone_name:
            base = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            self.features = base.features
            in_channels = 1280

        # 2. Cấu trúc Head đạt mAP 0.79
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.mpn_cov = MPNCOV(iterNum=5)
        
        self.fc_projection = nn.Linear(256 * 256, embedding_size, bias=False)
        self.bn_head = nn.BatchNorm1d(embedding_size)
        
        # Classifiers
        self.fc_ce = nn.Linear(embedding_size, num_classes)
        self.proxy_cos = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxy_cos)

    def forward(self, x, labels=None): # Thêm labels=None để khớp với lời gọi trong train script
        x = self.features(x)
        x = self.reduce_conv(x)
        matrix_sqrt = self.mpn_cov(x)
        flat_feat = matrix_sqrt.view(matrix_sqrt.size(0), -1)
        
        # Quá trình tạo Embedding
        embedding = self.bn_head(self.fc_projection(flat_feat))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        # Tính toán Logits
        logits_sce = self.fc_ce(embedding)
        norm_proxies = F.normalize(self.proxy_cos, p=2, dim=1)
        logits_cos = torch.mm(norm_embedding, norm_proxies.t())
        
        # --- SỬA THỨ TỰ TRẢ VỀ ĐỂ KHỚP VỚI TRAIN SCRIPT ---
        # Thứ tự mong đợi: (logits_sce, logits_csce, norm_emb)
        return logits_sce, logits_cos, norm_embedding