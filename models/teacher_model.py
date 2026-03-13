import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.avg_pool2d(x, (x.size(-2), x.size(-1))).pow(1./self.p)

class MPNCOV(nn.Module):
    def __init__(self, iterNum=3):
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

class PillRetrievalModel(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet50', pooling_type='gem', embedding_size=512, s=64.0, m=0.35):
        super(PillRetrievalModel, self).__init__()
        self.s = s 
        self.m = m 
        self.pooling_type = pooling_type.lower()
        
        # --- BACKBONE SELECTION ---
        if backbone_type == 'convnext_large':
            base = models.convnext_large(weights='DEFAULT')
            self.features = base.features
        elif backbone_type == 'convnext_base':
            base = models.convnext_base(weights='DEFAULT')
            self.features = base.features
        elif backbone_type == 'resnet101':
            base = models.resnet101(weights='DEFAULT')
            base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            self.features = nn.Sequential(*list(base.children())[:-2])
        elif backbone_type == 'mobilenet_v3_large':
            base = models.mobilenet_v3_large(weights='DEFAULT')
            self.features = base.features
        elif backbone_type == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='DEFAULT')
            self.features = base.features
        elif backbone_type == 'resnet18':
            base = models.resnet18(weights='DEFAULT')
            base.layer4[0].conv1.stride = (1, 1) 
            base.layer4[0].downsample[0].stride = (1, 1)
            self.features = nn.Sequential(*list(base.children())[:-2])
        else: 
            base = models.resnet50(weights='DEFAULT')
            base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            self.features = nn.Sequential(*list(base.children())[:-2])
        
        # Tự động trích xuất kênh bằng Dummy Tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_feat = self.features(dummy_input)
            in_channels = dummy_feat.shape[1] 
            
        print(f"✅ Khởi tạo [{backbone_type}] - Channels: {in_channels} - Pooling: {self.pooling_type.upper()}")

        # --- POOLING & PROJECTION HEAD ---
        if self.pooling_type == 'mpncov':
            # Ép kênh xuống 256 trước khi tính ma trận Hiệp phương sai
            self.mpncov_bottleneck = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
            self.pool = MPNCOV(iterNum=3)
            # Ma trận Covariance sẽ có kích thước 256x256 = 65,536 chiều
            self.fc_projection = nn.Linear(256 * 256, embedding_size, bias=False)
        else:
            self.pool = GeMPooling(p=3.0)
            self.fc_projection = nn.Linear(in_channels, embedding_size, bias=False)

        self.bn_head = nn.BatchNorm1d(embedding_size)
        self.fc_ce = nn.Linear(embedding_size, num_classes, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels=None):
        feat = self.features(x)
        
        # Xử lý rẽ nhánh Pooling
        if self.pooling_type == 'mpncov':
            feat = self.mpncov_bottleneck(feat) # Nén xuống 256 channels
            pooled_feat = self.pool(feat).view(feat.size(0), -1) 
        else:
            pooled_feat = self.pool(feat).view(feat.size(0), -1) 
        
        embedding = self.bn_head(self.fc_projection(pooled_feat))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            weight_norm = F.normalize(self.fc_ce.weight, p=2, dim=1)
            cosine_sce = F.linear(norm_embedding, weight_norm)
            logits_sce = cosine_sce * self.s
            
            cosine = F.linear(norm_embedding, F.normalize(self.weight))
            cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7) 
            
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m 
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            
            logits_csce = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            logits_csce *= self.s
            
            return logits_sce, logits_csce, norm_embedding
        
        return norm_embedding