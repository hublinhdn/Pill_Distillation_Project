import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

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

class PillTeacher(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet101', embedding_size=512, s=32.0, m=0.50):
        super(PillTeacher, self).__init__()
        self.s = s 
        self.m = m 
        
        # Backbone Selection
        if backbone_type == 'resnet101':
            base = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
            base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        elif backbone_type == 'convnext_base':
            base = models.convnext_base(weights='ConvNeXt_Base_Weights.IMAGENET1K_V1')
            self.features = base.features
            in_channels = 1024
        else:
            base = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
            base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.mpn_cov = MPNCOV(iterNum=3)
        self.fc_projection = nn.Linear(256 * 256, embedding_size, bias=False)
        self.bn_head = nn.BatchNorm1d(embedding_size)

        # Classifier Heads
        self.fc_ce = nn.Linear(embedding_size, num_classes) 
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # ArcFace constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels=None):
        feat = self.features(x)
        feat = self.reduce_conv(feat)
        matrix_sqrt = self.mpn_cov(feat)
        flat_feat = matrix_sqrt.view(matrix_sqrt.size(0), -1)
        
        # Normalize and Project
        flat_feat = F.normalize(flat_feat, p=2, dim=1)
        embedding = self.bn_head(self.fc_projection(flat_feat))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            # 1. Nhánh SCE
            logits_sce = self.fc_ce(embedding)
            
            # 2. Nhánh ArcFace (CSCE)
            cosine = F.linear(norm_embedding, F.normalize(self.weight))
            cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7) # Ổn định số học
            
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m 
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            
            logits_csce = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            logits_csce *= self.s
            
            return logits_sce, logits_csce, norm_embedding
        
        return norm_embedding