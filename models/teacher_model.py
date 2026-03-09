import torch
import torch.nn as nn
import torch.nn.functional as F
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
        Y = y
        Z = I
        for i in range(self.iterNum):
            ZY = Z.bmm(Y)
            Y = 0.5 * Y.bmm(3.0 * I - ZY)
            Z = 0.5 * (3.0 * I - ZY).bmm(Z)
        out = Y * torch.sqrt(trY).view(batchSize, 1, 1)
        return out

class PillTeacher(nn.Module):
    def __init__(self, num_classes, embedding_size=512, backbone_type='resnet50'):
        super(PillTeacher, self).__init__()
        
        # Load ResNet50 chuẩn
        base = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        
        # CHIẾN LƯỢC UPSIZE NGẦM: 
        # Chỉnh layer4 (khối cuối) để stride=1 thay vì stride=2
        # Điều này giúp output feature map là 14x14 thay vì 7x7
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)
        
        self.features = nn.Sequential(*list(base.children())[:-2])
        in_channels = 2048

        # Bottleneck để giảm channel xuống 256 trước khi vào MPN-COV
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.mpn_cov = MPNCOV(iterNum=3)
        self.fc_projection = nn.Linear(256 * 256, embedding_size)
        self.bn_head = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(p=0.5)

        self.fc_ce = nn.Linear(embedding_size, num_classes)
        self.proxy_cos = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxy_cos)

    def forward(self, x, labels=None):
        feat = self.features(x) # Output lúc này là [B, 2048, 14, 14]
        feat = self.reduce_conv(feat) # [B, 256, 14, 14]
        
        matrix_sqrt = self.mpn_cov(feat) # [B, 256, 256]
        flat_feat = matrix_sqrt.view(matrix_sqrt.size(0), -1)
        flat_feat = F.normalize(flat_feat, p=2, dim=1)
        
        embedding = self.bn_head(self.fc_projection(self.dropout(flat_feat)))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            logits_sce = self.fc_ce(embedding)
            norm_proxy = F.normalize(self.proxy_cos, p=2, dim=1)
            logits_csce = F.linear(norm_embedding, norm_proxy) * 16.0
            return logits_sce, logits_csce, norm_embedding
        
        return norm_embedding