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
    def __init__(self, num_classes, embedding_size=512):
        super(PillTeacher, self).__init__()
        base = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        
        # Upsize ngầm: Giữ Resolution layer 4 là 14x14
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)
        
        self.features = nn.Sequential(*list(base.children())[:-2])
        
        # Bottleneck giảm chiều channel trước MPN-COV
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.mpn_cov = MPNCOV(iterNum=3)
        
        # Benchmark thường dùng BN sau FC để ổn định vector embedding
        self.fc_projection = nn.Linear(256 * 256, embedding_size, bias=False)
        self.bn_head = nn.BatchNorm1d(embedding_size)
        
        # Classifier với Scale chuẩn benchmark
        self.fc_ce = nn.Linear(embedding_size, num_classes)
        self.proxy_cos = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxy_cos)

    def forward(self, x, labels=None):
        feat = self.features(x)
        feat = self.reduce_conv(feat)
        
        matrix_sqrt = self.mpn_cov(feat)
        flat_feat = matrix_sqrt.view(matrix_sqrt.size(0), -1)
        
        # L2 Normalize trước khi chiếu vào không gian embedding
        flat_feat = F.normalize(flat_feat, p=2, dim=1)
        embedding = self.bn_head(self.fc_projection(flat_feat))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            logits_sce = self.fc_ce(embedding)
            norm_proxy = F.normalize(self.proxy_cos, p=2, dim=1)
            # Scale 32-64 tùy theo độ khó của batch
            logits_csce = F.linear(norm_embedding, norm_proxy) * 32.0
            return logits_sce, logits_csce, norm_embedding
        
        return norm_embedding