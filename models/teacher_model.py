import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MPNCOV(nn.Module):
    """Lớp tính toán ma trận hiệp phương sai để bắt đặc trưng chi tiết (Fine-grained)"""
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
        out = Y * torch.sqrt(trY).view(batchSize, 1, 1)
        return out

class PillTeacher(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=5000, embedding_size=512):
        super(PillTeacher, self).__init__()
        
        self.backbone_name = backbone_name
        print(f"🛠️ Đang khởi tạo Teacher với Backbone: {backbone_name}")

        # 1. LỰA CHỌN BACKBONE
        if backbone_name == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.in_channels = 2048
        
        elif backbone_name == 'resnet101':
            base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.in_channels = 2048

        elif backbone_name == 'resnext101':
            base = models.resnext101_32x8d(weights=models.ResNext101_32X8D_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.in_channels = 2048

        elif backbone_name == 'convnext_base':
            base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.in_channels = 1024

        elif backbone_name == 'efficientnet_v2_s':
            base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.in_channels = 1280
            
        else:
            raise ValueError(f"Chưa hỗ trợ backbone: {backbone_name}")

        # 2. CẤU TRÚC HẬU BACKBONE (CỐ ĐỊNH ĐỂ SO SÁNH CÔNG BẰNG)
        # Nén về 256 channels trước khi vào MPN-COV để giảm tải tính toán
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.mpn_cov = MPNCOV(iterNum=3)
        
        # FC Projection & BN Head (Nơi lấy vector trích xuất tri thức cho KD)
        self.fc_projection = nn.Linear(256 * 256, embedding_size, bias=False)
        self.bn_head = nn.BatchNorm1d(embedding_size)
        
        # Classifiers
        self.fc_ce = nn.Linear(embedding_size, num_classes)
        self.proxy_cos = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxy_cos)

    def forward(self, x, labels=None):
        # Feature Extraction
        x = self.features(x)
        
        # Nếu là ConvNeXt hoặc EfficientNet, đôi khi output cần điều chỉnh lại shape 
        # nhưng lớp .features của torchvision thường trả về đúng (B, C, H, W)
        x = self.reduce_conv(x)
        
        # MPN-COV
        matrix_sqrt = self.mpn_cov(x)
        flat_feat = matrix_sqrt.view(matrix_sqrt.size(0), -1)
        
        # Embedding
        flat_feat = F.normalize(flat_feat, p=2, dim=1)
        embedding = self.bn_head(self.fc_projection(flat_feat))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            logits_sce = self.fc_ce(embedding)
            logits_cos = F.linear(norm_embedding, F.normalize(self.proxy_cos, p=2, dim=1))
            return norm_embedding, logits_sce, logits_cos
        
        return norm_embedding