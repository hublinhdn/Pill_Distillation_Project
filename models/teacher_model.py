import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MPNCOV(nn.Module):
    """
    Matrix Power Normalization (MPN-COV) 
    Thực hiện tính căn bậc hai ma trận bằng lặp Newton-Schulz để khử nhiễu đặc trưng.
    """
    def __init__(self, iterNum=3):
        super(MPNCOV, self).__init__()
        self.iterNum = iterNum

    def forward(self, x):
        batchSize, channels, h, w = x.data.shape
        M = h * w
        
        # 1. Tính ma trận hiệp phương sai (Covariance Matrix)
        x = x.view(batchSize, channels, M)
        I_hat = (-1.0/M) * torch.ones(M, M, device=x.device) + torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2)) / M

        # 2. Trace normalization giúp hội tụ ổn định
        trY = y.diagonal(dim1=-2, dim2=-1).sum(1)
        y = y / trY.view(batchSize, 1, 1)

        # 3. Lặp Newton-Schulz (Y_{k+1} = 0.5 * Y_k * (3I - Z_k * Y_k))
        I = torch.eye(channels, channels, device=x.device).view(1, channels, channels).repeat(batchSize, 1, 1)
        Y = y
        Z = I
        for i in range(self.iterNum):
            ZY = Z.bmm(Y)
            Y = 0.5 * Y.bmm(3.0 * I - ZY)
            Z = 0.5 * (3.0 * I - ZY).bmm(Z)
        
        # Denormalize theo Trace ban đầu
        out = Y * torch.sqrt(trY).view(batchSize, 1, 1)
        return out

class PillTeacher(nn.Module):
    def __init__(self, num_classes, embedding_size=512, backbone_type='resnet50', pooling_type='mpn-cov'):
        """
        pooling_type: 'mpn-cov' (mặc định) hoặc 'bcnn' (Bilinear CNN kiểu cũ)
        """
        super(PillTeacher, self).__init__()
        self.pooling_type = pooling_type
        
        # 1. Khởi tạo Backbone (Features Extractor)
        if backbone_type == 'resnet50':
            base = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone_type} chưa được hỗ trợ.")

        # 2. Bottleneck Layer (Giảm chiều sâu để tránh bùng nổ tham số khi nhân ma trận)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 3. Pooling Head
        if self.pooling_type == 'mpn-cov':
            self.pool_layer = MPNCOV(iterNum=3)
            # MPN-COV trả về ma trận đối xứng, duỗi phẳng là 256*256
            fc_input_dim = 256 * 256 
        else:
            # BCNN truyền thống
            fc_input_dim = 256 * 256

        self.fc_projection = nn.Linear(fc_input_dim, embedding_size)
        self.bn_head = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(p=0.5) # Thêm Dropout để tránh Overfitting như paper gợi ý

        # 4. Classification & Metric Heads
        self.fc_ce = nn.Linear(embedding_size, num_classes)
        self.proxy_cos = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxy_cos)

    def forward(self, x, labels=None):
        # Trích xuất đặc trưng spatial
        feat = self.features(x) 
        feat = self.reduce_conv(feat) # [B, 256, H, W]
        
        batch_size, channels, h, w = feat.size()

        if self.pooling_type == 'mpn-cov':
            # Matrix Square Root Pooling
            pooled_feat = self.pool_layer(feat) 
        else:
            # Simple Bilinear Pooling (X * X^T)
            feat_flat = feat.view(batch_size, channels, h * w)
            pooled_feat = torch.bmm(feat_flat, feat_flat.transpose(1, 2)) / (h * w)
            # Element-wise Sign-sqrt normalization
            pooled_feat = torch.sign(pooled_feat) * torch.sqrt(torch.abs(pooled_feat) + 1e-8)
        
        # Vectorize và Projection
        flat_feat = pooled_feat.view(batch_size, -1)
        flat_feat = F.normalize(flat_feat, p=2, dim=1) # Chuẩn hóa L2 trước FC
        
        embedding = self.bn_head(self.fc_projection(self.dropout(flat_feat)))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            logits_sce = self.fc_ce(embedding)
            norm_proxy = F.normalize(self.proxy_cos, p=2, dim=1)
            logits_csce = F.linear(norm_embedding, norm_proxy) * 16.0
            return logits_sce, logits_csce, norm_embedding
        
        return norm_embedding