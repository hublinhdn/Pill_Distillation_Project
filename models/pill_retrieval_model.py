import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
import timm
import os, sys
# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from middle.pooling import GeMPooling, MPNCOV


class PillRetrievalModel(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet50', pooling_type='gem', embedding_size=512, s=64.0, m=0.35):
        super(PillRetrievalModel, self).__init__()
        self.s = s 
        self.m = m 
        self.pooling_type = pooling_type.lower()
        

        # ==========================================
        # 🚀 SỬ DỤNG TIMM ĐỂ TẠO BACKBONE ĐỘNG
        # ==========================================
        try:
            # Tạo model không có lớp phân loại (num_classes=0) 
            # và không có Pooling mặc định (global_pool='') để giữ nguyên 4D Feature Map (B, C, H, W)
            self.features = timm.create_model(
                backbone_type,
                pretrained=True,
                num_classes=0,
                global_pool='' 
            )
            
            # (Tùy chọn) Thử chỉnh Stride=16 cho họ ResNet/MobileNet để feature map to hơn giống code cũ của bạn
            if 'resnet' in backbone_type or 'mobilenet' in backbone_type:
                try:
                    self.features = timm.create_model(backbone_type, pretrained=True, num_classes=0, global_pool='', output_stride=16)
                except:
                    pass # Nếu mạng không hỗ trợ đổi stride thì bỏ qua, dùng bản gốc
                    
        except Exception as e:
            raise ValueError(f"❌ Không khởi tạo được '{backbone_type}'. Hãy đảm bảo tên này có trong thư viện timm. Lỗi: {e}")

        # Tự động trích xuất kênh bằng Dummy Tensor (Phần này bạn viết quá chuẩn, giữ nguyên!)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_feat = self.features(dummy_input)
            in_channels = dummy_feat.shape[1] 
            
        print(f"✅ Khởi tạo [{backbone_type}] - Channels: {in_channels} - Pooling: {self.pooling_type.upper()}")





        # # --- BACKBONE SELECTION ---
        # if backbone_type == 'convnext_large':
        #     base = models.convnext_large(weights='DEFAULT')
        #     self.features = base.features
        # elif backbone_type == 'convnext_base':
        #     base = models.convnext_base(weights='DEFAULT')
        #     self.features = base.features
        # elif backbone_type == 'resnet101':
        #     base = models.resnet101(weights='DEFAULT')
        #     base.layer4[0].conv2.stride = (1, 1)
        #     base.layer4[0].downsample[0].stride = (1, 1)
        #     self.features = nn.Sequential(*list(base.children())[:-2])
        # elif backbone_type == 'mobilenet_v3_large':
        #     base = models.mobilenet_v3_large(weights='DEFAULT')
        #     self.features = base.features
        # elif backbone_type == 'efficientnet_b0':
        #     base = models.efficientnet_b0(weights='DEFAULT')
        #     self.features = base.features
        # elif backbone_type == 'resnet18':
        #     base = models.resnet18(weights='DEFAULT')
        #     base.layer4[0].conv1.stride = (1, 1) 
        #     base.layer4[0].downsample[0].stride = (1, 1)
        #     self.features = nn.Sequential(*list(base.children())[:-2])
        # else: 
        #     base = models.resnet50(weights='DEFAULT')
        #     base.layer4[0].conv2.stride = (1, 1)
        #     base.layer4[0].downsample[0].stride = (1, 1)
        #     self.features = nn.Sequential(*list(base.children())[:-2])

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