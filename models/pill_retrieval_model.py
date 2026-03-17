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
        # 🚀 CƠ CHẾ KHỞI TẠO ĐỘNG (TIMM + TORCHVISION FALLBACK)
        # ==========================================
        try:
            # ƯU TIÊN 1: Tìm trong thư viện timm (Dành cho các mạng hiện đại)
            self.features = timm.create_model(
                backbone_type,
                pretrained=True,
                num_classes=0,
                global_pool='' 
            )
            
            # (Tùy chọn) Chỉnh Stride=16 cho họ ResNet/MobileNet trong timm
            if 'resnet' in backbone_type or 'mobilenet' in backbone_type:
                try:
                    self.features = timm.create_model(backbone_type, pretrained=True, num_classes=0, global_pool='', output_stride=16)
                except:
                    pass
                    
        except Exception as e_timm:
            print(f"⚠️ timm không có '{backbone_type}', đang tự động chuyển hướng tìm trong torchvision...")
            
            # ƯU TIÊN 2: Fallback sang torchvision (Dành cho SqueezeNet, DenseNet...)
            try:
                # Gọi động hàm từ torchvision.models (ví dụ: models.squeezenet1_1)
                base_model_func = getattr(models, backbone_type)
                base = base_model_func(weights='DEFAULT')
                
                # Tách phần Features Extractors tùy theo cấu trúc của torchvision
                if hasattr(base, 'features'):
                    # Dành cho SqueezeNet, DenseNet, MobileNet, VGG...
                    self.features = base.features
                else:
                    # Dành cho họ ResNet truyền thống
                    self.features = nn.Sequential(*list(base.children())[:-2])
                    
            except AttributeError:
                raise ValueError(f"❌ TÊN KHÔNG HỢP LỆ! Không tìm thấy '{backbone_type}' ở cả timm và torchvision.\nLỗi timm: {e_timm}")
            except Exception as e_tv:
                raise ValueError(f"❌ Khởi tạo thất bại từ torchvision. Lỗi: {e_tv}")

        # Tự động trích xuất kênh bằng Dummy Tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_feat = self.features(dummy_input)
            in_channels = dummy_feat.shape[1] 
            
        print(f"✅ Khởi tạo [{backbone_type}] thành công! - Channels: {in_channels} - Pooling: {self.pooling_type.upper()}")

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