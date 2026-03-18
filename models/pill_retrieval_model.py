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
        # 🚀 CƠ CHẾ KHỞI TẠO ĐỘNG (BẢN CHỐT THỰC NGHIỆM)
        # ==========================================
        backbone_lower = backbone_type.lower()
        
        # ---------------------------------------------------------
        # 👑 ĐƯỜNG ƯU TIÊN 1: TRẢ LẠI BẢN HACK CỦA BẠN CHO RESNET
        # (Không dùng Dilation để tránh lọt khe mất chữ dập)
        # ---------------------------------------------------------
        if backbone_lower in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            print(f"🌟 SỬ DỤNG TORCHVISION: Đang load {backbone_type} với Stride=1 thủ công (Dense Sampling)")
            base_model_func = getattr(models, backbone_lower)
            base = base_model_func(weights='DEFAULT')
            
            if backbone_lower in ['resnet18', 'resnet34']:
                base.layer4[0].conv1.stride = (1, 1)
            else:
                base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            
            self.features = nn.Sequential(*list(base.children())[:-2])

        # ---------------------------------------------------------
        # 👑 ĐƯỜNG ƯU TIÊN 2: BẢO TOÀN LỚP FEATURES CHO MOBILENET
        # ---------------------------------------------------------
        elif backbone_lower in ['mobilenet_v3_large', 'mobilenetv3_large_100', 'mobilenet_v2', 'mobilenetv2_100']:
            # Xử lý đồng nhất tên gọi cho torchvision
            tv_name = 'mobilenet_v3_large' if 'v3' in backbone_lower else 'mobilenet_v2'
            print(f"🌟 SỬ DỤNG TORCHVISION: Đang load {tv_name} để giữ nguyên khối đặc trưng")
            base_model_func = getattr(models, tv_name)
            base = base_model_func(weights='DEFAULT')
            self.features = base.features

        # ---------------------------------------------------------
        # ⚡ ĐƯỜNG CHUNG CHO TIMM (CÁC MẠNG HIỆN ĐẠI/TRANSFORMER)
        # ---------------------------------------------------------
        else:
            try:
                # Họ ResNeSt, SEResNet vẫn có thể thử Dilation xem sao (nếu rớt mAP, ta sẽ đưa lên nhóm trên sau)
                if any(x in backbone_lower for x in ['resnest', 'seresnet', 'tresnet']):
                    print(f"⚡ DÙNG TIMM: Load {backbone_type} với output_stride=16")
                    self.features = timm.create_model(backbone_lower, pretrained=True, num_classes=0, global_pool='', output_stride=16)
                else:
                    print(f"⚡ DÙNG TIMM: Load {backbone_type} với cấu trúc mặc định")
                    self.features = timm.create_model(backbone_lower, pretrained=True, num_classes=0, global_pool='')
                    
            except Exception as e_timm:
                print(f"⚠️ TIMM không chứa '{backbone_type}'. Tìm kiếm dự phòng trong Torchvision...")
                try:
                    # Dự phòng cho DenseNet, SqueezeNet...
                    base_model_func = getattr(models, backbone_lower)
                    base = base_model_func(weights='DEFAULT')
                    if hasattr(base, 'features'):
                        self.features = base.features
                    else:
                        self.features = nn.Sequential(*list(base.children())[:-2])
                except Exception as e_tv:
                    raise ValueError(f"❌ KHÔNG TÌM THẤY '{backbone_type}'. Lỗi chi tiết: {e_tv}")

        # Tự động trích xuất số lượng kênh bằng Dummy Tensor (Size 384x384 để an toàn cho Swin/MaxViT)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 384, 384)
            dummy_feat = self.features(dummy_input)
            in_channels = dummy_feat.shape[1] 
            
        print(f"✅ HOÀN TẤT KHỞI TẠO! Channels: {in_channels} | Pooling: {self.pooling_type.upper()}")

        # ==========================================
        # 🧠 POOLING & PROJECTION HEAD
        # ==========================================
        if self.pooling_type == 'mpncov':
            self.mpncov_bottleneck = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
            self.pool = MPNCOV(iterNum=3)
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
        
        if self.pooling_type == 'mpncov':
            feat = self.mpncov_bottleneck(feat) 
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