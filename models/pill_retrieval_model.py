import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
import timm
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from middle.pooling import GeMPooling, MPNCOV
from models.model_category_config import super_large_backbones, large_backbones, medium_backbones, small_backbones, pure_transformer_backbones

class PillRetrievalModel(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet50', pooling_type='gem', embedding_size=512, s=64.0, m=0.35):
        super(PillRetrievalModel, self).__init__()
        self.s = s 
        self.m = m 
        self.pooling_type = pooling_type.lower()
        
        # Xác định xem mạng có phải là Vision Transformer thuần túy hay không
        clean_name = backbone_type.lower().replace('_tv', '').replace('_timm', '')
        self.is_pure_vit = any(x in clean_name for x in pure_transformer_backbones)

        # 1. KHỞI TẠO BACKBONE
        self.features = self._build_backbone(backbone_type)

        # 2. ĐẾM KÊNH BẰNG DUMMY INPUT ĐỘNG
        self.features.eval() 
        with torch.no_grad():
            # Tự động chọn kích thước an toàn cho Patch14
            safe_size = 392 if 'patch14' in clean_name else 384
            dummy_input = torch.zeros(1, 3, safe_size, safe_size)
            dummy_out = self.features(dummy_input)
            
            # Xử lý hình dạng Tensor đầu ra
            if isinstance(dummy_out, tuple): 
                dummy_out = dummy_out[0]
                
            if dummy_out.dim() == 2: # ViT trả về (Batch, Channels)
                in_channels = dummy_out.shape[1]
            else: # CNN trả về (Batch, Channels, H, W)
                in_channels = dummy_out.shape[1]
                
        self.features.train() 
        print(f"✅ HOÀN TẤT KHỞI TẠO! Channels: {in_channels} | Pooling: {'Bypass (ViT)' if self.is_pure_vit else self.pooling_type.upper()}")

        # 3. POOLING & PROJECTION HEAD
        if self.pooling_type == 'mpncov' and not self.is_pure_vit:
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

    def _build_backbone(self, backbone_type):
        raw_name = backbone_type.lower()
        force_tv = raw_name.endswith('_tv')
        force_timm = raw_name.endswith('_timm')
        clean_name = raw_name.replace('_tv', '').replace('_timm', '')

        tv_name = clean_name
        if 'mobilenetv3' in tv_name: tv_name = 'mobilenet_v3_large'
        elif 'mobilenetv2' in tv_name: tv_name = 'mobilenet_v2'
        elif 'efficientnetv2' in tv_name: tv_name = tv_name.replace('efficientnetv2', 'efficientnet_v2')

        if force_tv:
            return self._load_torchvision(tv_name)
        elif force_timm:
            return self._load_timm(clean_name)

        if clean_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            return self._load_torchvision(clean_name)
        elif any(x in clean_name for x in ['mobilenet', 'efficientnet']):
            try: return self._load_torchvision(tv_name)
            except: return self._load_timm(clean_name)
        else:
            try: return self._load_timm(clean_name)
            except: return self._load_torchvision(clean_name)

    def _load_torchvision(self, name):
        base_model_func = getattr(models, name)
        base = base_model_func(weights='DEFAULT')
        
        if 'resnet' in name and not any(x in name for x in ['seresnet', 'resnest', 'tresnet']):
            if name in ['resnet18', 'resnet34']: base.layer4[0].conv1.stride = (1, 1)
            else: base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            return nn.Sequential(*list(base.children())[:-2])
            
        if hasattr(base, 'features'):
            return base.features
        return nn.Sequential(*list(base.children())[:-2])
    
    def _load_timm(self, name):
        # Drop path gắt hơn cho mạng siêu lớn
        drop_path = 0.4 if any(x in name for x in ['large', 'xlarge', 'efficientnetv2_l']) else 0.2
        
        # 1. Khởi tạo model từ TIMM
        if self.is_pure_vit:
            model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='token', drop_path_rate=drop_path, dynamic_img_size=True)
        elif any(x in name for x in ['resnet', 'resnest', 'seresnet']) and not any(x in name for x in ['tresnet', 'convnext', 'mobilenet']):
            model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='', output_stride=16, drop_path_rate=drop_path)
        else:
            model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='', drop_path_rate=drop_path)

        # =========================================================
        # 🚀 2. BẬT GRADIENT CHECKPOINTING CHO CẢ SUPER LARGE VÀ LARGE (TIMM)
        # =========================================================
        checkpoint_keywords = super_large_backbones + large_backbones
        if any(x in name for x in checkpoint_keywords):
            try:
                model.set_grad_checkpointing(True)
                print(f"🔥 Đã kích hoạt Gradient Checkpointing cho [{name}]. VRAM đã được nén!")
            except Exception as e:
                print(f"⚠️ [{name}] không hỗ trợ Gradient Checkpointing. Lỗi: {e}")
                
        return model

    def forward(self, x, labels=None):
        feat = self.features(x)
        
        # RẼ NHÁNH XỬ LÝ DỮ LIỆU ĐẦU RA (Xử lý mượt mà cả CNN 4D và ViT 2D)
        if feat.dim() == 2:
            # ViT đã tự gộp thành (Batch, Channels), bỏ qua Pooling
            pooled_feat = feat
        else:
            # CNN trả về 4D, đi qua Pooling bình thường
            if self.pooling_type == 'mpncov' and not self.is_pure_vit:
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