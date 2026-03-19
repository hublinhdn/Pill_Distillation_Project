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
        # 🚀 CƠ CHẾ KHỞI TẠO ĐỘNG (HỖ TRỢ ABLATION STUDY _TV / _TIMM)
        # ==========================================
        backbone_lower = backbone_type.lower()
        
        # 1. Bắt hậu tố điều hướng
        force_tv = False
        force_timm = False
        
        if backbone_lower.endswith('_tv'):
            force_tv = True
            backbone_lower = backbone_lower.replace('_tv', '')
        elif backbone_lower.endswith('_timm'):
            force_timm = True
            backbone_lower = backbone_lower.replace('_timm', '')
            
        # Chuẩn hóa tên cho Torchvision (nếu cần dùng)
        tv_name = backbone_lower
        if 'mobilenetv3' in tv_name: tv_name = 'mobilenet_v3_large'
        elif 'mobilenetv2' in tv_name: tv_name = 'mobilenet_v2'
        elif 'efficientnetv2' in tv_name: tv_name = tv_name.replace('efficientnetv2', 'efficientnet_v2')

        try:
            # ---------------------------------------------------------
            # 🛑 NHÁNH 1: ÉP BUỘC DÙNG TORCHVISION (Khi có đuôi _tv)
            # ---------------------------------------------------------
            if force_tv:
                print(f"🌟 THỰC NGHIỆM: Ép load {backbone_lower} bằng TORCHVISION")
                base_model_func = getattr(models, tv_name)
                base = base_model_func(weights='DEFAULT')
                
                # Áp dụng Hack Stride cho ResNet thuần
                if 'resnet' in tv_name and not any(x in tv_name for x in ['seresnet', 'resnest', 'tresnet']):
                    if tv_name in ['resnet18', 'resnet34']:
                        base.layer4[0].conv1.stride = (1, 1)
                    else:
                        base.layer4[0].conv2.stride = (1, 1)
                    base.layer4[0].downsample[0].stride = (1, 1)
                    self.features = nn.Sequential(*list(base.children())[:-2])
                elif hasattr(base, 'features'):
                    self.features = base.features
                else:
                    self.features = nn.Sequential(*list(base.children())[:-2])

            # ---------------------------------------------------------
            # 🛑 NHÁNH 2: ÉP BUỘC DÙNG TIMM (Khi có đuôi _timm)
            # ---------------------------------------------------------
            elif force_timm:
                print(f"⚡ THỰC NGHIỆM: Ép load {backbone_lower} bằng TIMM")
                # CHỈ áp dụng Dilation cho ResNet, ResNeSt, SEResNet. TUYỆT ĐỐI KHÔNG ép TResNet hay MobileNet
                if any(x in backbone_lower for x in ['resnet', 'resnest', 'seresnet']) and not any(x in backbone_lower for x in ['tresnet', 'convnext', 'mobilenet']):
                    self.features = timm.create_model(backbone_lower, pretrained=True, num_classes=0, global_pool='', output_stride=16)
                else:
                    self.features = timm.create_model(backbone_lower, pretrained=True, num_classes=0, global_pool='')

            # ---------------------------------------------------------
            # 🌊 NHÁNH 3: AUTO-ROUTING (Không có hậu tố - Chạy tối ưu như cũ)
            # ---------------------------------------------------------
            else:
                if backbone_lower in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
                    print(f"🌟 AUTO: Load {backbone_lower} bằng TORCHVISION (Hack Stride=1)")
                    base_model_func = getattr(models, backbone_lower)
                    base = base_model_func(weights='DEFAULT')
                    if backbone_lower in ['resnet18', 'resnet34']:
                        base.layer4[0].conv1.stride = (1, 1)
                    else:
                        base.layer4[0].conv2.stride = (1, 1)
                    base.layer4[0].downsample[0].stride = (1, 1)
                    self.features = nn.Sequential(*list(base.children())[:-2])

                elif any(x in backbone_lower for x in ['mobilenet', 'efficientnet']):
                    print(f"🌟 AUTO: Load {tv_name} bằng TORCHVISION (Lấy weight classic)")
                    try:
                        base_model_func = getattr(models, tv_name)
                        base = base_model_func(weights='DEFAULT')
                        self.features = base.features
                    except Exception as e:
                        self.features = timm.create_model(backbone_lower, pretrained=True, num_classes=0, global_pool='')

                else:
                    # LƯỚI AN TOÀN ĐÃ ĐƯỢC THÊM LẠI Ở ĐÂY
                    try:
                        if any(x in backbone_lower for x in ['resnet', 'resnest', 'seresnet']) and not any(x in backbone_lower for x in ['tresnet', 'convnext', 'mobilenet']):
                            print(f"⚡ AUTO: Load {backbone_lower} bằng TIMM (Output Stride 16)")
                            self.features = timm.create_model(backbone_lower, pretrained=True, num_classes=0, global_pool='', output_stride=16)
                        else:
                            print(f"⚡ AUTO: Load {backbone_lower} bằng TIMM (Mặc định)")
                            self.features = timm.create_model(backbone_lower, pretrained=True, num_classes=0, global_pool='')
                    
                    except Exception as e_timm:
                        print(f"⚠️ TIMM không chứa '{backbone_lower}'. Tự động tìm kiếm dự phòng trong TORCHVISION...")
                        try:
                            # Cứu cánh cho SqueezeNet, ShuffleNet...
                            base_model_func = getattr(models, backbone_lower)
                            base = base_model_func(weights='DEFAULT')
                            if hasattr(base, 'features'):
                                self.features = base.features
                            else:
                                self.features = nn.Sequential(*list(base.children())[:-2])
                            print(f"🌟 SỬ DỤNG TORCHVISION: Đã load thành công {backbone_lower}")
                        except Exception as e_tv:
                            raise ValueError(f"Hoàn toàn không tìm thấy {backbone_lower}. Lỗi TIMM: {e_timm} | Lỗi TV: {e_tv}")

        except Exception as e:
            raise ValueError(f"❌ KHÔNG TÌM THẤY {backbone_type} hoặc lỗi khởi tạo. Chi tiết: {e}")

        # Bật chế độ Eval để tránh lỗi BatchNorm với Batch Size = 1 của họ ResNeSt
        self.features.eval()
        # Tự động trích xuất số lượng kênh bằng Dummy Tensor (Size 384x384 để an toàn cho Swin/MaxViT)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 384, 384)
            dummy_feat = self.features(dummy_input)
            in_channels = dummy_feat.shape[1] 
        
        # Trả lại chế độ Train bình thường
        self.features.train()
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