import timm
from torchvision import models

def verify_backbones(backbone_string):
    # Tách chuỗi thành danh sách và xóa khoảng trắng thừa
    backbone_list = [b.strip() for b in backbone_string.split(',')]
    
    # Lấy toàn bộ danh sách tên model hợp lệ từ timm
    all_timm_models = timm.list_models()
    
    timm_found = []
    torchvision_found = []
    not_found = []

    print(f"🔍 ĐANG KIỂM TRA {len(backbone_list)} BACKBONES...\n" + "-"*50)

    for bb in backbone_list:
        if bb in all_timm_models:
            timm_found.append(bb)
            print(f"✅ [timm]        : {bb}")
        elif hasattr(models, bb):
            torchvision_found.append(bb)
            print(f"✅ [torchvision] : {bb}")
        else:
            not_found.append(bb)
            print(f"❌ [NOT FOUND]   : {bb}")

    # BÁO CÁO TỔNG KẾT
    print("\n" + "="*50)
    print(f"📊 BÁO CÁO KIỂM TRA")
    print("="*50)
    print(f"✔️ Hợp lệ trong timm        : {len(timm_found)}")
    print(f"✔️ Hợp lệ trong torchvision : {len(torchvision_found)}")
    print(f"❌ KHÔNG TỒN TẠI            : {len(not_found)}")
    
    if len(not_found) > 0:
        print("\n⚠️ CẢNH BÁO: Các model sau bị sai tên hoặc không được hỗ trợ:")
        for missing in not_found:
            print(f"  - {missing}")
        print("\n💡 Gợi ý: Dùng lệnh `timm.list_models('*từ_khóa*')` để tìm tên đúng.")
    else:
        print("\n🚀 TẤT CẢ ĐỀU HỢP LỆ! BẠN CÓ THỂ BẮT ĐẦU TRAINING.")

if __name__ == "__main__":
    # Đã gộp sẵn 15 Teachers và 15 Students chốt lại từ nãy đến giờ vào đây cho bạn test
    TEACHERS = "resnet101, densenet161, seresnet101, resnest101e, tresnet_l, efficientnet_b5, efficientnetv2_l, nfnet_l0, convnext_base, convnext_large, convnextv2_base, swin_base_patch4_window12_384, swin_large_patch4_window12_384, maxvit_base_tf_384, coatnet_2_rw_224"
    
    STUDENTS = "squeezenet1_1, resnet18, densenet121, mobilenetv2_100, mobilenetv3_large_100, shufflenet_v2_x1_0, ghostnet_100, regnety_004, efficientnet_b0, efficientnet_b1, repvgg_a0, convnext_atto, convnextv2_atto, convnext_femto, mobilevit_s"
    
    ALL_MODELS = TEACHERS + ", " + STUDENTS
    
    verify_backbones(ALL_MODELS)