import torch
import segmentation_models_pytorch as smp

def create_model(device="cuda"):
    """
    [Upgrade] Backbone을 ResNet-50에서 EfficientNet-B4로 교체
    - EfficientNet은 더 적은 연산량으로 더 높은 정확도를 냅니다.
    - DeepLabV3+ 구조는 유지하되, 눈(Encoder)을 더 좋은 것으로 바꿉니다.
    """
    print("Creating DeepLabV3+ with EfficientNet-B4 backbone...")
    
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4", # 핵심 변경 사항! (b0 ~ b7 중 b4가 성능/속도 균형 최적)
        encoder_weights="imagenet",     # ImageNet 사전학습 가중치 사용
        in_channels=3,
        classes=7
    )
    
    return model.to(device)

if __name__ == "__main__":
    # 테스트 코드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = create_model(device)
        print("모델 생성 성공! (EfficientNet-B4)")
        
        # 더미 데이터로 테스트
        x = torch.randn(1, 3, 512, 512).to(device)
        y = model(x)
        print(f"출력 크기 확인: {y.shape}")
    except Exception as e:
        print(f"오류 발생: {e}")