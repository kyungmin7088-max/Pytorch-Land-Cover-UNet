import torch
import segmentation_models_pytorch as smp

def create_model(device="cuda"):
    """
    계획서 7장(DeepLabV3+)에 명시된 최종 목표 모델을 생성합니다.
    - Backbone: ResNet-50 
    - Architecture: DeepLabV3+ (ASPP 포함)
    """
    
    # smp 라이브러리를 사용해 DeepLabV3+ 모델 정의
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",        # 인코더: ResNet-50
        encoder_weights="imagenet",   # 사전 학습 가중치 사용
        in_channels=3,                # 입력: RGB (3채널)
        classes=7                     # 출력: 7개 클래스
    )
    
    model = model.to(device)
    return model

if __name__ == "__main__":
    # --- 테스트 코드 ---
    print("--- DeepLabV3+ 모델 생성 테스트 ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = create_model(device)
        print(f"모델 생성 성공: DeepLabV3+ (ResNet-50) on {device}")
        
        # 더미 입력으로 출력 형태 확인
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"출력 텐서 모양: {output.shape}")
        # 예상: [1, 7, 512, 512]
        
    except Exception as e:
        print(f"오류 발생: {e}")