import torch
import segmentation_models_pytorch as smp

def create_model(device="cuda"):
    """
    김경민님의 계획서 7장(U-Net & ResNet-50)에 명시된
    베이스라인 모델을 생성합니다.
    """
    
    # smp 라이브러리를 사용해 U-Net 모델을 정의합니다.
    model = smp.Unet(
        encoder_name="resnet50",        # 인코더(Backbone)로 ResNet-50 사용 
        encoder_weights="imagenet",   # ImageNet 사전 학습 가중치 사용 
        in_channels=3,                # 입력 이미지 채널 (RGB=3)
        classes=7                     # 최종 출력 클래스 개수 (우리가 7개로 정의함)
    )
    
    # 모델을 GPU(RTX 5070 Ti)로 보냅니다.
    model = model.to(device)
    
    return model

if __name__ == "__main__":
    # --- model.py 직접 실행 테스트 ---
    print("--- 모델 생성 및 GPU 호환성 테스트 ---")
    
    # 1. GPU(CUDA) 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용할 장치(Device): {device}")
    
    if device == "cpu":
        print("[경고] CUDA를 사용할 수 없습니다. PyTorch 설치를 확인하세요.")
        print("       (지난번처럼 GPU 경고가 뜬다면 알려주세요)")

    try:
        # 2. 모델 생성
        model = create_model(device)
        print("모델 생성 성공. (smp.Unet with resnet50)")
        
        # 3. 더미(dummy) 입력 텐서로 테스트
        # (BatchSize=1, Channels=3, Height=512, Width=512)
        # 우리 데이터 로더의 출력 크기와 동일하게 맞춤
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        print(f"더미 입력 텐서 생성: {dummy_input.shape}")
        
        # 4. 모델 추론(Forward pass) 실행
        with torch.no_grad(): # 추론 시에는 gradient 계산 불필요
            output = model(dummy_input)
            
        print("\n--- 모델 테스트 성공! ---")
        print(f"모델 출력 텐서 모양: {output.shape}")
        print("[정상] 출력 모양이 (1, 7, 512, 512) 여야 합니다.")
        
    except Exception as e:
        print(f"\n[오류] 모델 테스트 중 오류 발생: {e}")