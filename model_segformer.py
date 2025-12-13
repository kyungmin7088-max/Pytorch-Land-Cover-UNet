import torch
import segmentation_models_pytorch as smp

def create_model(device="cuda"):
    """
    [Final Weapon] SegFormer (MiT-B3)
    - CNN이 아닌 Transformer 기반 모델입니다.
    - 'Noisy Label'(잘못된 정답)을 무시하고 이미지의 '진짜 문맥'을 파악하는 능력이 탁월합니다.
    """
    print("Creating SegFormer with MiT-B3 backbone...")
    
    model = smp.Segformer(
        encoder_name="mit_b3",      # Vision Transformer (MiT) 백본
        encoder_weights="imagenet", # 사전 학습 가중치
        in_channels=3,
        classes=7,
        decoder_segmentation_head_dim=256 # SegFormer 전용 헤드 설정
    )
    
    return model.to(device)

if __name__ == "__main__":
    # 테스트 코드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = create_model(device)
        print("SegFormer (MiT-B3) 모델 생성 성공!")
        
        x = torch.randn(1, 3, 512, 512).to(device)
        y = model(x)
        print(f"출력 크기 확인: {y.shape}") # (1, 7, 128, 128) -> 내부적으로 4배 Upsample됨
    except Exception as e:
        print(f"오류 발생: {e}")
        print("Tip: pip install -U segmentation-models-pytorch 로 라이브러리를 최신화해보세요.")