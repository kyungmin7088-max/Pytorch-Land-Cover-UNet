import torch
import segmentation_models_pytorch as smp
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- 1. 우리가 만든 스크립트 임포트 ---
from dataset import get_loader
from model import create_model

# --- 2. 하이퍼파라미터 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # [조절 가능] GPU 메모리에 따라 4, 8, 16 등으로 조절
NUM_EPOCHS = 25 # [조절 가능] 전체 데이터셋 학습 횟수
LR = 1e-4       # 학습률 (Learning Rate)
SAVE_DIR = "./checkpoints" # 모델 가중치를 저장할 폴더

# --- 3. 준비 (데이터, 모델, 손실함수, 최적화) ---

def setup():
    print("--- 훈련 준비 시작 ---")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 데이터 로더
    train_loader = get_loader('./data', 'train', BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = get_loader('./data', 'val', BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"데이터 로드 완료. Train: {len(train_loader)} 배치, Val: {len(val_loader)} 배치")

    # 2. 모델
    model = create_model(DEVICE)
    print(f"모델 생성 완료. (U-Net + ResNet-50) -> {DEVICE}")

    # 3. 손실 함수 (계획서의 DiceLoss)
    loss_fn = smp.losses.DiceLoss(
        mode=smp.losses.MULTICLASS_MODE,
        from_logits=True
    )
    print("손실 함수: DiceLoss (Multiclass)")

    # 4. 옵티마이저
    optimizer = Adam(model.parameters(), lr=LR)
    
    # 5. 성능 지표 -> evaluate 함수 내부에서 직접 계산
    print("성능 지표: IoU Score (mIoU), Accuracy (Pixel Acc) (Evaluate 함수에서 계산)")
    
    return model, train_loader, val_loader, loss_fn, optimizer

# --- 4. 훈련 (1 에포크) ---

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train() # 훈련 모드
    total_loss = 0.0
    
    for images, masks in tqdm(loader, desc="Train Epoch"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad() 
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

# --- 5. 평가 (1 에포크) ---
#
# === [버그 수정됨 v9] ===
# get_stats()가 CPU로 반환한 텐서(tp, fp, fn, tn)를
# .to(device)를 사용해 CUDA로 이동시킨 후, 
# CUDA에 있는 total_tp/fp/fn/tn에 더합니다.
#
def evaluate(model, loader, loss_fn, device):
    model.eval() # 평가 모드
    
    total_loss = 0.0
    # [C] (num_classes) 모양으로 CUDA에 초기화
    total_tp = torch.zeros(7).to(device)
    total_fp = torch.zeros(7).to(device)
    total_fn = torch.zeros(7).to(device)
    total_tn = torch.zeros(7).to(device)

    with torch.no_grad(): # 기울기 계산 비활성화
        for images, masks in tqdm(loader, desc="Validate Epoch"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images) # Shape: [B, 7, H, W] (logits)
            
            # 1. 손실 계산 (손실 함수는 logits를 그대로 사용)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            # 2. 성능 지표 계산
            preds = torch.argmax(outputs, dim=1) # Shape: [B, H, W]

            # (2) 통계 계산 (preds와 masks를 비교)
            # tp, fp, fn, tn은 [B, C] 모양 (e.g., [4, 7]) - CPU로 반환됨
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds,
                masks,
                mode='multiclass',
                num_classes=7
            )
            
            # (3) [핵심 수정] CPU 텐서를 CUDA로 이동(.to(device)) 후 누적
            total_tp += tp.sum(dim=0).to(device)
            total_fp += fp.sum(dim=0).to(device)
            total_fn += fn.sum(dim=0).to(device)
            total_tn += tn.sum(dim=0).to(device)

    # --- 루프 종료 후 ---
    
    # 3. 누적된 통계로 최종 지표 계산
    avg_loss = total_loss / len(loader)
    
    avg_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    avg_acc = smp.metrics.accuracy(total_tp, total_fp, total_fn, total_tn, reduction='micro')

    # 결과를 딕셔너리로 반환
    avg_metrics = {
        "iou_score": avg_iou.item(),
        "accuracy": avg_acc.item()
    }
    
    return avg_loss, avg_metrics

# --- 6. 메인 실행 ---

def main():
    model, train_loader, val_loader, loss_fn, optimizer = setup()
    
    best_iou = 0.0 # 최고 mIoU 점수 추적
    
    print("\n--- 훈련 시작 ---")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        
        # 훈련
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        
        # 평가
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, DEVICE) 
        
        val_iou = val_metrics['iou_score']
        val_acc = val_metrics['accuracy']
        
        # 결과 출력
        print(f"Epoch {epoch} 결과:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss  : {val_loss:.4f}")
        print(f"  Val mIoU  : {val_iou:.4f}  <-- (핵심 지표)")
        print(f"  Val Acc   : {val_acc:.4f}")
        
        # 최고 성능 모델 저장
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(SAVE_DIR, "best_model_unet.pth")
            torch.save(model.state_dict(), save_path)
            print(f"*** 최고 mIoU 갱신: {best_iou:.4f}. 모델 저장됨: {save_path} ***")
            
    print("\n--- 훈련 종료 ---")
    print(f"최고 검증 mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()