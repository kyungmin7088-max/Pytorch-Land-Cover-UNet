import torch
import segmentation_models_pytorch as smp
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- [변경점 1] model_deeplabv3에서 모델 가져오기 ---
from dataset import get_loader
from model_deeplabv3 import create_model 

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # DeepLabV3+는 메모리를 조금 더 쓰지만 RTX 5070Ti(16GB)면 4는 충분합니다.
NUM_EPOCHS = 25
LR = 1e-4
SAVE_DIR = "./checkpoints"

# --- [변경점 2] 저장 파일명 변경 ---
SAVE_FILENAME = "best_model_deeplabv3.pth" 

def setup():
    print("--- DeepLabV3+ 훈련 준비 시작 ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 데이터 로더 (기존과 동일)
    train_loader = get_loader('./data', 'train', BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = get_loader('./data', 'val', BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 모델 생성 (DeepLabV3+)
    model = create_model(DEVICE)
    print(f"모델 로드 완료: DeepLabV3+ (ResNet-50) -> {DEVICE}")

    # 손실 함수 (DiceLoss)
    loss_fn = smp.losses.DiceLoss(
        mode=smp.losses.MULTICLASS_MODE,
        from_logits=True
    )
    
    # 옵티마이저
    optimizer = Adam(model.parameters(), lr=LR)
    
    return model, train_loader, val_loader, loss_fn, optimizer

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
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

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    # 통계 집계용 텐서
    total_tp = torch.zeros(7).to(device)
    total_fp = torch.zeros(7).to(device)
    total_fn = torch.zeros(7).to(device)
    total_tn = torch.zeros(7).to(device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validate Epoch"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
            # 성능 지표 계산
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds, masks, mode='multiclass', num_classes=7
            )
            
            total_tp += tp.sum(dim=0).to(device)
            total_fp += fp.sum(dim=0).to(device)
            total_fn += fn.sum(dim=0).to(device)
            total_tn += tn.sum(dim=0).to(device)

    avg_loss = total_loss / len(loader)
    avg_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    avg_acc = smp.metrics.accuracy(total_tp, total_fp, total_fn, total_tn, reduction='micro')

    return avg_loss, avg_iou.item(), avg_acc.item()

def main():
    model, train_loader, val_loader, loss_fn, optimizer = setup()
    
    best_iou = 0.0
    print("\n--- DeepLabV3+ 훈련 시작 ---")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_iou, val_acc = evaluate(model, val_loader, loss_fn, DEVICE)
        
        print(f"Epoch {epoch} 결과:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss  : {val_loss:.4f}")
        print(f"  Val mIoU  : {val_iou:.4f}")
        print(f"  Val Acc   : {val_acc:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(SAVE_DIR, SAVE_FILENAME)
            torch.save(model.state_dict(), save_path)
            print(f"*** 최고 mIoU 갱신: {best_iou:.4f} -> 저장됨: {SAVE_FILENAME} ***")
            
    print("\n--- 훈련 종료 ---")
    print(f"DeepLabV3+ 최종 최고 mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()