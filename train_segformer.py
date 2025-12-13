import torch
import torch.nn as nn  # [추가] PyTorch 내장 신경망 모듈
import segmentation_models_pytorch as smp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# SegFormer 모듈 임포트
from dataset import get_loader
from model_segformer import create_model 

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4       # 메모리 부족 시 2로 줄이세요
NUM_EPOCHS = 30
LR = 6e-5            # Transformer는 학습률을 낮게 설정
SAVE_DIR = "./checkpoints"
SAVE_FILENAME = "best_model_segformer.pth"

def setup():
    print(f"--- SegFormer (Transformer) 훈련 준비 ({DEVICE}) ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    train_loader = get_loader('./data', 'train', BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = get_loader('./data', 'val', BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = create_model(DEVICE)

    # [수정된 부분] smp.losses가 아니라 PyTorch 내장 함수 사용
    # label_smoothing=0.1 : 정답을 90%만 믿어라 (노이즈 라벨 극복 핵심)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    return model, train_loader, val_loader, loss_fn, optimizer

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(loader, desc="Train"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # 모델 추론
        outputs = model(images)
        
        # [중요] SegFormer는 출력 크기가 입력보다 작을 수 있음 (1/4) -> 원본 크기로 복원
        if outputs.shape[-1] != masks.shape[-1]:
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
            )
            
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    total_tp = torch.zeros(7).to(device)
    total_fp = torch.zeros(7).to(device)
    total_fn = torch.zeros(7).to(device)
    total_tn = torch.zeros(7).to(device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Valid"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # 크기 복원
            if outputs.shape[-1] != masks.shape[-1]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )

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
    print("\n--- SegFormer 학습 시작 (스스로 깨우치는 모델) ---")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_iou, val_acc = evaluate(model, val_loader, loss_fn, DEVICE)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch} | Val mIoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(SAVE_DIR, SAVE_FILENAME)
            torch.save(model.state_dict(), save_path)
            print(f"*** 신기록! ({best_iou:.4f}) -> 모델 저장됨 ***")
            
    print(f"\n최종 최고 mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()