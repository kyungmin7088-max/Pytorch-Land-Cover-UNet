import torch
import segmentation_models_pytorch as smp
from torch.optim import AdamW  # [수정] 대문자 W로 변경
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 모듈 임포트
from dataset import get_loader
from model_efficientnet import create_model 

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4       # 메모리 부족 시 2로 줄이세요
NUM_EPOCHS = 30
LR = 1e-4
SAVE_DIR = "./checkpoints"
SAVE_FILENAME = "best_model_efficientnet.pth"

def setup():
    print(f"--- EfficientNet-B4 훈련 준비 ({DEVICE}) ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 데이터 로더
    train_loader = get_loader('./data', 'train', BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = get_loader('./data', 'val', BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 모델 생성
    model = create_model(DEVICE)

    # 손실 함수
    loss_fn = smp.losses.DiceLoss(
        mode=smp.losses.MULTICLASS_MODE,
        from_logits=True
    )
    
    # 옵티마이저 [수정] 대문자 W 적용
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    return model, train_loader, val_loader, loss_fn, optimizer

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(loader, desc="Train"):
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
    
    total_tp = torch.zeros(7).to(device)
    total_fp = torch.zeros(7).to(device)
    total_fn = torch.zeros(7).to(device)
    total_tn = torch.zeros(7).to(device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Valid"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
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
    print("\n--- 학습 시작 (목표: mIoU 0.73 돌파) ---")
    
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