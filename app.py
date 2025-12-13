import os
import math
from flask import Flask, render_template, request, url_for
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# [변경] SegFormer 모델 모듈 임포트
from model_segformer import create_model

app = Flask(__name__)

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [변경] SegFormer 가중치 파일 경로
MODEL_PATH = "./checkpoints/best_model_segformer.pth"
UPLOAD_FOLDER = './static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 패치 크기 및 스트라이드 (오버랩 설정)
PATCH_SIZE = 512
STRIDE = 256  # 50% 겹침

PALETTE = [
    [0, 255, 255],   # 0: Urban
    [255, 255, 0],   # 1: Agri
    [255, 0, 255],   # 2: Range
    [0, 255, 0],     # 3: Forest
    [0, 0, 255],     # 4: Water
    [255, 255, 255], # 5: Barren
    [0, 0, 0]        # 6: Unknown
]

# --- 모델 로드 ---
print(f"[{DEVICE}] SegFormer (MiT-B3) 모델 로딩 중...")
model = create_model(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(">>> 모델 로드 완료! (최종 우승 모델)")
except Exception as e:
    print(f">>> [치명적 오류] 모델 로드 실패: {e}")

# --- 헬퍼 함수 ---
def decode_mask(mask_index_map):
    h, w = mask_index_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        rgb[mask_index_map == idx] = color
    return Image.fromarray(rgb)

def predict_patch_probs(image_numpy):
    """
    SegFormer 예측 및 크기 보정
    """
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    augmented = transform(image=image_numpy)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        
        # [중요] SegFormer 출력이 입력보다 작을 경우 강제 확대 (Upsampling)
        if output.shape[-1] != input_tensor.shape[-1]:
            output = F.interpolate(
                output, 
                size=input_tensor.shape[-2:], # (512, 512)
                mode='bilinear', 
                align_corners=False
            )
            
        probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return probs

def process_sliding_window(image_path):
    """오버랩 슬라이딩 윈도우 (로직 동일)"""
    origin_image = Image.open(image_path).convert("RGB")
    w, h = origin_image.size
    image_np = np.array(origin_image)
    
    # 패딩 계산
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    
    image_padded = np.pad(image_np, ((0, pad_h + STRIDE), (0, pad_w + STRIDE), (0, 0)), mode='constant')
    padded_h, padded_w, _ = image_padded.shape
    
    full_probs = np.zeros((padded_h, padded_w, 7), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w, 1), dtype=np.float32)
    
    print(f"SegFormer 분석 시작... (크기: {w}x{h})")

    for y in range(0, padded_h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, padded_w - PATCH_SIZE + 1, STRIDE):
            patch = image_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            patch_probs = predict_patch_probs(patch)
            
            # 차원 변경 (C, H, W) -> (H, W, C)
            patch_probs = patch_probs.transpose(1, 2, 0)
            
            full_probs[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += patch_probs
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    full_probs /= count_map
    final_probs = full_probs[:h, :w, :]
    final_mask = np.argmax(final_probs, axis=2).astype(np.uint8)

    # 후처리 (노이즈 제거)
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    result_image = decode_mask(final_mask)
    filename = os.path.basename(image_path)
    save_name = "pred_segformer_" + filename # 파일명 구분
    save_path = os.path.join(UPLOAD_FOLDER, save_name)
    result_image.save(save_path)
    
    return save_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return '파일 없음'
        file = request.files['file']
        if file.filename == '': return '파일 선택 안함'
            
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            result_filename = process_sliding_window(filepath)
            return render_template('index.html', original_image=file.filename, result_image=result_filename)
                                   
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)