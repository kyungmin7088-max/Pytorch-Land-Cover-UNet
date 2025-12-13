import os
import glob
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# --- 1. 설정 ---

# DeepGlobe 데이터셋을 다운로드하고 압축을 푼 'root' 디렉터리 경로
# 예: './DeepGlobe'
ORIGINAL_DATA_DIR = './archive' # [중요] 이 경로는 실제 경로로 수정해야 합니다.

# 타일링된 데이터가 저장될 경로
OUTPUT_DIR = './data'

# 계획서의 타일링 크기 
PATCH_SIZE = 512
VALIDATION_SPLIT = 0.2  # 20%를 검증용으로 사용

# --- 2. 클래스 색상 정의 (DeepGlobe Kaggle 기준) ---
# DeepGlobe 마스크는 RGB 색상으로 클래스를 구분합니다.
# 이를 0, 1, 2... 같은 단일 숫자로 바꿔줘야 모델이 학습할 수 있습니다.
# (R, G, B) : Class Index
# --- 2. 클래스 색상 정의 (DeepGlobe Kaggle 기준) ---
# ...
# --- 2. 클래스 색상 정의 (803개 전수 조사로 수정됨) ---
# (R, G, B) : Class Index
CLASS_MAP = {
    (0, 255, 255): 0,  # 0: Urban land (Cyan)
    (255, 255, 0): 1,  # 1: Agriculture land (Yellow)
    (255, 0, 255): 2,  # 2: Rangeland (Magenta)
    (0, 255, 0): 3,    # 3: Forest land (Green)
    (0, 0, 255): 4,    # 4: Water (Blue)  <--- 버그 수정됨!
    (255, 255, 255): 5, # 5: Barren land (White)
    (0, 0, 0): 6       # 6: Unknown (Black)
}

def rgb_to_mask(rgb_mask_image):
    """RGB 마스크 이미지를 (H, W)의 단일 채널 인덱스 맵으로 변환합니다."""
    mask = np.zeros(rgb_mask_image.shape[:2], dtype=np.uint8)
    for rgb_color, class_index in CLASS_MAP.items():
        # np.all을 사용하여 R, G, B가 모두 일치하는 픽셀을 찾습니다.
        matches = np.all(rgb_mask_image == rgb_color, axis=-1)
        mask[matches] = class_index
    return mask

# --- 3. 타일링 및 저장 함수 ---

def tile_and_save(image_path_list, output_set_dir):
    """
    주어진 이미지 경로 리스트를 타일링하여 output_set_dir에 저장합니다.
    (예: output_set_dir = './data/train')
    """
    img_dir = os.path.join(output_set_dir, 'images')
    mask_dir = os.path.join(output_set_dir, 'masks')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"'{output_set_dir}' 경로에 타일링을 시작합니다...")
    
    for full_image_path in tqdm(image_path_list):
        # 1. 이미지 로드
        image_id = os.path.basename(full_image_path).replace('_sat.jpg', '')
        # 마스크 파일 경로 (원본 마스크는 _mask.png 입니다)
        mask_path = full_image_path.replace('_sat.jpg', '_mask.png') 
        
        try:
            image = Image.open(full_image_path)
            mask_rgb = Image.open(mask_path).convert("RGB")
            
            # 2. 원본 이미지 크기 (2448x2448) 
            width, height = image.size
            
            # 3. 512x512 크기로 타일링 
            for i in range(0, width, PATCH_SIZE):
                for j in range(0, height, PATCH_SIZE):
                    # 경계가 밖으로 나가지 않도록 처리
                    if i + PATCH_SIZE > width or j + PATCH_SIZE > height:
                        continue
                        
                    # 이미지 타일
                    patch_image = image.crop((i, j, i + PATCH_SIZE, j + PATCH_SIZE))
                    
                    # 마스크 타일
                    patch_mask_rgb = mask_rgb.crop((i, j, i + PATCH_SIZE, j + PATCH_SIZE))
                    
                    # RGB 마스크를 -> 단일 채널 인덱스 맵으로 변환
                    patch_mask_indexed = rgb_to_mask(np.array(patch_mask_rgb))
                    
                    # PIL.Image 객체로 다시 변환하여 저장
                    patch_mask_to_save = Image.fromarray(patch_mask_indexed)

                    # 4. 저장 (파일 이름: 원본ID_x좌표_y좌표.png)
                    patch_name = f"{image_id}_{i}_{j}.png" # png로 통일
                    
                    patch_image.save(os.path.join(img_dir, patch_name))
                    patch_mask_to_save.save(os.path.join(mask_dir, patch_name))

        except Exception as e:
            print(f"[경고] {image_id} 처리 중 오류 발생: {e}")

# --- 4. 메인 실행 로직 ---

def main():
    print("--- 데이터 전처리 시작 ---")
    
    # 1. 원본 학습 데이터셋 경로 설정
    train_dir = os.path.join(ORIGINAL_DATA_DIR, 'train')
    
    # 2. 원본 이미지 목록 가져오기 (803개) 
    # glob을 사용해 모든 _sat.jpg 파일을 찾습니다.
    all_image_paths = sorted(glob.glob(os.path.join(train_dir, '*_sat.jpg')))
    
    if not all_image_paths:
        print(f"[오류] '{train_dir}' 경로에 이미지가 없습니다.")
        print("ORIGINAL_DATA_DIR 변수를 올바르게 설정했는지 확인하세요.")
        return

    print(f"총 {len(all_image_paths)}개의 원본 이미지를 찾았습니다.")

    # 3. 학습 / 검증 데이터 분리 
    train_paths, val_paths = train_test_split(
        all_image_paths, 
        test_size=VALIDATION_SPLIT, 
        random_state=42 # 재현 가능성을 위해 random_state 고정
    )
    
    print(f"학습용: {len(train_paths)}개 / 검증용: {len(val_paths)}개로 분리합니다.")

    # 4. 타일링 및 저장 실행
    tile_and_save(train_paths, os.path.join(OUTPUT_DIR, 'train'))
    tile_and_save(val_paths, os.path.join(OUTPUT_DIR, 'val'))
    
    print("--- 모든 데이터 전처리 완료 ---")

if __name__ == "__main__":
    main()