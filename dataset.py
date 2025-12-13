import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 1. Albumentations 변환(Transform) 정의 ---

def get_train_transforms():
    """
    학습용 데이터에 적용할 데이터 증강 및 전처리
    - 계획서 6장 (Data Augmentation) 
    - 계획서 6장 (Normalization) 
    """
    return A.Compose([
        # --- 데이터 증강 (과적합 방지) ---
        # 50% 확률로 좌우 반전 
        A.HorizontalFlip(p=0.5),
        # 50% 확률로 상하 반전 
        A.VerticalFlip(p=0.5),
        
        # --- 정규화 및 텐서 변환 ---
        # 픽셀 값을 0~1 사이로 정규화 (MaxAbsValue=255.0) [cite: 140]
        # PyTorch 텐서로 변환
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    """
    검증용 데이터에는 데이터 증강을 적용하지 않음.
    오직 정규화와 텐서 변환만 수행.
    """
    return A.Compose([
        # 정규화 및 텐서 변환
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# --- 2. PyTorch Dataset 클래스 ---

class DeepGlobeDataset(Dataset):
    """
    prepare_data.py로 전처리된 512x512 패치들을 불러오는 PyTorch Dataset
    """
    def __init__(self, data_root_dir='./data', set_type='train'):
        """
        :param data_root_dir: './data' 폴더
        :param set_type: 'train' 또는 'val'
        """
        print(f"Loading '{set_type}' dataset...")
        
        self.image_dir = os.path.join(data_root_dir, set_type, 'images')
        self.mask_dir = os.path.join(data_root_dir, set_type, 'masks')
        
        # ./data/train/images/*.png 파일 목록을 모두 가져옴
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        
        if not self.image_files:
            raise FileNotFoundError(f"'{self.image_dir}' 경로에 이미지 파일이 없습니다.")
            
        # 변환(Transform) 설정
        if set_type == 'train':
            self.transforms = get_train_transforms()
        elif set_type == 'val':
            self.transforms = get_val_transforms()
        else:
            raise ValueError(f"set_type은 'train' 또는 'val' 이어야 합니다. (입력: {set_type})")
            
        print(f"'{set_type}' dataset loaded. 총 {len(self.image_files)}개의 패치 발견.")

    def __len__(self):
        """데이터셋의 총 패치 개수를 반환"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        idx(인덱스)에 해당하는 이미지와 마스크를 불러오고 변환을 적용
        """
        
        # 1. 파일 경로 찾기
        img_path = self.image_files[idx]
        
        # '.../images/10078_2048_2048.png' -> '.../masks/10078_2048_2048.png'
        mask_path = img_path.replace(self.image_dir, self.mask_dir)
        
        # 2. 이미지와 마스크 불러오기
        # .convert("RGB") : 이미지가 흑백이거나 4채널이어도 3채널(RGB)로 통일
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # 마스크는 prepare_data.py에서 이미 단일 채널(0~6)로 저장했음
        mask = np.array(Image.open(mask_path)) 
        
        # 3. Albumentations 변환 적용
        # 'image'와 'mask'를 함께 넣어주면 동일한 변환이 적용됨
        try:
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
        except Exception as e:
            print(f"변환 오류 발생 (파일: {img_path}): {e}")
            # 문제가 생긴 경우, 그냥 원본을 반환 (임시방편)
            augmented = get_val_transforms()(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']

        # 4. 마스크 텐서 타입 변경
        # PyTorch의 CrossEntropyLoss는 마스크가 Float이 아닌 Long 타입이어야 함
        mask_tensor = mask_tensor.long()
        
        return image_tensor, mask_tensor

# --- 3. (테스트용) DataLoader 생성 함수 ---

def get_loader(data_root_dir, set_type, batch_size, shuffle=True, num_workers=4):
    """
    테스트 및 훈련에 사용할 DataLoader를 반환하는 헬퍼 함수
    """
    dataset = DeepGlobeDataset(data_root_dir, set_type)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True # GPU로 데이터 전송 속도 향상 (옵션)
    )
    return loader

if __name__ == "__main__":
    # 이 스크립트를 직접 실행하면, 데이터 로더가 잘 작동하는지 테스트합니다.
    print("--- dataset.py 직접 실행 테스트 ---")
    
    # 1. 라이브러리 설치 확인
    try:
        import albumentations
        print(f"Albumentations (A) version: {albumentations.__version__}")
    except ImportError:
        print("[오류] albumentations가 설치되지 않았습니다.")
        print("터미널에 'pip install albumentations'를 입력해 설치해주세요.")
        exit()

    # 2. 데이터 로더 테스트
    print("학습용(train) 데이터 로더 생성을 시도합니다...")
    try:
        train_loader = get_loader(
            './data', 
            set_type='train', 
            batch_size=4
        )
        
        # 데이터 1배치(4개)만 뽑아보기
        images, masks = next(iter(train_loader))
        
        print("\n--- 데이터 로더 테스트 성공! ---")
        print(f"불러온 이미지 텐서 모양: {images.shape}")
        print(f"불러온 마스크 텐서 모양: {masks.shape}")
        print(f"이미지 텐서 타입: {images.dtype}")
        print(f"마스크 텐서 타입: {masks.dtype} (Long 타입이 정상입니다)")
        
        print("\n[팁] 이 데이터 로더를 Jupyter Notebook에서 시각화해보세요.")

    except FileNotFoundError as e:
        print(f"\n[오류] {e}")
        print("먼저 'prepare_data.py' 스크립트를 실행해서 데이터를 전처리해야 합니다.")
    except Exception as e:
        print(f"\n[오류] 데이터 로더 생성 중 예상치 못한 오류 발생: {e}")