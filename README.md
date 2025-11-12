# Pytorch-Land-Cover-UNet: 자동 토지 피복 지도 생성 시스템

이 리포지토리는 '파이썬 기반 딥딥러닝' 기말 프로젝트 ('위성 이미지를 활용한 자동 토지 피복 지도 생성 시스템')의 공식 코드 저장소입니다.

본 프로젝트는 PyTorch를 기반으로 DeepGlobe Land Cover Classification Dataset의 위성 이미지를 픽셀 단위로 분류하는 Semantic Segmentation 모델을 개발합니다. 베이스라인으로 U-Net을, 최종 목표로 DeepLabV3+ 모델을 사용하여 성능을 비교 분석합니다.

최종 목표는 사용자가 이미지를 업로드하면, 학습된 모델이 토지 피복 지도를 실시간으로 생성하고 시각화하는 Flask 기반의 웹 애플리케이션을 구축하는 것입니다.

## 개발 환경
* **OS:** Windows 11
* **Language:** Python 3.9+
* **Core Framework:**
    * PyTorch 2.0+
    * Flask 2.3+
* **Key Libraries:**
    * OpenCV-Python, Pillow
    * NumPy
    * Matplotlib

## 설치 및 준비

### 1. 리포지토리 클론
```bash
git clone [https://github.com/kyungmin7088-max/Pytorch-Land-Cover-UNet.git](https://github.com/kyungmin7088-max/Pytorch-Land-Cover-UNet.git)
cd Pytorch-Land-Cover-UNet
```

### 2. (권장) 가상 환경 생성 및 활성화
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. 필요 라이브러리 설치
```bash
pip install torch torchvision flask opencv-python-headless numpy matplotlib
```

### 4. 데이터셋 준비
1.  Kaggle에서 [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset?resource=download)을 다운로드합니다.
2.  프로젝트 루트 디렉터리에 `data` 폴더를 생성한 후, 그 안에 다운로드한 파일의 압축을 풉니다.
3.  (참고: 원본 이미지는 2448x2448 해상도이며, 데이터 로더에서 학습에 용이한 512x512 크기의 패치(Patch)로 분할하여 사용할 예정입니다.)

## 사용 방법 (개발 진행 중)

본 프로젝트는 개발 일정에 따라 다음 순서로 기능이 구현될 예정입니다.

1.  **데이터 전처리 (`dataset.py`):**
    * 원본 이미지를 512x512 패치로 타일링하고, 정규화 및 데이터 증강을 적용하는 PyTorch `Dataset` 및 `DataLoader`를 구현합니다.

2.  **모델 학습 및 평가 (`train.py`):**
    * U-Net (베이스라인) 및 DeepLabV3+ (최종 목표) 모델을 구현하고 학습을 진행합니다.
    * 성능은 Mean IoU (mIoU)를 핵심 지표로 하여 평가합니다.

3.  **웹 애플리케이션 실행 (`app.py`):**
    * 학습이 완료된 최종 모델(.pth)을 로드합니다.
    * 사용자가 업로드한 이미지에 대해 추론을 수행하고, 결과를 컬러 지도 형태로 시각화하는 Flask 앱을 실행합니다.

*(각 스크립트가 완성되는 대로, 구체적인 실행 방법이 이곳에 업데이트될 것입니다.)*
