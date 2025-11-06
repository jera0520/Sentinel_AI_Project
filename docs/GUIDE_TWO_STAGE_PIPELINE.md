# 🎓 2단계 파이프라인 구축 가이드 (비전공자용)

> **목표**: 사람을 먼저 탐지하고, 그 사람이 헬멧을 착용했는지 분류하는 2단계 시스템 만들기

---

## 📖 목차
1. [왜 2단계로 나누나요?](#1-왜-2단계로-나누나요)
2. [전체 흐름 이해하기](#2-전체-흐름-이해하기)
3. [Step 1: 데이터 크롭하기](#step-1-데이터-크롭하기)
4. [Step 2: 헬멧 분류기 학습하기](#step-2-헬멧-분류기-학습하기)
5. [Step 3: 파이프라인 통합하기](#step-3-파이프라인-통합하기)
6. [문제 해결](#문제-해결)

---

## 1. 왜 2단계로 나누나요?

### 현재 방식 (Single-Stage)
```
카메라 영상 → YOLO → person, helmet, no_helmet, fallen 동시 탐지
```
**문제점:**
- 배경이 복잡하면 작은 헬멧을 놓침
- 헬멧만 있고 사람이 없는 경우 오탐지
- 정확도가 낮음

### 개선 방식 (Two-Stage) ⭐
```
카메라 영상 → [1단계] 사람 탐지 → 사람 영역 크롭 
            → [2단계] 헬멧 분류 → helmet / no_helmet
```
**장점:**
- ✅ 배경 제거로 정확도 향상
- ✅ 헬멧만 있는 오탐지 제거
- ✅ 작은 헬멧도 잘 탐지 (크롭 후 확대되므로)

---

## 2. 전체 흐름 이해하기

### 🎯 전체 작업 단계

```
┌─────────────────────────────────────────────────────┐
│ Step 1: 데이터 준비 (1-2시간)                        │
├─────────────────────────────────────────────────────┤
│ 1-1. person bbox 추출                               │
│ 1-2. 이미지 크롭                                     │
│ 1-3. helmet/no_helmet 라벨링                        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Step 2: 헬멧 분류기 학습 (2-3시간)                   │
├─────────────────────────────────────────────────────┤
│ 2-1. TensorFlow 환경 설정                           │
│ 2-2. 분류 모델 학습 (MobileNet)                     │
│ 2-3. 모델 테스트                                     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Step 3: 파이프라인 통합 (1시간)                      │
├─────────────────────────────────────────────────────┤
│ 3-1. main_scale.py 수정                             │
│ 3-2. 테스트 및 검증                                  │
└─────────────────────────────────────────────────────┘
```

**총 예상 시간: 4-6시간**

---

## Step 1: 데이터 크롭하기

### 📂 작업 폴더 준비

**1-1. 터미널 열기**
```bash
cd /home/jera/Sentinel_AI_Project
```

**1-2. 작업 폴더 만들기**
```bash
mkdir -p two_stage_data
cd two_stage_data
mkdir -p person_crops/helmet
mkdir -p person_crops/no_helmet
mkdir -p person_crops/unlabeled
```

**폴더 구조:**
```
two_stage_data/
└── person_crops/
    ├── helmet/          # 헬멧 착용한 사람들
    ├── no_helmet/       # 헬멧 미착용 사람들
    └── unlabeled/       # 아직 라벨 안 붙인 것들
```

---

### 📝 크롭 스크립트 만들기

**1-3. 크롭 스크립트 생성**

터미널에 다음 명령어 입력:
```bash
cat > crop_persons.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Hub 데이터에서 person bbox를 크롭하여 저장하는 스크립트
"""
import cv2
import os
import json
from pathlib import Path

def read_yolo_label(txt_path):
    """YOLO 형식의 txt 파일에서 bbox 읽기"""
    bboxes = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    bboxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    except Exception as e:
        print(f"라벨 읽기 오류: {txt_path}, {e}")
    return bboxes

def yolo_to_pixel(bbox, img_width, img_height):
    """YOLO 좌표를 픽셀 좌표로 변환"""
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    w = bbox['width'] * img_width
    h = bbox['height'] * img_height
    
    xmin = int(x_center - w/2)
    ymin = int(y_center - h/2)
    xmax = int(x_center + w/2)
    ymax = int(y_center + h/2)
    
    return xmin, ymin, xmax, ymax

def crop_persons(data_dir, output_dir, person_class_id=0, min_size=50):
    """
    데이터 디렉토리에서 person 클래스만 크롭하여 저장
    
    Args:
        data_dir: 이미지와 라벨이 있는 폴더
        output_dir: 크롭된 이미지를 저장할 폴더
        person_class_id: person 클래스 ID (기본: 0)
        min_size: 최소 크기 (픽셀, 이것보다 작으면 무시)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    image_files = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
    print(f"✅ 총 {len(image_files)}개 이미지 발견")
    
    crop_count = 0
    skip_count = 0
    
    for img_path in image_files:
        # 라벨 파일 경로
        txt_path = img_path.with_suffix('.txt')
        if not txt_path.exists():
            print(f"⚠️  라벨 없음: {img_path.name}")
            continue
        
        # 이미지 읽기
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ 이미지 읽기 실패: {img_path.name}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 라벨 읽기
        bboxes = read_yolo_label(txt_path)
        
        # person 클래스만 필터링
        person_bboxes = [b for b in bboxes if b['class_id'] == person_class_id]
        
        if not person_bboxes:
            continue
        
        # 각 person bbox 크롭
        for idx, bbox in enumerate(person_bboxes):
            xmin, ymin, xmax, ymax = yolo_to_pixel(bbox, img_width, img_height)
            
            # 경계 체크
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width, xmax)
            ymax = min(img_height, ymax)
            
            # 크기 체크
            w = xmax - xmin
            h = ymax - ymin
            if w < min_size or h < min_size:
                skip_count += 1
                continue
            
            # 크롭
            cropped = img[ymin:ymax, xmin:xmax]
            
            # 저장
            filename = f"{img_path.stem}_person{idx}.jpg"
            output_file = output_path / filename
            cv2.imwrite(str(output_file), cropped)
            crop_count += 1
            
            if crop_count % 100 == 0:
                print(f"진행 중... {crop_count}개 크롭 완료")
    
    print(f"\n✅ 크롭 완료!")
    print(f"   - 저장된 이미지: {crop_count}개")
    print(f"   - 건너뛴 이미지: {skip_count}개 (너무 작음)")
    print(f"   - 저장 위치: {output_dir}")

if __name__ == '__main__':
    # 설정
    DATA_DIR = '/home/jera/Sentinel_AI_Project/Dataset/Data/Keypoint/1.Tranining/labels'
    OUTPUT_DIR = '/home/jera/Sentinel_AI_Project/two_stage_data/person_crops/unlabeled'
    PERSON_CLASS_ID = 0  # obj.names에서 person의 인덱스 확인
    
    print("=" * 60)
    print("🚀 Person 크롭 스크립트 시작")
    print("=" * 60)
    print(f"📂 입력 폴더: {DATA_DIR}")
    print(f"💾 출력 폴더: {OUTPUT_DIR}")
    print(f"👤 Person 클래스 ID: {PERSON_CLASS_ID}")
    print("=" * 60)
    
    crop_persons(DATA_DIR, OUTPUT_DIR, PERSON_CLASS_ID, min_size=50)
    
    print("\n📋 다음 단계:")
    print("1. unlabeled 폴더의 이미지들을 확인하세요")
    print("2. 헬멧 착용 여부에 따라 helmet / no_helmet 폴더로 이동하세요")
    print("   - 헬멧 O: helmet 폴더로")
    print("   - 헬멧 X: no_helmet 폴더로")
EOF

chmod +x crop_persons.py
```

**1-4. 스크립트 실행**
```bash
python3 crop_persons.py
```

**예상 출력:**
```
============================================================
🚀 Person 크롭 스크립트 시작
============================================================
📂 입력 폴더: .../1.Tranining/labels
💾 출력 폴더: .../person_crops/unlabeled
👤 Person 클래스 ID: 0
============================================================
✅ 총 23899개 이미지 발견
진행 중... 100개 크롭 완료
진행 중... 200개 크롭 완료
...
✅ 크롭 완료!
   - 저장된 이미지: 1523개
   - 건너뛴 이미지: 245개 (너무 작음)
```

---

### 🏷️ 라벨링 하기 (중요!)

이제 크롭된 이미지를 helmet / no_helmet 폴더로 분류해야 합니다.

**방법 1: 수동 분류 (추천, 정확함)**

1. 파일 탐색기로 `two_stage_data/person_crops/unlabeled` 폴더 열기
2. 이미지를 하나씩 보면서:
   - 헬멧 착용 → `helmet` 폴더로 드래그
   - 헬멧 미착용 → `no_helmet` 폴더로 드래그

**팁:**
- 한 번에 100개씩 나눠서 작업하세요 (지치지 않게)
- 확실하지 않은 이미지는 건너뛰세요
- 각 클래스별 최소 100개씩은 필요합니다

**방법 2: 스크립트로 자동 분류 (빠르지만 부정확)**

```bash
# AI Hub 데이터의 원본 JSON을 읽어서 자동 분류하는 스크립트
# (JSON 파일이 있는 경우에만 가능)
```

---

### ✅ 데이터 확인

라벨링이 끝나면 확인:
```bash
cd /home/jera/Sentinel_AI_Project/two_stage_data/person_crops

# 각 폴더의 이미지 개수 확인
echo "helmet: $(ls helmet/*.jpg 2>/dev/null | wc -l)개"
echo "no_helmet: $(ls no_helmet/*.jpg 2>/dev/null | wc -l)개"
echo "unlabeled: $(ls unlabeled/*.jpg 2>/dev/null | wc -l)개"
```

**목표:**
- helmet: 최소 100개 (더 많을수록 좋음)
- no_helmet: 최소 100개
- unlabeled: 0개 (모두 분류 완료)

---

## Step 2: 헬멧 분류기 학습하기

### 🔧 TensorFlow 설치

**2-1. 가상환경 만들기**
```bash
cd /home/jera/Sentinel_AI_Project/two_stage_data
python3 -m venv classifier_env
source classifier_env/bin/activate
```

**2-2. 필요한 패키지 설치**
```bash
pip install --upgrade pip
pip install tensorflow==2.10.0
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pillow
```

설치 시간: 약 5-10분

---

### 🤖 분류 모델 학습 스크립트

**2-3. 학습 스크립트 생성**

```bash
cat > train_helmet_classifier.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헬멧 착용 여부 분류 모델 학습 스크립트
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("🤖 헬멧 분류기 학습 시작")
print("=" * 60)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {tf.test.is_gpu_available()}")
print("=" * 60)

# ========== 설정 ==========
DATA_DIR = 'person_crops'  # helmet, no_helmet 폴더가 있는 위치
IMG_SIZE = (224, 224)  # 이미지 크기
BATCH_SIZE = 32
EPOCHS = 20  # 학습 반복 횟수
LEARNING_RATE = 0.001

# ========== 데이터 로드 ==========
print("\n📂 데이터 로딩 중...")

# 데이터 증강 (데이터를 다양하게 변형하여 더 잘 학습하게 함)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 0-255 값을 0-1로 정규화
    rotation_range=20,  # 랜덤 회전
    width_shift_range=0.2,  # 좌우 이동
    height_shift_range=0.2,  # 상하 이동
    horizontal_flip=True,  # 좌우 반전
    validation_split=0.2  # 20%는 검증용으로
)

# 학습 데이터
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # helmet(0) vs no_helmet(1)
    subset='training'
)

# 검증 데이터
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"✅ 학습 데이터: {train_generator.samples}개")
print(f"✅ 검증 데이터: {validation_generator.samples}개")
print(f"📋 클래스: {train_generator.class_indices}")

# ========== 모델 구축 ==========
print("\n🏗️  모델 구축 중...")

# MobileNetV2 사용 (가볍고 빠른 모델)
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # 마지막 분류 레이어 제외
    weights='imagenet'  # ImageNet으로 사전 학습된 가중치 사용
)

# 사전 학습된 부분은 고정 (처음에는)
base_model.trainable = False

# 우리 모델 구축
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),  # 과적합 방지
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # 0 or 1 출력
])

# 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== 모델 학습 ==========
print("\n🔥 학습 시작...")

# 학습 중 최고 성능 모델 저장
checkpoint = keras.callbacks.ModelCheckpoint(
    'helmet_classifier_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 학습
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# ========== 결과 저장 ==========
print("\n💾 모델 저장 중...")
model.save('helmet_classifier_final.h5')
print("✅ 모델 저장 완료: helmet_classifier_final.h5")

# ========== 학습 그래프 ==========
print("\n📊 학습 그래프 생성 중...")

plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ 그래프 저장 완료: training_history.png")

# ========== 최종 성능 출력 ==========
print("\n" + "=" * 60)
print("🎉 학습 완료!")
print("=" * 60)
print(f"📊 최종 학습 정확도: {history.history['accuracy'][-1]:.2%}")
print(f"📊 최종 검증 정확도: {history.history['val_accuracy'][-1]:.2%}")
print(f"💾 저장된 파일:")
print(f"   - helmet_classifier_best.h5 (최고 성능 모델)")
print(f"   - helmet_classifier_final.h5 (최종 모델)")
print(f"   - training_history.png (학습 그래프)")
print("=" * 60)
EOF

chmod +x train_helmet_classifier.py
```

**2-4. 학습 실행**
```bash
source classifier_env/bin/activate
python3 train_helmet_classifier.py
```

**예상 소요 시간:**
- GPU 있음: 5-10분
- GPU 없음: 20-30분

**학습 과정 출력 예시:**
```
============================================================
🤖 헬멧 분류기 학습 시작
============================================================
TensorFlow 버전: 2.10.0
GPU 사용 가능: True
============================================================

📂 데이터 로딩 중...
✅ 학습 데이터: 240개
✅ 검증 데이터: 60개
📋 클래스: {'helmet': 0, 'no_helmet': 1}

🏗️  모델 구축 중...
Model: "sequential"
...

🔥 학습 시작...
Epoch 1/20
8/8 [==============================] - 12s 1s/step - loss: 0.6234 - accuracy: 0.6917 - val_loss: 0.4521 - val_accuracy: 0.8000
Epoch 2/20
8/8 [==============================] - 10s 1s/step - loss: 0.4123 - accuracy: 0.8250 - val_loss: 0.3012 - val_accuracy: 0.8833
...
Epoch 20/20
8/8 [==============================] - 10s 1s/step - loss: 0.1234 - accuracy: 0.9583 - val_loss: 0.1521 - val_accuracy: 0.9333

🎉 학습 완료!
📊 최종 학습 정확도: 95.83%
📊 최종 검증 정확도: 93.33%
```

---

### 🧪 모델 테스트

**2-5. 테스트 스크립트 생성**

```bash
cat > test_classifier.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 헬멧 분류기 테스트
"""
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import sys

# 모델 로드
print("🤖 모델 로딩 중...")
model = keras.models.load_model('helmet_classifier_best.h5')
print("✅ 모델 로드 완료")

def predict_image(image_path):
    """이미지에서 헬멧 착용 여부 예측"""
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 읽기 실패: {image_path}")
        return None
    
    # 전처리
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # 예측
    prediction = model.predict(img_batch, verbose=0)[0][0]
    
    # 결과 해석
    if prediction < 0.5:
        label = "helmet"
        confidence = (1 - prediction) * 100
    else:
        label = "no_helmet"
        confidence = prediction * 100
    
    return label, confidence

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python3 test_classifier.py <이미지_경로>")
        print("예시: python3 test_classifier.py person_crops/helmet/test.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = predict_image(image_path)
    
    if result:
        label, confidence = result
        print(f"\n📸 이미지: {image_path}")
        print(f"🎯 예측: {label}")
        print(f"📊 신뢰도: {confidence:.2f}%")
EOF

chmod +x test_classifier.py
```

**2-6. 테스트 실행**
```bash
# helmet 폴더의 이미지로 테스트
python3 test_classifier.py person_crops/helmet/S2-N6001M00001_person0.jpg

# no_helmet 폴더의 이미지로 테스트
python3 test_classifier.py person_crops/no_helmet/S2-N6002M00050_person1.jpg
```

**예상 출력:**
```
🤖 모델 로딩 중...
✅ 모델 로드 완료

📸 이미지: person_crops/helmet/S2-N6001M00001_person0.jpg
🎯 예측: helmet
📊 신뢰도: 94.32%
```

---

## Step 3: 파이프라인 통합하기

이제 학습한 분류 모델을 main_scale.py에 통합합니다.

### 📝 통합 가이드 문서 생성

```bash
cd /home/jera/Sentinel_AI_Project
cat > INTEGRATION_GUIDE.md << 'EOF'
# 2단계 파이프라인 통합 가이드

## 필요한 파일

1. 학습된 모델: `helmet_classifier_best.h5`
2. 수정할 파일: `anu_example/main_scale.py`

## 통합 방법

### 1단계: 모델 파일 복사

```bash
# 학습한 모델을 anu_example로 복사
cp two_stage_data/helmet_classifier_best.h5 anu_example/model/
```

### 2단계: 분류기 클래스 추가

`anu_example/main_scale.py` 파일 상단에 추가:

```python
import tensorflow as tf
from tensorflow import keras

class HelmetClassifier:
    """헬멧 착용 여부 분류기"""
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(f"[HelmetClassifier] 모델 로드 완료: {model_path}")
    
    def predict(self, image):
        """
        크롭된 사람 이미지에서 헬멧 착용 여부 예측
        
        Args:
            image: BGR 이미지 (numpy array)
        
        Returns:
            (label, confidence): ('helmet' or 'no_helmet', 신뢰도 0-1)
        """
        # 전처리
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # 예측
        prediction = self.model.predict(img_batch, verbose=0)[0][0]
        
        # 결과
        if prediction < 0.5:
            return 'helmet', (1 - prediction)
        else:
            return 'no_helmet', prediction
```

### 3단계: DetectParser 수정

```python
class DetectParser(multiprocessing.Process):
    def __init__(self, queue_in, queue_out, save_crops=True, use_two_stage=False):
        # ... 기존 코드 ...
        self.use_two_stage = use_two_stage
        self.helmet_classifier = None
    
    def add_model(self):
        # Stage 1: Person Detector
        self.person_net = Yolo(
            'model/person5l/model.cfg',
            'model/person5l/model.weights',
            'model/person5l/model.names'
        )
        
        # Stage 2: Helmet Classifier (옵션)
        if self.use_two_stage:
            self.helmet_classifier = HelmetClassifier('model/helmet_classifier_best.h5')
    
    def run(self):
        self.add_model()
        frame_cnt = 0
        
        while True:
            frame_pair = self.queue_in.get()
            if frame_pair is None:
                self.queue_out.put(None)
                break
            
            orig_frame, up_frame = frame_pair
            frame_cnt += 1
            
            if frame_cnt % 3 == 0:
                # Stage 1: Detect persons
                yolo_dets_orig = self.person_net.detect(orig_frame, 0.5, 0.5)
                
                # Stage 2: Classify helmet (옵션)
                if self.use_two_stage and self.helmet_classifier:
                    for det in yolo_dets_orig:
                        if det['label'] == 0:  # person 클래스만
                            # 크롭
                            x1, y1, x2, y2 = det['bbox']
                            crop = orig_frame[y1:y2, x1:x2]
                            
                            if crop.size > 0:
                                # 헬멧 분류
                                helmet_label, confidence = self.helmet_classifier.predict(crop)
                                det['helmet_status'] = helmet_label
                                det['helmet_confidence'] = confidence
                
                # 추적
                tracks_orig, _ = self.tracker_orig.update(yolo_dets_orig, orig_frame)
                
                # ... 나머지 코드 ...
```

### 4단계: main 함수에 옵션 추가

```python
if __name__ == '__main__':
    parser.add_argument('--two-stage', action='store_true',
                        help='2단계 파이프라인 활성화')
    
    # ...
    
    detector = DetectParser(q_video, q_detect, 
                           save_crops=args.save_crops,
                           use_two_stage=args.two_stage)
```

### 5단계: 실행

```bash
# 2단계 파이프라인으로 실행
cd /home/jera/Sentinel_AI_Project/anu_example
source venv/bin/activate
python3 main_scale.py --two-stage
```

## 성능 비교

실행 후:
- Single-Stage (기존): 빠르지만 오탐지 가능
- Two-Stage (개선): 느리지만 정확도 높음

둘 다 테스트해보고 선택하세요!
EOF
```

---

## 문제 해결

### Q1: 크롭 이미지가 너무 적어요
**A:** `crop_persons.py`에서 `min_size=50`을 `min_size=30`으로 줄여보세요.

### Q2: TensorFlow 설치 오류
**A:** 
```bash
pip install tensorflow-cpu==2.10.0  # CPU 버전만
```

### Q3: 학습이 너무 오래 걸려요
**A:** `EPOCHS = 20`을 `EPOCHS = 10`으로 줄이세요.

### Q4: 정확도가 너무 낮아요
**A:** 
1. 데이터를 더 수집하세요 (각 클래스 200개 이상)
2. 라벨링을 다시 확인하세요
3. EPOCHS를 30으로 늘려보세요

### Q5: 메모리 부족 오류
**A:** `BATCH_SIZE = 32`를 `BATCH_SIZE = 16`으로 줄이세요.

---

## 📚 참고 자료

- TensorFlow 공식 문서: https://www.tensorflow.org/
- MobileNet 논문: https://arxiv.org/abs/1704.04861
- 이미지 분류 튜토리얼: https://www.tensorflow.org/tutorials/images/classification

---

## ✅ 체크리스트

완료한 단계를 체크하세요:

**Step 1: 데이터 준비**
- [ ] crop_persons.py 스크립트 생성
- [ ] 스크립트 실행으로 person 크롭
- [ ] helmet / no_helmet 라벨링 완료
- [ ] 각 클래스 최소 100개 확보

**Step 2: 모델 학습**
- [ ] TensorFlow 설치
- [ ] train_helmet_classifier.py 생성
- [ ] 모델 학습 완료
- [ ] 학습 그래프 확인
- [ ] 테스트로 검증

**Step 3: 통합**
- [ ] 모델 파일 복사
- [ ] main_scale.py 수정
- [ ] 2단계 파이프라인 테스트
- [ ] 성능 비교

---

**작성일**: 2024-10-14
**대상**: 비전공자
**예상 소요 시간**: 4-6시간
**난이도**: ⭐⭐⭐☆☆

화이팅! 천천히 따라하시면 됩니다. 막히는 부분이 있으면 언제든 질문하세요! 🚀
