# 데이터 수집 및 전처리 가이드

이 문서는 Sentinel AI 프로젝트의 데이터 수집, 정제, 라벨링 과정을 정리한 것입니다.

---

## 📥 데이터 출처

### AI Hub - 공사현장 안전장비 인식 이미지
- **URL**: https://aihub.or.kr/
- **카테고리**: 안전/재난
- **데이터 구성**:
  - 원천 데이터: 이미지 파일(.jpg)
  - 라벨링 데이터: JSON 형식의 keypoint 정보
- **총 규모**: 23,899장

### 데이터 특징
- **촬영 환경**: 실제 공사 현장 (건설, 도로, 건축 등)
- **촬영 조건**: 다양한 조명(낮, 밤), 날씨(맑음, 흐림), 각도
- **포함 객체**: 작업자, 안전모, 안전 장비, 공사 장비
- **해상도**: 평균 1920x1080 (Full HD)

---

## 🔄 전처리 과정

### 1단계: 데이터 다운로드
```bash
# AI Hub에서 데이터 다운로드 (회원가입 필요)
# Dataset/Data/Keypoint/ 경로에 압축 해제
cd Dataset/Data/Keypoint/
unzip 원천데이터.zip -d 1.Training/
unzip 라벨링데이터.zip -d 1.Training/
```

### 2단계: JSON → YOLO 변환
AI Hub의 라벨링 데이터는 JSON 형식이므로, Darknet에서 사용하는 YOLO txt 형식으로 변환이 필요합니다.

**JSON 형식 예시**:
```json
{
  "image": {
    "resolution": [1920, 1080]
  },
  "annotations": [
    {
      "class": "60",  # person
      "point": [[x1, y1, v1], [x2, y2, v2], ...]
    }
  ]
}
```

**YOLO 형식 변환 후**:
```
0 0.512 0.345 0.125 0.234
# <class_id> <x_center> <y_center> <width> <height>
# 모든 좌표는 0~1로 정규화
```

**변환 스크립트 실행**:
```bash
python data/preprocessing/convert_json_to_yolo.py
```

### 3단계: 클래스 매핑
```python
CLASS_MAPPING = {
    "60": 0,   # person
    "61": 1,   # helmet
    "62": 2,   # no_helmet
    "63": 3,   # fallen
}
```

### 4단계: 데이터 검수 (YOLO Mark)
자동 변환된 라벨의 정확도를 수동으로 검수합니다.

```bash
cd tools/Yolo_mark
./linux_mark.sh
```

---

## 🎯 데이터 재정제 과정

### 문제점 발견 (1차 학습 후)
1. **흐릿한 이미지**: 야간 촬영, 흔들림 등으로 품질 낮음
2. **클래스 불균형**: person 15,000장, helmet 5,000장, fallen 500장
3. **중복 샘플**: 동일 장면의 연속 프레임
4. **라벨링 오류**: 헬멧을 사람으로 잘못 표시 등

### 정제 전략
1. **품질 필터링**:
   - 해상도 1280x720 이상만 선택
   - 블러 정도 측정 (Laplacian variance) 후 선명한 이미지만 유지
   
2. **균형 샘플링**:
   ```
   Target: 각 클래스당 250장 (총 1,000장)
   - person: 250장
   - helmet: 250장
   - no_helmet: 250장
   - fallen: 250장
   ```

3. **중복 제거**:
   - 이미지 해시(perceptual hash)로 유사도 계산
   - 유사도 95% 이상은 중복으로 간주하고 제거

4. **수동 검수**:
   - YOLO Mark로 전체 1,000장 재검수
   - 바운딩 박스 위치 미세 조정
   - 잘못된 라벨 수정

### 정제 스크립트
```python
# data/preprocessing/filter_quality.py
import cv2

def calculate_blur(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var

# 블러 점수 100 이하는 제외
images = [img for img in images if calculate_blur(img) > 100]
```

---

## 📊 데이터 통계

### 1차 학습 데이터 (전체)
| 클래스 | 이미지 수 | 비율 |
|--------|----------|------|
| person | 15,234 | 64% |
| helmet | 5,821 | 24% |
| no_helmet | 2,344 | 10% |
| fallen | 500 | 2% |
| **합계** | **23,899** | **100%** |

### 2차 학습 데이터 (정제)
| 클래스 | 이미지 수 | 비율 |
|--------|----------|------|
| person | 250 | 25% |
| helmet | 250 | 25% |
| no_helmet | 250 | 25% |
| fallen | 250 | 25% |
| **합계** | **1,000** | **100%** |

### 품질 비교
| 지표 | 1차 데이터 | 2차 데이터 |
|------|----------|----------|
| 평균 해상도 | 1280x720 | 1920x1080 |
| 평균 블러 점수 | 85 | 142 |
| 라벨 정확도 | 82% | 98% |
| 중복 비율 | 18% | 0% |

---

## 🛠️ 전처리 도구

### 1. convert_json_to_yolo.py
JSON 라벨을 YOLO txt 형식으로 변환

**사용법**:
```bash
python data/preprocessing/convert_json_to_yolo.py \
    --input Dataset/Data/Keypoint/1.Training/라벨링데이터 \
    --output Dataset/Data/Keypoint/1.Training/labels
```

### 2. filter_quality.py
품질 낮은 이미지 필터링

**사용법**:
```bash
python data/preprocessing/filter_quality.py \
    --input Dataset/Data/Keypoint/1.Training/원천데이터 \
    --output data/filtered/ \
    --blur-threshold 100
```

### 3. balance_dataset.py
클래스 균형 맞추기

**사용법**:
```bash
python data/preprocessing/balance_dataset.py \
    --input data/filtered/ \
    --output data/balanced/ \
    --samples-per-class 250
```

### 4. YOLO Mark (GUI 도구)
라벨 수동 검수 및 수정

**설치**:
```bash
git clone https://github.com/AlexeyAB/Yolo_mark.git
cd Yolo_mark
mkdir build && cd build
cmake .. && make
```

---

## 📁 최종 데이터 구조

```
data/
├── train/                    # 학습 데이터 (800장, 80%)
│   ├── images/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_0001.txt
│       ├── img_0002.txt
│       └── ...
│
└── valid/                    # 검증 데이터 (200장, 20%)
    ├── images/
    └── labels/
```

---

## 💡 핵심 인사이트

> **"데이터 양보다 품질과 균형이 더 중요하다"**

- 23,899장 → 1,000장으로 줄였지만 성능은 2배 향상 (mAP 45% → 87%)
- 클래스 불균형 해소가 정확도에 큰 영향
- 라벨 품질(바운딩 박스 정확도)이 모델 수렴 속도를 결정

---

## 🔗 관련 문서
- [학습 과정](../docs/training_process.md)
- [모델 정보](../models/README.md)
- [성능 분석](../docs/performance.md)
