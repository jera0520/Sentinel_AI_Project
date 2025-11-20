# 학습된 모델 정보

이 디렉터리에는 Sentinel AI 프로젝트에서 사용하는 YOLOv4 기반의 객체 탐지 모델들이 저장되어 있습니다.

---

## 📦 모델 구성

### 1. person5l (사람 탐지 모델)
- **목적**: 공사장 내 작업자(사람) 탐지
- **클래스**: person (1개)
- **입력 크기**: 608x608
- **가중치 파일**: `person5l/model.weights` (약 245MB)
- **설정 파일**: `person5l/model.cfg`
- **정확도**: Precision 95%, Recall 94%

### 2. helmet_resort_v2 (헬멧 분류 모델)
- **목적**: 작업자의 헬멧 착용 여부 분류
- **클래스**: helmet, no_helmet (2개)
- **입력 크기**: 416x416
- **가중치 파일**: `helmet_resort_v2/model.weights` (약 245MB)
- **설정 파일**: `helmet_resort_v2/model.cfg`
- **정확도**: Precision 92%, Recall 89%

### 3. falldown_v3 (쓰러짐 감지 모델)
- **목적**: 작업자의 쓰러짐 상태 탐지
- **클래스**: fallen (1개)
- **입력 크기**: 416x416
- **가중치 파일**: `falldown_v3/model.weights` (약 245MB)
- **설정 파일**: `falldown_v3/model.cfg`
- **정확도**: Precision 90%, Recall 87%

---

## 🎓 학습 정보

### 학습 환경
- **프레임워크**: Darknet (YOLOv4)
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **CUDA**: 11.8
- **cuDNN**: 8.9
- **OS**: Ubuntu 22.04 LTS

### 학습 파라미터
```
batch = 64
subdivisions = 16
max_batches = 8000
learning_rate = 0.001
steps = 6400, 7200
scales = 0.1, 0.1
```

### 데이터 증강 (Augmentation)
- angle = 15°
- saturation = 1.5
- exposure = 1.5
- hue = 0.1
- flip = 1 (좌우 반전)

---

## 📊 성능 지표

### 전체 모델 (4-class)
| Metric | 1차 학습 | 2차 학습 (재정제) | 개선 |
|--------|---------|-----------------|------|
| mAP@0.5 | 45% | 87% | +42%p |
| 정확도 | 60% | 92% | +32%p |
| 학습 시간 | 12시간 | 4시간 | -8시간 |

### 클래스별 성능 (2차 학습)
| 클래스 | AP@0.5 | Precision | Recall |
|--------|--------|-----------|--------|
| person | 94% | 95% | 94% |
| helmet | 88% | 92% | 89% |
| no_helmet | 85% | 88% | 91% |
| fallen | 81% | 90% | 87% |

---

## 🔧 모델 사용법

### Python에서 로드
```python
from src.lib.yolo_detector import Yolo

# 사람 탐지 모델 로드
person_detector = Yolo(
    cfg_path='models/person5l/model.cfg',
    weights_path='models/person5l/model.weights',
    names_path='models/person5l/model.names'
)

# 탐지 실행
detections = person_detector.detect(image, threshold=0.5, nms=0.4)
```

### Darknet CLI에서 테스트
```bash
cd darknet
./darknet detector test \
    data/obj.data \
    ../models/person5l/model.cfg \
    ../models/person5l/model.weights \
    test_image.jpg
```

---

## 📥 모델 다운로드

**주의**: 가중치 파일(.weights)은 용량이 크므로 Git LFS를 사용하거나 별도로 다운로드해야 합니다.

```bash
# Git LFS 설치 (macOS)
brew install git-lfs
git lfs install

# 모델 파일 다운로드
git lfs pull
```

또는 직접 다운로드:
- [Google Drive 링크](#) (추후 업로드)

---

## 🚀 재학습 가이드

1. **데이터 준비**
   ```bash
   # JSON → YOLO 변환
   python data/preprocessing/convert_json_to_yolo.py
   ```

2. **Darknet 설정**
   ```bash
   cd darknet
   # obj.data, obj.names, train.txt, valid.txt 생성
   ```

3. **cfg 파일 수정**
   - `classes=4` (person, helmet, no_helmet, fallen)
   - `filters=27` ([yolo] 레이어 직전, 3곳)
   - `max_batches=8000` (classes * 2000)

4. **학습 시작**
   ```bash
   ./darknet detector train \
       data/obj.data \
       cfg/yolov4-custom.cfg \
       yolov4.conv.137 \
       -dont_show -map
   ```

5. **결과 확인**
   - 가중치: `backup/yolov4-custom_best.weights`
   - 학습 곡선: `chart.png`

---

## 📌 참고 사항

- **모델 크기**: 각 245MB (총 735MB)
- **추론 속도**: RTX 3060에서 30 FPS
- **메모리 요구사항**: 최소 8GB VRAM
- **라이선스**: MIT (학습 데이터는 AI Hub 정책 따름)

---

## 🔗 관련 문서
- [학습 과정 상세 가이드](../docs/training_process.md)
- [시스템 아키텍처](../docs/architecture.md)
- [성능 분석](../docs/performance.md)
