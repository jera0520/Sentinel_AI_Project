# 🎓 Sentinel AI Project - 구현 완료 요약

## 📅 작업 일자: 2024-10-14

---

## ✅ 완료된 작업

### Phase 1: 현재 모델로 과제 완료 (완료)

#### 1. 모델 학습 및 적용
- ✅ **4-class YOLOv4 모델 학습 완료**
  - 클래스: person, helmet, no_helmet, fallen
  - 학습 데이터: 23,899개 이미지
  - 학습 iteration: 8,000
  - 모델 위치: `darknet/backup/yolov4-custom_best.weights` (245MB)

- ✅ **anu_example에 새 모델 적용**
  - 기존: person만 탐지 (person5l)
  - 변경: 4개 클래스 탐지 (custom_4class)
  - 경로: `anu_example/model/custom_4class/`

#### 2. 과제 2: 객체 크롭 저장 기능 구현
- ✅ **main_scale.py 수정 완료**
  - DetectParser 클래스에 크롭 저장 기능 추가
  - 클래스별 자동 분류 저장
  - 중복 방지 메커니즘
  - 세션별 폴더 자동 생성

- ✅ **기능 세부사항**
  - 저장 주기: 30프레임마다 (약 1초)
  - 파일명 형식: `{클래스명}_id{추적ID}_{카운터}.jpg`
  - 저장 위치: `detected_crops/{타임스탬프}/`
  - 명령행 옵션: `--save-crops` / `--no-save-crops`

---

## 📂 프로젝트 구조

```
Sentinel_AI_Project/
├── darknet/                          # YOLOv4 학습 환경
│   ├── cfg/yolov4-custom.cfg        # 커스텀 모델 설정 (4 classes)
│   ├── data/
│   │   ├── obj.names                # person, helmet, no_helmet, fallen
│   │   ├── obj.data
│   │   ├── train.txt                # 23,899개 이미지
│   │   └── valid.txt
│   └── backup/
│       └── yolov4-custom_best.weights  # 학습된 모델
│
├── anu_example/                      # 실행 환경
│   ├── main_scale.py                # ⭐ 메인 실행 파일 (수정됨)
│   ├── model/
│   │   ├── person5l/                # 기존 모델 (person만)
│   │   └── custom_4class/           # ⭐ 새 모델 (4 classes)
│   │       ├── model.cfg
│   │       ├── model.weights        # darknet에서 복사
│   │       ├── model.names
│   │       └── model.data
│   ├── detected_crops/              # ⭐ 크롭 이미지 저장 위치
│   │   └── {timestamp}/
│   │       ├── person/
│   │       ├── helmet/
│   │       ├── no_helmet/
│   │       └── fallen/
│   ├── lib/                         # 라이브러리
│   ├── test.mp4                     # 테스트 비디오
│   ├── venv/                        # Python 가상환경
│   └── README_CROP.md               # 사용 가이드
│
└── Dataset/                          # 학습 데이터
    └── Data/Keypoint/
        ├── 1.Tranining/labels/      # 학습용 이미지 및 라벨
        └── 2.Vaildation/원천데이터(zip)/  # 검증용 데이터
```

---

## 🚀 사용 방법

### 1. 기본 실행 (크롭 저장 활성화)
```bash
cd /home/jera/Sentinel_AI_Project/anu_example
source venv/bin/activate
python3 main_scale.py
```

### 2. 크롭 저장 비활성화
```bash
python3 main_scale.py --no-save-crops
```

### 3. 결과 확인
```bash
# 크롭된 이미지 확인
ls -lh detected_crops/*/person/
ls -lh detected_crops/*/helmet/
ls -lh detected_crops/*/no_helmet/
ls -lh detected_crops/*/fallen/

# 총 저장된 이미지 수
find detected_crops -name "*.jpg" | wc -l
```

---

## 🎯 주요 특징

### 현재 구현 (Single-Stage)
```
입력 영상 → YOLOv4 (4-class) → 추적 → 크롭 저장
                ↓
    person, helmet, no_helmet, fallen
```

**장점:**
- ✅ 간단하고 빠름
- ✅ 실시간 처리 가능
- ✅ 이미 학습 완료

**단점:**
- ⚠️ 복잡한 배경에서 정확도 낮을 수 있음
- ⚠️ 작은 객체(helmet) 탐지 어려움

---

## 🔄 Phase 2: 2단계 파이프라인 개선 계획 (추후)

### 개선 방향: Two-Stage Detection

```
입력 영상 → Stage 1: Person Detector → 크롭 → Stage 2: Helmet Classifier
                ↓                              ↓
            person, fallen               helmet / no_helmet
```

### 예상 개선 사항
1. **정확도 향상**: 배경과 객체 분리 후 분류
2. **False Positive 감소**: 헬멧만 단독으로 있는 경우 제외
3. **작은 객체 탐지 향상**: 크롭 후 분류로 해상도 확보

### 필요 작업
1. ✏️ 데이터 재가공
   - person bbox 크롭
   - helmet/no_helmet 라벨링

2. 🤖 Helmet Classifier 학습
   - MobileNetV2 또는 EfficientNet
   - Binary classification

3. 🔧 파이프라인 구현
   - DetectParser 수정
   - 2단계 추론 로직 추가

---

## 📊 성능 평가 (추후 수행)

### 평가 지표
- [ ] Stage 1 (Person Detection): mAP, Precision, Recall
- [ ] Stage 2 (Helmet Classification): Accuracy, F1-Score
- [ ] End-to-End 정확도
- [ ] FPS (처리 속도)

### 테스트 데이터셋
- Validation 데이터 활용
- 다양한 조명 조건
- 다양한 거리/각도

---

## 🐛 알려진 이슈 및 해결

### 1. 모델 테스트 시 아무것도 탐지되지 않음
**원인:** 테스트 이미지에 객체가 없거나, placeholder 경로 사용
**해결:** 실제 이미지 경로 사용, threshold 조정

### 2. 명령어 실행 오류
**원인:** 한 줄에 여러 명령어 입력
**해결:** 각 명령어를 엔터로 구분하여 실행

---

## 📚 참고 문서

1. `assignment.txt` - 과제 요구사항
2. `gemini.md.txt` - 학습 과정 노트
3. `project.md` - 프로젝트 구조 분석
4. `README_CROP.md` - 크롭 기능 사용법

---

## 🎓 학습 내용

### YOLOv4 커스텀 모델 학습
- Darknet 프레임워크 사용
- AI Hub 데이터 정제 및 변환 (JSON → TXT)
- YOLO Mark를 이용한 라벨링 검수
- cfg 파일 수정 (classes, filters, max_batches)
- cuDNN 오류 해결

### 멀티프로세싱 파이프라인
- VideoParser: 영상 로드 및 업스케일
- DetectParser: 객체 탐지 및 추적
- DispEvent: 결과 표시
- Queue를 통한 프로세스 간 통신

### ByteTrack 객체 추적
- Track ID 관리
- 매칭 알고리즘
- 미스 카운트 처리

---

## 👥 기여자

- 작성자: Sentinel AI Project Team
- 지도: gemini.md.txt 기반 학습
- 목표: 공사 현장 안전 모니터링

---

## 📝 다음 할 일

- [ ] main_scale.py 실행 및 테스트
- [ ] 크롭된 이미지 품질 확인
- [ ] 성능 평가 수행
- [ ] 2단계 파이프라인 데이터 준비
- [ ] Helmet Classifier 학습
- [ ] 최종 보고서 작성

---

**마지막 업데이트:** 2024-10-14
**버전:** 1.0 (Single-Stage with Crop)
