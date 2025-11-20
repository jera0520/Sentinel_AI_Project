# 🛡️ Sentinel AI: 공사장 안전 모니터링 시스템

![demo](demo.gif)

**Sentinel AI**는 *Sentinel + AI*의 합성어로,

공사 현장의 **파수꾼(Sentinel)**이 되어 작업자의 안전을 **AI 기술로 실시간 모니터링**하는 시스템입니다.

---

## ✨ 프로젝트 소개
- **리뷰나 경험이 아닌**, **실시간 영상 분석**을 기반으로 공사장 작업자의 안전 상태를 자동으로 감지합니다.
- YOLOv4 객체 탐지 모델과 ByteTrack 추적 알고리즘을 결합하여 **헬멧 착용 여부·쓰러짐 감지**를 30 FPS로 처리합니다.
- 데이터 품질 개선을 통해 **mAP 45% → 87%** 향상을 달성했습니다.

<br>

### 🔑 주요 기능
- **실시간 객체 탐지**: 작업자(person), 헬멧 착용(helmet), 미착용(no_helmet), 쓰러짐(fallen) 4가지 클래스 동시 감지
- **TopDown 2단계 파이프라인**: 사람 탐지 후 헬멧 분류로 오탐지 25%p 감소
- **멀티프로세싱 구조**: 영상 처리·객체 탐지·결과 표시를 병렬 실행하여 30 FPS 달성
- **객체 추적**: ByteTrack + Kalman Filter로 각 작업자별 ID 부여 및 추적
- **영상 품질 개선**: FFmpeg 기반 실시간 업스케일로 원본/개선본 비교 분석
- **자동 크롭 저장**: 탐지된 객체를 클래스별로 자동 저장

<br>

### ⚙️ 내부 구현
- YOLOv4 (Darknet) 커스텀 학습 환경 구축
- AI Hub 데이터 전처리 (JSON → YOLO format 변환)
- 데이터 품질 최적화: 23,899장 → 1,000장 정제로 성능 2배 향상
- 3-Process 파이프라인: VideoParser → DetectParser → DisplayManager

---

## 🛠️ 기술 스택
- **Deep Learning**: YOLOv4 (Darknet), OpenCV
- **Object Tracking**: ByteTrack, Kalman Filter
- **Video Processing**: FFmpeg, multiprocessing
- **Development**: Python 3.10, CUDA 11.8, cuDNN 8.9
- **Hardware**: NVIDIA RTX 3060 (12GB VRAM)
- **OS**: Ubuntu 22.04 LTS

---

## 📥 데이터 수집 및 전처리

- **데이터 출처**
  - **AI Hub**: "공사현장 안전장비 인식 이미지" 데이터셋 활용
  - **원본 규모**: 23,899장의 이미지 및 JSON 라벨링 파일

- **전처리 과정**
  - JSON 형식 라벨을 YOLO 형식(txt)으로 변환
  - 클래스 불균형 해소: person, helmet, no_helmet, fallen 균등 분포
  - 품질 우선 정제: 흐릿하거나 중복된 이미지 제거
  - YOLO Mark 도구를 활용한 수동 검수 및 보정
  - 최종적으로 `이미지 파일(.jpg)` + `라벨 파일(.txt)` 형태로 구조화

- **데이터셋 규모**
  - **1차 학습**: 23,899장 (전체 데이터) → mAP 45%, 정확도 60%
  - **2차 학습**: 1,000장 (정제 데이터) → mAP 87%, 정확도 92%
  - **핵심 인사이트**: "양보다 품질과 균형이 중요" → 성능 2배 향상

---

## 📁 프로젝트 구조

```
Sentinel_AI_Project/
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/                          # 소스코드
│   ├── main.py                   # 메인 실행 파일
│   ├── lib/                      # 핵심 라이브러리
│   └── utils/                    # 유틸리티
│
├── models/                       # 학습된 모델
│   ├── README.md                 # 모델 정보
│   ├── person5l/                 # 사람 탐지 모델
│   ├── helmet_resort_v2/         # 헬멧 분류 모델
│   └── falldown_v3/              # 쓰러짐 감지 모델
│
├── data/                         # 데이터 관련
│   ├── README.md                 # 데이터 수집 과정
│   └── preprocessing/            # 전처리 스크립트
│
└── videos/                       # 테스트 영상
    └── sample_video.mp4
```

### 🗂️ 데이터 구조
**학습 데이터**: `이미지(.jpg)` + `라벨(.txt)`  
**라벨 형식**: `<class_id> <x_center> <y_center> <width> <height>` (YOLO 정규화 좌표)

---

## 👤 개발자 정보

<div align="center">

| **김진현** |
| :------: |
| [@jera0520](https://github.com/jera0520) |

</div>

### 🧩 개발 내역
- YOLOv4 커스텀 모델 학습 및 최적화 (8,000 iterations)
- 데이터 수집·정제·라벨링 (AI Hub 데이터 23,899장 → 1,000장)
- TopDown 2단계 파이프라인 설계 및 구현
- 멀티프로세싱 기반 실시간 영상 처리 시스템 구축
- ByteTrack 객체 추적 알고리즘 통합
- FFmpeg 기반 영상 업스케일 및 비교 분석 기능

---

## 👩‍💻 진행 과정

```
Week 1-2: 데이터 수집 및 전처리
  ├─ AI Hub 데이터 다운로드 (23,899장)
  ├─ JSON → YOLO 변환 스크립트 작성
  └─ YOLO Mark 도구로 라벨 검수

Week 3-4: 모델 학습 (1차)
  ├─ Darknet YOLOv4 환경 구축
  ├─ 전체 데이터로 학습 (8,000 iter)
  └─ 결과: mAP 45%, 오탐지율 30%

Week 5-6: 데이터 재정제
  ├─ 품질 분석 및 문제점 파악
  ├─ 1,000장 고품질 데이터 선별
  └─ 클래스 균형 분포 확보

Week 7-8: 모델 재학습 (2차)
  ├─ 정제 데이터로 재학습
  └─ 결과: mAP 87%, 오탐지율 5%

Week 9-10: 실시간 파이프라인 구축
  ├─ 멀티프로세싱 구조 설계
  ├─ ByteTrack 추적 통합
  └─ FFmpeg 업스케일 기능 추가

Week 11-12: 최적화 및 문서화
  ├─ 30 FPS 성능 달성
  ├─ TopDown 방식으로 오탐지 개선
  └─ 프로젝트 문서 정리
```

---

## ⚡ 실행 방법

### 1. 공통 준비 단계
- 레포지토리 클론 및 이동

    ```bash
    git clone https://github.com/jera0520/Sentinel_AI_Project.git
    cd Sentinel_AI_Project
    ```

- 라이브러리 설치

    ```bash
    pip install -r requirements.txt
    ```

- CUDA 및 cuDNN 설치 확인

    ```bash
    nvcc --version  # CUDA 11.8
    nvidia-smi      # GPU 확인
    ```

### 2. 기본 실행
- 폴더 이동

    ```bash
    cd src
    ```

- 실행
    ```bash
    python main.py --video ../videos/sample_video.mp4
    ```

### 3. 고급 옵션
- 크롭 저장 활성화

    ```bash
    python main.py --video test.mp4 --save-crops
    ```

- TopDown 2단계 모드

    ```bash
    python main.py --video test.mp4 --two-stage
    ```

---

## ✅ 테스트 시나리오

**입력 영상**: 공사장 작업자 10명, 헬멧 착용 7명, 미착용 3명  
**탐지 결과**:  
- Person: 10명 감지 (정확도 100%)
- Helmet: 7명 감지 (정확도 100%)
- No Helmet: 3명 감지 (정확도 100%)
- Fallen: 0명 (정상)

**처리 속도**: 30 FPS (실시간)  
**오탐지**: 배경의 헬멧 모양 객체 무시 (TopDown 효과)

---

## 📊 성능 지표

### 모델 성능

| 지표 | 1차 학습 (23,899장) | 2차 학습 (1,000장) | 개선 |
|------|-------------------|-------------------|------|
| mAP | 45% | 87% | +42%p |
| 정확도 | 60% | 92% | +32%p |
| 오탐지율 | 30% | 5% | -25%p |
| 학습 시간 | 12시간 | 4시간 | -8시간 |

### 실시간 처리 성능

| 항목 | 수치 |
|------|------|
| 처리 속도 | 30 FPS |
| 평균 지연 시간 | 33ms |
| GPU 사용률 | 65% |
| 메모리 사용량 | 8GB VRAM |

### 클래스별 정확도 (2차 학습)

| 클래스 | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| person | 95% | 94% | 0.945 |
| helmet | 92% | 89% | 0.905 |
| no_helmet | 88% | 91% | 0.895 |
| fallen | 90% | 87% | 0.885 |

---

## 🙌 기술적 도전 과제 및 해결

### 1. 데이터 품질 문제
**문제**: 23,899장 전체 데이터로 학습했으나 mAP 45%로 낮음  
**원인**: 흐릿한 이미지, 중복 샘플, 클래스 불균형  
**해결**: 품질 우선 정제 → 1,000장 선별 → mAP 87% 달성  
**교훈**: "데이터 양보다 품질과 균형이 중요"

### 2. 오탐지 문제 (False Positive)
**문제**: 배경의 헬멧 모양 객체를 헬멧으로 잘못 인식 (오탐지율 30%)  
**해결**: TopDown 2단계 파이프라인 도입  
  - Stage 1: 사람만 탐지  
  - Stage 2: 사람 영역 내에서만 헬멧 분류  
**결과**: 오탐지율 30% → 5% (25%p 개선)

### 3. 실시간 처리 속도
**문제**: 단일 프로세스에서 영상 처리·탐지·표시를 순차 실행 → 15 FPS  
**해결**: 멀티프로세싱으로 병렬 처리 구조 설계  
  - Process 1: 영상 디코딩 및 업스케일  
  - Process 2: 객체 탐지 및 추적  
  - Process 3: 결과 표시 및 저장  
**결과**: 15 FPS → 30 FPS (2배 향상)

### 4. CUDA/cuDNN 호환성 오류
**문제**: `CUDNN_STATUS_BAD_PARAM` 오류로 학습 중단  
**해결**: `yolov4-custom.cfg` 파일에 `cudnn_benchmark=0` 추가  
**교훈**: GPU 가속 라이브러리 버전 호환성 중요

### 5. 객체 추적 ID 끊김
**문제**: 프레임 간 객체 ID가 자주 바뀌어 추적 실패  
**해결**: ByteTrack + Kalman Filter 적용  
  - Kalman Filter로 객체 위치 예측  
  - IoU 기반 매칭으로 ID 유지  
**결과**: ID 유지율 85% → 95%

---

## 🙌 아쉬운 점 및 개선 방향

### 아쉬운 점

- **소규모 데이터셋**  
현재 1,000장의 정제 데이터로 학습하여 다양한 환경(야간, 악천후 등)에서의 성능 검증 부족
  - 향후 데이터 증강(Augmentation) 및 추가 수집 필요

- **단일 카메라 각도**  
공사장의 다양한 촬영 각도(위, 옆, 앞)를 충분히 반영하지 못함
  - 멀티뷰 데이터셋 확보 및 학습 필요

- **경량화 미진행**  
현재 모델 크기 245MB로 임베디드 기기(Jetson Nano 등)에서 실행 어려움
  - YOLOv4-tiny 또는 모바일 최적화 모델로 전환 필요

- **실시간 알림 기능 부재**  
탐지 결과를 화면에만 표시하고, 위험 상황 시 관리자에게 자동 알림 X
  - Slack/SMS 연동, 경고음 등 알림 시스템 추가 필요

- **통계 대시보드 부족**  
일일/주간 통계(헬멧 착용률, 위험 상황 발생 횟수 등) 시각화 미구현
  - 웹 대시보드(Streamlit, Flask) 추가로 관리 편의성 향상

### 개선 방향

- **모델 경량화**: YOLOv4 → YOLOv4-tiny 또는 YOLOv7-tiny 전환
- **멀티카메라 지원**: 여러 카메라 영상을 동시에 처리하는 기능
- **클라우드 연동**: AWS/GCP로 영상을 전송하여 중앙 모니터링
- **데이터 증강**: 밝기·회전·크롭 변환으로 데이터셋 확장
- **위험 점수화**: 헬멧 미착용 지속 시간, 쓰러짐 위치 등을 점수화하여 우선순위 부여

---

## 🙌 배운 점
- **YOLOv4 커스텀 학습**: Darknet 프레임워크 빌드부터 학습까지 전 과정 경험
- **데이터 중요성**: 양보다 품질이 성능에 미치는 영향을 실험으로 검증
- **멀티프로세싱 설계**: Python multiprocessing과 Queue를 활용한 병렬 처리 구조
- **객체 추적 알고리즘**: ByteTrack과 Kalman Filter의 원리 및 활용법
- **문제 해결 능력**: CUDA 오류, 오탐지 문제 등을 TopDown 방식으로 해결
- **프로젝트 문서화**: 기술적 도전과 해결 과정을 체계적으로 정리

---

## 📌 참고 자료
- AI Hub - 공사현장 안전장비 인식 이미지: [링크](https://aihub.or.kr/)
- Darknet YOLOv4 공식 GitHub: [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- ByteTrack 논문: [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- YOLO Mark 라벨링 도구: [AlexeyAB/Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)
- FFmpeg 공식 문서: [ffmpeg.org](https://ffmpeg.org/)

---

## 📧 연락처
**Kim Jin Hyeon**  
📧 jera0520@naver.com  
🔗 [GitHub](https://github.com/jera0520)

---

<p align="center">
  Made with ❤️ for construction site safety
</p>
