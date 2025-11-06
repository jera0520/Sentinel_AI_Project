# 🛡️ Sentinel AI - 건설 현장 안전 모니터링 시스템

> YOLOv4 기반 실시간 작업자 안전 모니터링 AI 시스템

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📌 프로젝트 소개

건설 현장에서 작업자의 안전을 실시간으로 모니터링하는 AI 시스템입니다. YOLOv4 딥러닝 모델을 활용하여 헬멧 미착용과 작업자 쓰러짐을 자동으로 감지합니다.

### 🎯 주요 성과

- **검출 정확도**: mAP 87% (45% → 87%, +42%p 개선)
- **실시간 처리**: 30 FPS
- **오검출 감소**: 30% → 5% (-25%p 개선)
- **데이터 효율**: 23,899장 → 1,000장 정제로 성능 2배 향상

## ✨ 핵심 기능

### 1. 헬멧 착용 감지
- TopDown 2단계 검출 방식으로 환경 노이즈 제거
- 사람 검출 후 헬멧 영역만 집중 분석
- 프레임별 스코어링 시스템(-100~100점)으로 오검출 최소화

### 2. 작업자 쓰러짐 탐지
- 실시간 자세 분석
- ByteTrack 기반 객체 추적 (Kalman Filter)
- 즉각적인 위험 상황 알림

### 3. 멀티 스케일 분석
- 원본/업스케일 영상 동시 처리
- FFmpeg 파이프라인 최적화
- 3프로세스 병렬 처리

## 🚀 Quick Start

### 실행 환경
```bash
# Python 3.10+, CUDA 11.8+, cuDNN 8.9+ 필요
```

### 데모 실행
```bash
# 데모 스크립트 (데이터 없이도 안전하게 실행)
python examples/run_demo.py
```

### 전체 시스템 (아카이브)
```bash
# 주의: 실제 실행에는 모델 가중치(.weights) 파일과 영상 데이터 필요
cd archive/anu_example_3
python3 main_scale_v2.py
```

## 📁 프로젝트 구조

```
Sentinel_AI_Project/
├── docs/                          # 📚 프로젝트 문서
│   ├── GUIDE_TWO_STAGE_PIPELINE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── PPT_04_프로젝트_수행_경과.md
│   └── ...
├── src/
│   └── scripts/                   # 🔧 데이터 처리 스크립트
│       ├── convert_json_to_yolo.py
│       ├── create_data_distribution.py
│       └── create_slide5_visuals.py
├── examples/
│   └── run_demo.py               # 🎬 데모 실행 래퍼
├── archive/                       # 📦 아카이브 (완성된 시스템 코드)
│   ├── anu_example_3/            # 최종 구현 코드
│   │   ├── main_scale_v2.py     # 메인 실행 파일
│   │   ├── model/               # YOLOv4 모델 (3종)
│   │   └── videos/              # 테스트 영상
│   ├── Dataset/                  # 학습 데이터
│   └── *.png                     # 분석 시각 자료
├── .github/
│   └── workflows/
│       └── lint.yml              # CI/CD 린트 워크플로
├── README.md                      # 📖 이 문서
├── .gitignore
└── LICENSE                        # MIT License
```

## 🎓 기술 스택

### AI/ML
- **YOLOv4** (Darknet): 객체 검출 백본
- **ByteTrack**: 다중 객체 추적
- **3-Stage Model Pipeline**:
  - `person5l`: 사람 검출
  - `helmet_resort_v2`: 헬멧 분류
  - `falldown_v3`: 쓰러짐 감지

### 개발 환경
- **Python 3.10**: 메인 언어
- **OpenCV**: 영상 처리
- **FFmpeg**: 실시간 스트리밍
- **NumPy**: 수치 연산
- **CUDA 11.8 + cuDNN 8.9**: GPU 가속

## 💡 핵심 인사이트

### 데이터 품질 > 데이터 양
23,899장의 원본 데이터를 1,000장으로 정제하면서 오히려 성능이 2배 향상되었습니다.

**개선 전략:**
- 클래스 불균형 해소
- 저품질 이미지 제거
- 중복/유사 이미지 제거
- 다양한 각도/조명 조건 확보

**결과:**
| 지표 | 1차 학습 | 재학습 | 개선폭 |
|------|---------|--------|-------|
| mAP | 45% | 87% | +42%p |
| 정확도 | 60% | 92% | +32%p |
| 오검출 | 30% | 5% | -25%p |

### TopDown 방식의 효과
배경 객체(소화기, 교통콘 등)를 사람 영역으로 오인하는 문제를 해결하기 위해 2단계 검출 방식을 도입했습니다.

1. **1단계**: 사람 검출 (person5l)
2. **2단계**: 검출된 사람 영역 내에서만 헬멧 확인 (helmet_resort_v2)

→ 오검출률 30% → 5%로 대폭 감소

## 📚 문서

자세한 내용은 [docs/](./docs/) 디렉터리를 참고하세요:

- **구현 가이드**: [GUIDE_TWO_STAGE_PIPELINE.md](./docs/GUIDE_TWO_STAGE_PIPELINE.md)
- **구현 요약**: [IMPLEMENTATION_SUMMARY.md](./docs/IMPLEMENTATION_SUMMARY.md)
- **발표 자료**: [PPT_04_프로젝트_수행_경과.md](./docs/PPT_04_프로젝트_수행_경과.md)

## 🔧 개발 도구

프로젝트에 포함된 유틸리티 스크립트:

```bash
# JSON 어노테이션을 YOLO 형식으로 변환
python src/scripts/convert_json_to_yolo.py

# 데이터 분포 분석
python src/scripts/create_data_distribution.py

# 시각 자료 생성
python src/scripts/create_slide5_visuals.py
```

## 🏆 성능 벤치마크

### 하드웨어
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **CPU**: Intel i7 (8 cores)
- **RAM**: 32GB

### 결과
- **추론 속도**: 30 FPS (실시간)
- **모델 크기**: 245MB (person5l)
- **메모리 사용량**: ~8GB VRAM
- **지연 시간**: < 35ms/frame

## ⚠️ 주의사항

이 리포지토리는 포트폴리오 전시용으로 구조화되었습니다:

- **archive/** 디렉터리: 실제 동작하는 완성 코드가 포함되어 있으나, 실행을 위해서는 대용량 모델 가중치 파일(.weights, ~245MB)과 영상 데이터가 필요합니다.
- **examples/**: 데이터 없이도 구조를 이해할 수 있는 데모 래퍼를 제공합니다.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 참조하세요.

## 👤 개발자

**jera0520**

프로젝트 기간: 2024.10  
최종 업데이트: 2025.01

---

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!
