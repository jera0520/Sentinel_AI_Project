# 🛡️ Sentinel AI Project

**공사 현장 안전 모니터링 AI 시스템**

실시간 영상 분석을 통한 작업자 안전 모니터링 솔루션으로, 헬멧 착용 여부 자동 감지와 작업자 쓰러짐 실시간 탐지 기능을 제공합니다.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 📋 프로젝트 개요

이 프로젝트는 YOLOv4 기반의 객체 탐지와 ByteTrack 추적 알고리즘을 활용하여 공사 현장의 안전 상황을 실시간으로 모니터링하는 시스템입니다.

### 주요 기능
- 🪖 **헬멧 착용 여부 자동 감지**
- 🚨 **작업자 쓰러짐 실시간 탐지**
- 🎥 **원본/업스케일 영상 비교 분석**

### 주요 성과
- **검출 정확도**: 92% (mAP 87%)
- **처리 속도**: 30 FPS (실시간)
- **오검출률**: 5% (기존 30% → 25%p 개선)
- **모델 최적화**: 데이터 1,000장으로 최고 성능 달성

---

## 🚀 빠른 시작 (Quick Start)

### 데모 실행

프로젝트의 데이터 분석 기능을 빠르게 체험해보세요:

```bash
# 프로젝트 정보 확인
python examples/run_demo.py --info

# 데모 실행 (Dry-run 모드)
python examples/run_demo.py --demo all --dry-run

# 데이터 분포 분석 실행
python examples/run_demo.py --demo distribution
```

### 전체 시스템 실행

완전한 AI 모니터링 시스템을 실행하려면 (모델 파일 필요):

```bash
cd archive/anu_example_3
python3 main_scale_v2.py
```

> **참고**: 전체 시스템 실행에는 YOLOv4 모델 파일(.weights)이 필요합니다.

---

## 📁 프로젝트 구조

```
Sentinel_AI_Project/
├── README.md                   # 프로젝트 개요
├── LICENSE                     # MIT 라이선스
├── .gitignore                  # Git 제외 파일 목록
│
├── docs/                       # 📚 문서
│   ├── IMPLEMENTATION_SUMMARY.md       # 구현 요약
│   ├── GUIDE_TWO_STAGE_PIPELINE.md     # 2단계 파이프라인 가이드
│   ├── PPT_04_프로젝트_수행_경과.md    # 발표 자료
│   ├── project.md                      # 프로젝트 상세
│   ├── gemini.md.txt                   # 학습 노트
│   └── assignment.txt                  # 과제 요구사항
│
├── src/                        # 💻 소스 코드
│   └── scripts/
│       ├── convert_json_to_yolo.py         # JSON→YOLO 변환
│       ├── create_data_distribution.py     # 데이터 분포 시각화
│       └── create_slide5_visuals.py        # 슬라이드 시각화
│
├── examples/                   # 🎯 예제 및 데모
│   └── run_demo.py                 # 데모 실행 스크립트
│
└── archive/                    # 📦 아카이브
    ├── anu_example_3/              # 완성 시스템 (메인 실행 파일)
    ├── Dataset/                    # 데이터셋 구조
    └── *.png                       # 시각화 결과물
```

---

## 🏗️ 시스템 아키텍처

### 기술 스택

**모델 & 알고리즘**
- **YOLOv4** (Darknet): 실시간 객체 탐지
- **ByteTrack**: Kalman Filter 기반 객체 추적
- **3개 전문 모델**: 사람 탐지 / 헬멧 분류 / 쓰러짐 감지

**핵심 기술**
- **TopDown 2단계 검출**: 환경 요소 배제 후 정밀 검출
- **스코어링 시스템**: 프레임 누적 판단 (-100~100점)
- **FFmpeg 파이프라인**: 실시간 영상 처리 및 업스케일링
- **멀티프로세싱**: 3개 프로세스 병렬 실행으로 성능 최적화

### 개발 환경

| 구분 | 사양 |
|------|------|
| **GPU** | NVIDIA RTX 3060 (12GB) |
| **CUDA** | 11.8 + cuDNN 8.9 |
| **OS** | Ubuntu 22.04 LTS |
| **언어** | Python 3.10 |
| **주요 라이브러리** | OpenCV, FFmpeg, NumPy, Matplotlib |

---

## 📊 데이터셋 & 성능

### 데이터 재정제 전략

- **출처**: AI Hub "공사현장 안전장비 인식 이미지"
- **원본**: 23,899장 → **최종**: 1,000장
- **전략**: 양보다 **품질과 균형** 우선

### 성능 비교

| 항목 | 1차 학습 | 재학습 (최종) | 개선 |
|------|---------|--------------|------|
| **mAP** | 45% | 87% | +42%p ⬆️ |
| **정확도** | 60% | 92% | +32%p ⬆️ |
| **오검출률** | 30% | 5% | -25%p ⬇️ |
| **처리 속도** | 25 FPS | 30 FPS | +5 FPS ⬆️ |

### 핵심 인사이트

> 💡 **"데이터 양보다 품질과 균형이 더 중요하다"**
> 
> 23,899장 → 1,000장으로 줄였지만, 성능은 2배 향상

---

## 📚 문서

자세한 내용은 다음 문서를 참조하세요:

- **[구현 요약](docs/IMPLEMENTATION_SUMMARY.md)**: 프로젝트 구현 완료 요약
- **[2단계 파이프라인 가이드](docs/GUIDE_TWO_STAGE_PIPELINE.md)**: TopDown 검출 방식 상세 설명
- **[프로젝트 수행 경과](docs/PPT_04_프로젝트_수행_경과.md)**: 발표 자료 및 진행 과정
- **[프로젝트 상세](docs/project.md)**: 기술적 세부 사항
- **[학습 노트](docs/gemini.md.txt)**: 개발 과정 기록

---

## 🎓 주요 학습 내용

- ✅ YOLOv4 커스텀 모델 학습 (8,000 iterations)
- ✅ 데이터 품질 개선 및 편향 제거
- ✅ TopDown 검출 방식 구현 (오검출 대폭 감소)
- ✅ 프레임 기반 스코어링 시스템 설계
- ✅ 멀티프로세싱 파이프라인 최적화

---

## 🔧 CI/CD

이 프로젝트는 GitHub Actions를 통한 자동 린팅을 지원합니다:

```yaml
# .github/workflows/lint.yml
- flake8를 사용한 Python 코드 품질 검사
- Push 및 PR 시 자동 실행
```

---

## 📄 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

```
MIT License - Copyright (c) 2025 jera0520
```

---

## 👤 개발자

**jera0520** - [GitHub Profile](https://github.com/jera0520)

- 프로젝트: Sentinel AI
- 목표: 공사 현장 안전 모니터링
- 개발 기간: 2024.10

---

## 📮 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 GitHub Issues를 통해 연락해주세요.

---

**Last Updated**: 2025.11.06
