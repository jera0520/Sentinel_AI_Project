# 🛡️ Sentinel AI Project

> 공사 현장 안전을 위한 AI 기반 실시간 모니터링 시스템

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📋 프로젝트 소개

Sentinel AI는 공사 현장에서 작업자의 안전을 실시간으로 모니터링하는 딥러닝 기반 영상 분석 시스템입니다.

### 핵심 기능
- 🎯 **실시간 객체 탐지**: YOLOv4 기반 작업자 및 안전장비 인식
- ⛑️ **헬멧 착용 검증**: 작업자의 헬멧 착용 여부 자동 확인
- 🚨 **낙상 사고 감지**: 쓰러진 작업자 즉시 탐지
- 📹 **영상 품질 향상**: FFmpeg 기반 실시간 업스케일링
- 🎬 **멀티프로세싱**: 병렬 처리를 통한 30 FPS 실시간 분석

### 주요 성과
- **검출 정확도**: 92% (mAP 87%)
- **처리 속도**: 30 FPS (실시간 처리)
- **오검출률**: 5% (기존 대비 25%p 개선)
- **모델 크기**: 245MB (최적화)

## 🚀 빠른 시작 (Quick Start)

### 사전 요구사항
- Python 3.10 이상
- CUDA 11.x (GPU 사용 시)
- FFmpeg

### 설치

```bash
# 저장소 클론
git clone https://github.com/jera0520/Sentinel_AI_Project.git
cd Sentinel_AI_Project

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r archive/sentinel_examples/requirements.txt
```

### 실행

```bash
# 데모 실행 (안전 모드)
python examples/run_demo.py

# 또는 직접 실행 (모델 파일이 있는 경우)
cd archive/sentinel_examples
python main_scale_v2.py
```

**참고**: 모델 가중치 파일(.weights)은 크기 제한으로 저장소에 포함되지 않습니다. 실제 실행을 위해서는 학습된 모델이 필요합니다.

## 📁 프로젝트 구조

```
Sentinel_AI_Project/
├── docs/                           # 📚 문서
│   ├── GUIDE_TWO_STAGE_PIPELINE.md # 2단계 파이프라인 가이드
│   ├── IMPLEMENTATION_SUMMARY.md   # 구현 요약
│   ├── PPT_04_프로젝트_수행_경과.md # 프로젝트 발표 자료
│   └── ...
├── src/
│   └── scripts/                    # 🔧 유틸리티 스크립트
│       ├── convert_json_to_yolo.py # JSON → YOLO 포맷 변환
│       ├── create_data_distribution.py # 데이터 분포 분석
│       └── create_slide5_visuals.py    # 시각화 생성
├── archive/                        # 📦 아카이브
│   ├── Dataset/                    # 학습 데이터셋 (구조만 보존)
│   ├── sentinel_examples/          # ⭐ 완성된 시스템
│   │   ├── main_scale_v2.py       # 메인 실행 파일
│   │   ├── lib/                   # 라이브러리 모듈
│   │   ├── model/                 # 학습된 모델 (*.weights 제외)
│   │   └── videos/                # 테스트 영상
│   └── *.png                      # 분석 결과 시각화
├── examples/                       # 💡 사용 예제
│   └── run_demo.py                # 안전 데모 실행 스크립트
├── .github/
│   └── workflows/
│       └── lint.yml               # CI/CD - 코드 린팅
├── README.md                      # 이 파일
├── LICENSE                        # MIT 라이선스
└── .gitignore                     # Git 제외 파일

```

## 🏗️ 시스템 아키텍처

### 처리 파이프라인

```
입력 영상 → VideoParser → DetectParser → DispEvent → 출력
            (업스케일)    (탐지/추적)     (시각화)
```

### 주요 컴포넌트

1. **VideoParser**: 영상 로드 및 FFmpeg 기반 업스케일링
2. **DetectParser**: YOLOv4 객체 탐지 + ByteTrack 추적
3. **DispEvent**: 결과 시각화 및 스코어링 시스템

### 사용 모델

- **person5l**: 사람 탐지 전문 모델
- **helmet_resort_v2**: 헬멧 착용 분류
- **falldown_v3**: 낙상 사고 감지

## 📊 성능 지표

| 항목 | 1차 학습 | 재학습 (최종) | 개선 |
|------|---------|-------------|------|
| mAP | 45% | 87% | +42%p |
| 정확도 | 60% | 92% | +32%p |
| 오검출률 | 30% | 5% | -25%p |
| 처리 속도 | 25 FPS | 30 FPS | +5 FPS |

## 🎓 핵심 기술

### TopDown 2단계 검출
1. **1단계**: 사람 영역 탐지 및 추출
2. **2단계**: 추출된 영역에서 헬멧 착용 여부 분류

### 스코어링 시스템
- 프레임별 누적 점수 계산 (-100 ~ 100)
- 일시적 오류에 강건한 판단
- 자원 효율적 경고 시스템

### 멀티프로세싱
- 3개 프로세스 병렬 실행
- Queue 기반 비동기 통신
- 실시간 처리 보장

## 📚 문서

자세한 내용은 다음 문서를 참고하세요:

- [2단계 파이프라인 가이드](docs/GUIDE_TWO_STAGE_PIPELINE.md): 비전공자를 위한 상세 구현 가이드
- [구현 요약](docs/IMPLEMENTATION_SUMMARY.md): 프로젝트 완료 요약
- [프로젝트 발표 자료](docs/PPT_04_프로젝트_수행_경과.md): 진행 과정 발표 자료
- [과제 요구사항](docs/assignment.txt): 원래 과제 명세
- [학습 노트](docs/gemini.md.txt): 개발 과정 기록

## 🛠️ 개발 환경

- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **CUDA**: 11.8 + cuDNN 8.9
- **언어**: Python 3.10
- **주요 라이브러리**: 
  - OpenCV 4.x
  - NumPy < 2.0
  - FFmpeg-python

## 💡 주요 인사이트

> **"데이터 양보다 품질과 균형이 더 중요하다"**

- 원본 23,899장 → 정제 1,000장
- 데이터 양 감소에도 성능 2배 향상
- 클래스 불균형 해소가 핵심

## 🔬 데이터

### 출처
- AI Hub "공사현장 안전장비 인식 이미지" 데이터셋

### 재정제 전략
1. 클래스 균형 맞춤
2. 품질 낮은 이미지 제거
3. 다양한 환경 조건 확보
4. 중복 및 유사 이미지 제거

### 결과
- **최종 데이터셋**: 1,000장 (균형 분포)
- **성능 향상**: mAP 45% → 87% (+42%p)

## 🤝 기여

이 프로젝트는 포트폴리오 목적으로 제작되었습니다. 
질문이나 제안사항이 있으시면 Issue를 등록해 주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👤 개발자

- **jera0520**
- 프로젝트: Sentinel AI - 공사 현장 안전 모니터링
- 기간: 2024.10

---

**Last Update**: 2025.01.06
