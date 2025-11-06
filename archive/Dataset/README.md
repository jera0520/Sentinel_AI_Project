# Dataset 폴더

이 폴더에는 AI Hub에서 수집한 데이터셋이 포함됩니다.

## 📊 데이터 정보

- **출처**: AI Hub "공사현장 안전장비 인식 이미지"
- **원본 크기**: 23,899장
- **재정제 후**: 1,000장 (균형 분포)

## 📁 폴더 구조

```
Dataset/
└── Data/
    └── Keypoint/
        ├── 1.Training/
        │   ├── 원천데이터(zip)/     # 이미지 파일 (.jpg)
        │   └── 라벨링데이터(zip)/    # JSON 라벨 파일
        └── 2.Validation/
            ├── 원천데이터(zip)/
            └── 라벨링데이터(zip)/
```

## ⚠️ 주의사항

- 용량이 커서 Git에 포함되지 않음 (.gitignore)
- 필요시 AI Hub에서 직접 다운로드: https://www.aihub.or.kr/
- 데이터 사용 시 AI Hub 이용약관 준수 필요

## 🔄 데이터 전처리

`convert_json_to_yolo.py` 스크립트로 JSON → YOLO TXT 변환
