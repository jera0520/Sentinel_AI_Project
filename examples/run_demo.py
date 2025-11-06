#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel AI Project - Safe Demo Runner

이 스크립트는 모델 파일이나 비디오 파일이 없는 환경에서도
안전하게 실행할 수 있는 데모 래퍼입니다.
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """필수 요구사항 확인"""
    print("=" * 60)
    print("🛡️  Sentinel AI Project - Demo Runner")
    print("=" * 60)
    
    # 프로젝트 루트 찾기
    current_dir = Path(__file__).parent.parent
    sentinel_dir = current_dir / "archive" / "sentinel_examples"
    
    print(f"\n📂 프로젝트 경로: {current_dir}")
    print(f"📂 시스템 경로: {sentinel_dir}")
    
    # 디렉토리 존재 확인
    if not sentinel_dir.exists():
        print("\n❌ 오류: sentinel_examples 디렉토리를 찾을 수 없습니다.")
        print(f"   예상 경로: {sentinel_dir}")
        return False
    
    # 메인 스크립트 확인
    main_script = sentinel_dir / "main_scale_v2.py"
    if not main_script.exists():
        print(f"\n❌ 오류: 메인 실행 파일을 찾을 수 없습니다.")
        print(f"   예상 경로: {main_script}")
        return False
    
    # 모델 디렉토리 확인
    model_dir = sentinel_dir / "model"
    if not model_dir.exists():
        print(f"\n⚠️  경고: 모델 디렉토리를 찾을 수 없습니다.")
        print(f"   예상 경로: {model_dir}")
    else:
        # 모델 파일 확인
        model_weights_found = False
        for model_path in model_dir.rglob("*.weights"):
            model_weights_found = True
            print(f"✅ 모델 발견: {model_path.relative_to(sentinel_dir)}")
        
        if not model_weights_found:
            print(f"\n⚠️  경고: 학습된 모델 파일(*.weights)을 찾을 수 없습니다.")
            print(f"   모델 파일은 크기 제한으로 저장소에 포함되지 않습니다.")
            print(f"   실제 실행을 위해서는 학습된 모델이 필요합니다.")
    
    # 비디오 디렉토리 확인
    video_dir = sentinel_dir / "videos"
    if video_dir.exists():
        videos = list(video_dir.glob("*.mp4"))
        if videos:
            print(f"\n✅ 테스트 비디오 {len(videos)}개 발견")
        else:
            print(f"\n⚠️  경고: 테스트 비디오 파일을 찾을 수 없습니다.")
    else:
        print(f"\n⚠️  경고: videos 디렉토리를 찾을 수 없습니다.")
    
    # requirements.txt 확인
    req_file = sentinel_dir / "requirements.txt"
    if req_file.exists():
        print(f"\n✅ 의존성 파일 발견: {req_file.relative_to(sentinel_dir)}")
        print("\n📦 필요한 패키지를 설치하려면 다음 명령을 실행하세요:")
        print(f"   pip install -r {req_file}")
    else:
        print(f"\n⚠️  경고: requirements.txt를 찾을 수 없습니다.")
    
    return True


def show_usage():
    """사용법 안내"""
    print("\n" + "=" * 60)
    print("📖 사용 방법")
    print("=" * 60)
    print("\n1. 의존성 설치:")
    print("   pip install -r archive/sentinel_examples/requirements.txt")
    print("\n2. 직접 실행 (모델 파일이 있는 경우):")
    print("   cd archive/sentinel_examples")
    print("   python main_scale_v2.py")
    print("\n3. 또는 다음 옵션으로 실행:")
    print("   python main_scale_v2.py [옵션]")
    print("\n   옵션:")
    print("   --help              도움말 표시")
    print("   --input VIDEO       입력 비디오 파일 지정")
    print("   --output OUTPUT     출력 비디오 파일 지정")
    print("\n" + "=" * 60)
    print("⚠️  주의: 실제 실행을 위해서는 학습된 모델 파일이 필요합니다.")
    print("=" * 60)


def main():
    """메인 함수"""
    try:
        # 요구사항 확인
        if not check_requirements():
            print("\n❌ 필수 파일이 누락되어 실행할 수 없습니다.")
            return 1
        
        # 사용법 안내
        show_usage()
        
        print("\n💡 팁:")
        print("   - 이 스크립트는 데모 환경 확인용입니다.")
        print("   - 실제 실행은 archive/sentinel_examples 디렉토리에서 하세요.")
        print("   - 모델 파일(.weights)은 별도로 학습하거나 제공받아야 합니다.")
        
        print("\n✅ 환경 확인 완료!\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
