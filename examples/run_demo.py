#!/usr/bin/env python3
"""
Sentinel AI - 데모 실행 래퍼

이 스크립트는 실제 데이터나 모델 가중치 파일 없이도 안전하게 실행할 수 있는
포트폴리오 데모 래퍼입니다.

실제 시스템 실행을 위해서는:
1. archive/anu_example_3/ 디렉터리로 이동
2. 모델 가중치 파일(.weights) 준비 필요
3. python3 main_scale_v2.py 실행
"""

import os
import sys
from pathlib import Path


def print_banner():
    """프로젝트 배너 출력"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          🛡️  Sentinel AI - 건설 현장 안전 모니터링          ║
    ║                                                              ║
    ║          YOLOv4 기반 실시간 안전 감지 시스템                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """실행 환경 정보 표시"""
    print("📋 실행 환경 정보")
    print("─" * 60)
    print(f"  • Python 버전: {sys.version.split()[0]}")
    print(f"  • 작업 디렉터리: {os.getcwd()}")
    print("─" * 60)
    print()


def show_project_info():
    """프로젝트 정보 표시"""
    print("🎯 프로젝트 개요")
    print("─" * 60)
    print("  • 목표: 건설 현장 작업자 안전 실시간 모니터링")
    print("  • 검출: 헬멧 미착용, 작업자 쓰러짐 자동 감지")
    print("  • 성능: mAP 87%, 실시간 30 FPS 처리")
    print("  • 기술: YOLOv4, ByteTrack, FFmpeg Pipeline")
    print("─" * 60)
    print()


def show_structure():
    """프로젝트 구조 안내"""
    print("📁 프로젝트 구조")
    print("─" * 60)
    print("""
  Sentinel_AI_Project/
  ├── docs/                  # 📚 프로젝트 문서
  ├── src/scripts/           # 🔧 데이터 처리 스크립트
  ├── examples/              # 🎬 이 데모 파일
  └── archive/               # 📦 완성된 시스템 코드
      └── anu_example_3/     # ⭐ 메인 실행 파일 위치
          ├── main_scale_v2.py
          ├── model/         # YOLOv4 모델 (3종)
          └── videos/        # 테스트 영상
    """)
    print("─" * 60)
    print()


def show_usage():
    """사용 방법 안내"""
    print("🚀 실제 시스템 실행 방법")
    print("─" * 60)
    print("""
  ⚠️  주의: 실제 실행을 위해서는 모델 가중치 파일이 필요합니다!

  1. 아카이브 디렉터리로 이동:
     $ cd archive/anu_example_3

  2. 필요한 파일 준비:
     • 모델 가중치 (.weights 파일, ~245MB)
       - model/person5l/model.weights
       - model/helmet_resort_v2/model.weights
       - model/falldown_v3/model.weights
     • 테스트 영상 (videos/ 디렉터리)

  3. 실행:
     $ python3 main_scale_v2.py

  4. 종료: Ctrl+C
    """)
    print("─" * 60)
    print()


def show_scripts_help():
    """유틸리티 스크립트 도움말 표시"""
    print("🔧 유틸리티 스크립트")
    print("─" * 60)
    
    repo_root = Path(__file__).parent.parent
    scripts_dir = repo_root / "src" / "scripts"
    
    if scripts_dir.exists():
        scripts = list(scripts_dir.glob("*.py"))
        if scripts:
            print(f"  사용 가능한 스크립트 ({len(scripts)}개):")
            for script in sorted(scripts):
                print(f"    • {script.name}")
            print()
            print("  각 스크립트의 도움말:")
            print("    $ python src/scripts/<script_name> --help")
        else:
            print("  ℹ️  스크립트가 없습니다.")
    else:
        print("  ℹ️  scripts 디렉터리가 없습니다.")
    
    print("─" * 60)
    print()


def check_archive_system():
    """아카이브 시스템 존재 여부 확인"""
    print("📦 시스템 파일 확인")
    print("─" * 60)
    
    repo_root = Path(__file__).parent.parent
    archive_path = repo_root / "archive" / "anu_example_3"
    main_file = archive_path / "main_scale_v2.py"
    
    if archive_path.exists():
        print(f"  ✅ 시스템 디렉터리 발견: {archive_path.relative_to(repo_root)}")
        if main_file.exists():
            print(f"  ✅ 메인 파일 발견: {main_file.name}")
            print(f"     크기: {main_file.stat().st_size / 1024:.1f} KB")
        else:
            print(f"  ⚠️  메인 파일 없음: {main_file.name}")
    else:
        print(f"  ⚠️  시스템 디렉터리 없음: {archive_path}")
    
    print("─" * 60)
    print()


def main():
    """메인 함수"""
    print_banner()
    check_environment()
    show_project_info()
    show_structure()
    check_archive_system()
    show_scripts_help()
    show_usage()
    
    print("💡 추가 정보")
    print("─" * 60)
    print("  • README: 프로젝트 루트의 README.md 참조")
    print("  • 문서: docs/ 디렉터리 참조")
    print("  • 라이선스: LICENSE 파일 참조 (MIT)")
    print("─" * 60)
    print()
    print("✨ Sentinel AI 데모를 확인해 주셔서 감사합니다!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)
