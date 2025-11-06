#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel AI Project Demo Script

This script demonstrates the project's data distribution analysis capabilities.
It can be run without requiring large datasets.
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def print_project_info():
    """Print project information."""
    print("=" * 70)
    print("🛡️  Sentinel AI Project - Demo")
    print("=" * 70)
    print("\n프로젝트: 공사 현장 안전 모니터링 AI 시스템")
    print("주요 기능:")
    print("  - 헬멧 착용 여부 자동 감지")
    print("  - 작업자 쓰러짐 실시간 탐지")
    print("  - 원본/업스케일 영상 비교 분석")
    print("\n주요 성과:")
    print("  - 검출 정확도: 92% (mAP 87%)")
    print("  - 처리 속도: 30 FPS (실시간)")
    print("  - 오검출률: 5% (기존 30% → 25%p 개선)")
    print("\n" + "=" * 70)


def demo_data_distribution(dry_run=False):
    """
    Demonstrate data distribution analysis.
    
    Args:
        dry_run: If True, only print what would be done without executing.
    """
    print("\n📊 데이터 분포 분석 데모")
    print("-" * 70)
    
    if dry_run:
        print("DRY RUN 모드: 실제 스크립트를 실행하지 않습니다.")
        print("\n실행될 명령:")
        print("  python src/scripts/create_data_distribution.py")
        print("\n기대 출력:")
        print("  - data_distribution_analysis.png")
        print("  - data_distribution_table.png")
        print("  - data_problems_identified.png")
        print("\n데이터셋 요구사항:")
        print("  - 이 스크립트는 데이터셋 없이 실행 가능합니다")
        print("  - 하드코딩된 통계 데이터를 사용하여 시각화를 생성합니다")
        print("\n설명:")
        print("  이 스크립트는 프로젝트의 데이터 정제 과정을 시각화합니다.")
        print("  정제 전후의 클래스 분포, 환경 분포, 배경 유형 등을 비교합니다.")
        return True
    
    # Try to import and run the actual script
    try:
        print("실행 중: create_data_distribution.py")
        print("(이 스크립트는 matplotlib를 사용하여 차트를 생성합니다)")
        
        # Import and run the script
        from scripts import create_data_distribution
        
        print("\n✅ 데이터 분포 분석 완료!")
        print("생성된 파일을 확인하세요:")
        print("  - data_distribution_analysis.png")
        print("  - data_distribution_table.png") 
        print("  - data_problems_identified.png")
        return True
        
    except ImportError as e:
        print(f"\n⚠️  경고: 스크립트를 가져올 수 없습니다: {e}")
        print("필요한 종속성을 설치하세요:")
        print("  pip install matplotlib numpy")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False


def demo_convert_json():
    """Demonstrate JSON to YOLO conversion (dry run only)."""
    print("\n🔄 JSON to YOLO 변환 데모")
    print("-" * 70)
    print("DRY RUN 모드: 데이터셋이 필요한 작업입니다.")
    print("\n실행될 명령:")
    print("  python src/scripts/convert_json_to_yolo.py")
    print("\n기능:")
    print("  - JSON 형식의 어노테이션을 YOLO 형식으로 변환")
    print("  - 바운딩 박스 좌표를 정규화")
    print("  - 클래스 ID 매핑")
    print("\n데이터셋 요구사항:")
    print("  - 입력: JSON 어노테이션 파일")
    print("  - 출력: .txt 형식의 YOLO 어노테이션")


def demo_slide5_visuals():
    """Demonstrate slide 5 visuals creation (dry run only)."""
    print("\n📊 슬라이드 5 시각화 데모")
    print("-" * 70)
    print("DRY RUN 모드: 데이터셋이 필요한 작업입니다.")
    print("\n실행될 명령:")
    print("  python src/scripts/create_slide5_visuals.py")
    print("\n기능:")
    print("  - 데이터 재정제 과정 시각화")
    print("  - 선택 기준, 워크플로우, 전후 비교 차트 생성")
    print("\n기대 출력:")
    print("  - slide5_workflow.png")
    print("  - slide5_selection_criteria.png")
    print("  - slide5_before_after_comparison.png")
    print("  - slide5_final_dataset.png")
    print("  - slide5_key_message.png")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Sentinel AI Project 데모 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python examples/run_demo.py --info
  python examples/run_demo.py --demo all --dry-run
  python examples/run_demo.py --demo distribution
        """
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='프로젝트 정보 출력'
    )
    
    parser.add_argument(
        '--demo',
        choices=['all', 'distribution', 'convert', 'slide5'],
        default='all',
        help='실행할 데모 선택 (기본값: all)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제로 실행하지 않고 설명만 출력'
    )
    
    args = parser.parse_args()
    
    # Always show project info first
    print_project_info()
    
    if args.info:
        print("\n자세한 정보는 다음 문서를 참조하세요:")
        print("  - README.md")
        print("  - docs/IMPLEMENTATION_SUMMARY.md")
        print("  - docs/GUIDE_TWO_STAGE_PIPELINE.md")
        return 0
    
    # Run demos
    success = True
    
    if args.demo in ['all', 'distribution']:
        if not demo_data_distribution(dry_run=args.dry_run):
            success = False
    
    if args.demo in ['all', 'convert']:
        demo_convert_json()
    
    if args.demo in ['all', 'slide5']:
        demo_slide5_visuals()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ 데모 완료!")
    else:
        print("⚠️  일부 데모가 실패했습니다. 위의 메시지를 확인하세요.")
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
