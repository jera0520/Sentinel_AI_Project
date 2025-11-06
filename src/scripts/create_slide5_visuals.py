#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slide 5: 데이터 재정제 실행 및 결과 시각화
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 선별/제외 기준 시각화
# ============================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle('Data Refinement: Selection & Exclusion Criteria', 
              fontsize=16, fontweight='bold')

# 1-1. 선별 기준 (왼쪽)
ax1.axis('off')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# 제목
ax1.text(5, 9.5, 'SELECTION CRITERIA', 
         ha='center', fontsize=14, fontweight='bold', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

# 기준들
criteria_include = [
    ('1. Clear Object Visibility', 'Objects (helmet/person) clearly visible', '✓'),
    ('2. Person Included', 'Full person in frame', '✓'),
    ('3. Clear Status', 'Helmet wearing status unambiguous', '✓'),
    ('4. Moderate Background', 'Realistic construction site complexity', '✓'),
]

y_pos = 8
for title, desc, check in criteria_include:
    # 체크마크
    ax1.text(0.5, y_pos, check, fontsize=20, color='green', 
             fontweight='bold', ha='center')
    # 제목
    ax1.text(1.5, y_pos, title, fontsize=11, fontweight='bold', va='center')
    # 설명
    ax1.text(1.5, y_pos-0.4, desc, fontsize=9, color='gray', va='center')
    
    # 구분선
    if y_pos > 2:
        ax1.plot([0.3, 9.7], [y_pos-0.9, y_pos-0.9], 'k-', alpha=0.2)
    
    y_pos -= 2

ax1.text(5, 0.5, 'Target: ~200 images per category', 
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

# 1-2. 제외 기준 (오른쪽)
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# 제목
ax2.text(5, 9.5, 'EXCLUSION CRITERIA', 
         ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

# 기준들
criteria_exclude = [
    ('1. Blurry/Unclear', 'Poor image quality or focus', '✗'),
    ('2. Occluded Objects', 'Person/helmet heavily obscured', '✗'),
    ('3. Too Small Objects', 'Objects < 50px (too small)', '✗'),
    ('4. Ambiguous Status', 'Helmet status unclear', '✗'),
    ('5. Extreme Background', 'Too complex or too simple', '✗'),
]

y_pos = 8.5
for title, desc, check in criteria_exclude:
    # X마크
    ax2.text(0.5, y_pos, check, fontsize=20, color='red', 
             fontweight='bold', ha='center')
    # 제목
    ax2.text(1.5, y_pos, title, fontsize=11, fontweight='bold', va='center')
    # 설명
    ax2.text(1.5, y_pos-0.4, desc, fontsize=9, color='gray', va='center')
    
    # 구분선
    if y_pos > 2:
        ax2.plot([0.3, 9.7], [y_pos-0.9, y_pos-0.9], 'k-', alpha=0.2)
    
    y_pos -= 1.6

ax2.text(5, 0.5, 'Excluded: ~22,899 images', 
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

plt.tight_layout()
plt.savefig('slide5_selection_criteria.png', dpi=300, bbox_inches='tight')
print("✅ 선별/제외 기준 저장: slide5_selection_criteria.png")

# ============================================
# 2. 작업 프로세스 플로우
# ============================================
fig2, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

fig2.suptitle('Data Refinement Workflow', fontsize=16, fontweight='bold')

# 단계별 박스
steps = [
    {
        'y': 8.5, 
        'title': 'Step 1: Manual Selection',
        'desc': '• Review 23,899 images one by one\n• Select based on criteria\n• Ensure diversity',
        'color': '#FFE5CC'
    },
    {
        'y': 6.5,
        'title': 'Step 2: Category Distribution',
        'desc': '• Class: 200 each (helmet/no helmet/fallen)\n• Environment: 200 each (indoor/outdoor)\n• Background: 200 each (clean/complex)',
        'color': '#E5CCFF'
    },
    {
        'y': 4.5,
        'title': 'Step 3: YOLO Mark Validation',
        'desc': '• Load selected images\n• Verify bounding boxes\n• Correct label errors',
        'color': '#CCF5FF'
    },
    {
        'y': 2.5,
        'title': 'Step 4: Final Quality Check',
        'desc': '• Confirm balance\n• Check image quality\n• Validate labels',
        'color': '#CCFFCC'
    },
]

for i, step in enumerate(steps):
    # 박스
    box = FancyBboxPatch((1, step['y']-0.6), 8, 1.2, 
                         boxstyle="round,pad=0.1", 
                         facecolor=step['color'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # 제목
    ax.text(5, step['y']+0.3, step['title'], 
            ha='center', fontsize=12, fontweight='bold')
    
    # 설명
    ax.text(5, step['y']-0.2, step['desc'], 
            ha='center', fontsize=9, va='top')
    
    # 화살표 (마지막 제외)
    if i < len(steps) - 1:
        ax.annotate('', xy=(5, step['y']-0.7), xytext=(5, step['y']-1.2),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# 결과 박스
result_box = FancyBboxPatch((2, 0.3), 6, 1, 
                           boxstyle="round,pad=0.1",
                           facecolor='#FFD700', 
                           edgecolor='red', linewidth=3)
ax.add_patch(result_box)
ax.text(5, 0.8, 'Result: ~1,000 High-Quality Balanced Images', 
        ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('slide5_workflow.png', dpi=300, bbox_inches='tight')
print("✅ 작업 프로세스 저장: slide5_workflow.png")

# ============================================
# 3. Before/After 강조 비교
# ============================================
fig3, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

fig3.suptitle('Data Refinement: Before vs After Comparison', 
              fontsize=16, fontweight='bold')

# Before (왼쪽)
before_box = FancyBboxPatch((0.5, 3), 4, 5.5,
                           boxstyle="round,pad=0.2",
                           facecolor='#FFCCCC', alpha=0.3,
                           edgecolor='red', linewidth=3)
ax.add_patch(before_box)

ax.text(2.5, 8.2, 'BEFORE', ha='center', fontsize=16, fontweight='bold', color='red')
ax.text(2.5, 7.7, '(Initial Data)', ha='center', fontsize=10, style='italic')

before_data = [
    ('Total Images:', '23,899'),
    ('Class Balance:', 'Imbalanced ✗'),
    ('  - Helmet:', '18,000 (75%)'),
    ('  - No Helmet:', '2,000 (8%)'),
    ('  - Fallen:', '3,899 (17%)'),
    ('', ''),
    ('Environment:', 'Imbalanced ✗'),
    ('  - Indoor:', '20,000 (84%)'),
    ('  - Outdoor:', '3,899 (16%)'),
    ('', ''),
    ('Quality:', 'Mixed'),
    ('mAP:', '45% ✗'),
]

y = 7
for label, value in before_data:
    if label:
        ax.text(0.8, y, label, fontsize=10, fontweight='bold' if ':' in label else 'normal')
        ax.text(4, y, value, fontsize=10, ha='right',
               color='red' if '✗' in value else 'black')
    y -= 0.35

# 화살표
ax.annotate('REFINEMENT\nPROCESS', xy=(5.5, 5.5), xytext=(5.5, 6.5),
           ha='center', fontsize=14, fontweight='bold', color='green',
           arrowprops=dict(arrowstyle='->', lw=4, color='green'))

# After (오른쪽)
after_box = FancyBboxPatch((5.5, 3), 4, 5.5,
                          boxstyle="round,pad=0.2",
                          facecolor='#CCFFCC', alpha=0.3,
                          edgecolor='green', linewidth=3)
ax.add_patch(after_box)

ax.text(7.5, 8.2, 'AFTER', ha='center', fontsize=16, fontweight='bold', color='green')
ax.text(7.5, 7.7, '(Refined Data)', ha='center', fontsize=10, style='italic')

after_data = [
    ('Total Images:', '~1,000'),
    ('Class Balance:', 'Balanced ✓'),
    ('  - Helmet:', '200 (33%)'),
    ('  - No Helmet:', '200 (33%)'),
    ('  - Fallen:', '200 (33%)'),
    ('', ''),
    ('Environment:', 'Balanced ✓'),
    ('  - Indoor:', '200 (50%)'),
    ('  - Outdoor:', '200 (50%)'),
    ('', ''),
    ('Quality:', 'High ✓'),
    ('mAP:', '87% ✓'),
]

y = 7
for label, value in after_data:
    if label:
        ax.text(5.8, y, label, fontsize=10, fontweight='bold' if ':' in label else 'normal')
        ax.text(9, y, value, fontsize=10, ha='right',
               color='green' if '✓' in value else 'black',
               fontweight='bold' if '✓' in value or 'mAP' in label else 'normal')
    y -= 0.35

# 핵심 메시지
message_box = FancyBboxPatch((1, 0.5), 8, 1.8,
                            boxstyle="round,pad=0.2",
                            facecolor='#FFD700', alpha=0.5,
                            edgecolor='blue', linewidth=3)
ax.add_patch(message_box)

ax.text(5, 1.9, '💡 KEY INSIGHT', ha='center', fontsize=14, fontweight='bold', color='blue')
ax.text(5, 1.4, '"Quality & Balance > Quantity"', 
        ha='center', fontsize=16, fontweight='bold', style='italic')
ax.text(5, 0.9, 'Model performance improved from 45% to 87% mAP (+42%p)', 
        ha='center', fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('slide5_before_after_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Before/After 비교 저장: slide5_before_after_comparison.png")

# ============================================
# 4. 최종 데이터셋 구성 (파이 차트)
# ============================================
fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
fig4.suptitle('Final Dataset Composition (~1,000 images)', 
              fontsize=16, fontweight='bold')

# 4-1. 클래스 분포
ax1 = axes[0]
classes = ['Helmet\n(200)', 'No Helmet\n(200)', 'Fallen\n(200)']
sizes = [200, 200, 200]
colors1 = ['#66b3ff', '#ff9999', '#99ff99']

wedges1, texts1, autotexts1 = ax1.pie(sizes, labels=classes, colors=colors1,
                                        autopct='33.3%%', startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Class Distribution\n(Balanced ✓)', fontsize=12, fontweight='bold', color='green')

# 4-2. 환경 분포
ax2 = axes[1]
envs = ['Indoor\n(200)', 'Outdoor\n(200)']
sizes2 = [200, 200]
colors2 = ['#ffcc99', '#c2f0c2']

wedges2, texts2, autotexts2 = ax2.pie(sizes2, labels=envs, colors=colors2,
                                        autopct='50%%', startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Environment Distribution\n(Balanced ✓)', fontsize=12, fontweight='bold', color='green')

# 4-3. 배경 복잡도
ax3 = axes[2]
backgrounds = ['Clean\nBackground\n(200)', 'Complex\nBackground\n(200)']
sizes3 = [200, 200]
colors3 = ['#ff9999', '#99ccff']

wedges3, texts3, autotexts3 = ax3.pie(sizes3, labels=backgrounds, colors=colors3,
                                        autopct='50%%', startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax3.set_title('Background Complexity\n(Balanced ✓)', fontsize=12, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('slide5_final_dataset.png', dpi=300, bbox_inches='tight')
print("✅ 최종 데이터셋 구성 저장: slide5_final_dataset.png")

# ============================================
# 5. 핵심 메시지 강조
# ============================================
fig5, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 배경 그라데이션 효과 (박스들로)
for i in range(10):
    alpha = 0.1 + (i * 0.05)
    rect = Rectangle((0, i), 10, 1, facecolor='lightblue', alpha=alpha)
    ax.add_patch(rect)

# 메인 메시지
ax.text(5, 7.5, '💡 KEY MESSAGE', 
        ha='center', fontsize=20, fontweight='bold', color='darkblue')

# 핵심 문구
main_msg = '"Quality & Balance"\n>\n"Quantity"'
ax.text(5, 5, main_msg, 
        ha='center', va='center', fontsize=32, fontweight='bold',
        bbox=dict(boxstyle='round,pad=1', facecolor='gold', 
                 edgecolor='red', linewidth=4, alpha=0.8))

# 수치 비교
ax.text(5, 2.5, '23,899 images (imbalanced) → 1,000 images (balanced)', 
        ha='center', fontsize=14, fontweight='bold')
ax.text(5, 1.8, 'Result: mAP 45% → 87% (+42%p improvement)', 
        ha='center', fontsize=16, fontweight='bold', color='green')

# 결론
ax.text(5, 0.8, 'Data quality and balance are MORE important than quantity for model performance', 
        ha='center', fontsize=12, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('slide5_key_message.png', dpi=300, bbox_inches='tight')
print("✅ 핵심 메시지 저장: slide5_key_message.png")

print("\n" + "="*60)
print("Slide 5 시각 자료 생성 완료!")
print("="*60)
print("\n생성된 파일:")
print("1. slide5_selection_criteria.png - 선별/제외 기준")
print("2. slide5_workflow.png - 작업 프로세스")
print("3. slide5_before_after_comparison.png - Before/After 비교 (강조)")
print("4. slide5_final_dataset.png - 최종 데이터셋 구성")
print("5. slide5_key_message.png - 핵심 메시지")
print("="*60)

