#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 분포 시각화 스크립트
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 클래스 분포 (정제 전)
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Distribution Analysis - Before Refinement', fontsize=16, fontweight='bold')

# 1-1. 클래스 편향 (파이 차트)
ax1 = axes[0, 0]
classes = ['Helmet\n(wearing)', 'No Helmet', 'Fallen', 'Others']
sizes = [18000, 2000, 3899, 0]  # 추정값
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0)  # 헬멧 강조

ax1.pie(sizes, explode=explode, labels=classes, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.set_title('Class Distribution (Imbalanced)', fontsize=12, fontweight='bold')

# 1-2. 클래스 편향 (바 차트)
ax2 = axes[0, 1]
x_pos = np.arange(len(classes))
ax2.bar(x_pos, sizes, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Class', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title('Class Distribution (Bar Chart)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(classes)
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(sizes):
    ax2.text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')

# 1-3. 환경 분포
ax3 = axes[1, 0]
envs = ['Indoor', 'Outdoor']
env_sizes = [20000, 3899]
env_colors = ['#ff9999', '#66b3ff']
explode_env = (0.1, 0)

ax3.pie(env_sizes, explode=explode_env, labels=envs, colors=env_colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax3.set_title('Environment Distribution (Imbalanced)', fontsize=12, fontweight='bold')

# 1-4. 정제 전 vs 후 비교
ax4 = axes[1, 1]
categories = ['Helmet', 'No Helmet', 'Fallen', 'Indoor', 'Outdoor']
before = [18000, 2000, 3899, 20000, 3899]
after = [200, 200, 200, 200, 200]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, before, width, label='Before', color='#ff9999', alpha=0.7)
bars2 = ax4.bar(x + width/2, after, width, label='After', color='#66b3ff', alpha=0.7)

ax4.set_xlabel('Category', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('Before vs After Refinement', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories, rotation=15, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장 완료: data_distribution_analysis.png")

# ============================================
# 2. 상세 비교 표 (별도 이미지)
# ============================================
fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# 표 데이터
table_data = [
    ['Category', 'Before\n(Imbalanced)', 'After\n(Balanced)', 'Change'],
    ['', '', '', ''],
    ['Class Distribution', '', '', ''],
    ['  - Helmet (wearing)', '18,000 (75%)', '200 (33%)', '-17,800'],
    ['  - No Helmet', '2,000 (8%)', '200 (33%)', '+198'],
    ['  - Fallen', '3,899 (17%)', '200 (33%)', '-3,699'],
    ['', '', '', ''],
    ['Environment', '', '', ''],
    ['  - Indoor', '20,000 (84%)', '200 (50%)', '-19,800'],
    ['  - Outdoor', '3,899 (16%)', '200 (50%)', '-3,699'],
    ['', '', '', ''],
    ['Background', '', '', ''],
    ['  - Clean', 'Minor', '200 (50%)', '+'],
    ['  - Complex (structures)', 'Majority', '200 (50%)', 'Balanced'],
    ['', '', '', ''],
    ['Total', '23,899', '~1,000', '-22,899'],
]

# 색상 설정
colors = []
for i, row in enumerate(table_data):
    if i == 0:  # 헤더
        colors.append(['#4472C4'] * 4)
    elif i in [2, 7, 11]:  # 카테고리 제목
        colors.append(['#70AD47'] * 4)
    elif i in [1, 6, 10, 14]:  # 빈 줄
        colors.append(['white'] * 4)
    elif i == len(table_data) - 1:  # 마지막 줄
        colors.append(['#FFC000'] * 4)
    else:
        colors.append(['#E7E6E6'] * 4)

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                cellColours=colors)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 헤더 스타일
for i in range(4):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold', color='white')

# 카테고리 제목 스타일
for row_idx in [2, 7, 11]:
    for col_idx in range(4):
        cell = table[(row_idx, col_idx)]
        cell.set_text_props(weight='bold', color='white')

plt.title('Data Refinement: Detailed Comparison', 
          fontsize=14, fontweight='bold', pad=20)

plt.savefig('data_distribution_table.png', dpi=300, bbox_inches='tight')
print("✅ 표 저장 완료: data_distribution_table.png")

# ============================================
# 3. 문제점 시각화
# ============================================
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Problems Identified in Initial Data', fontsize=16, fontweight='bold')

# 3-1. 클래스 불균형 문제
ax1 = axes3[0]
problem1_data = [75, 8, 17]
problem1_labels = ['Helmet\n75%', 'No Helmet\n8%', 'Fallen\n17%']
colors1 = ['#FF6B6B', '#4ECDC4', '#95E1D3']
ax1.pie(problem1_data, labels=problem1_labels, colors=colors1,
        autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
ax1.set_title('Problem 1:\nClass Imbalance', fontsize=12, fontweight='bold', color='red')

# 3-2. 환경 불균형 문제
ax2 = axes3[1]
problem2_data = [84, 16]
problem2_labels = ['Indoor\n84%', 'Outdoor\n16%']
colors2 = ['#FF6B6B', '#4ECDC4']
ax2.pie(problem2_data, labels=problem2_labels, colors=colors2,
        autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
ax2.set_title('Problem 2:\nEnvironment Imbalance', fontsize=12, fontweight='bold', color='red')

# 3-3. 배경 복잡도
ax3 = axes3[2]
complexity = ['Clean\nBackground', 'Complex\n(Structures,\nTools, Vehicles)']
complexity_ratio = [15, 85]
colors3 = ['#4ECDC4', '#FF6B6B']
bars = ax3.bar(complexity, complexity_ratio, color=colors3, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Percentage (%)', fontsize=10)
ax3.set_title('Problem 3:\nBackground Complexity', fontsize=12, fontweight='bold', color='red')
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, complexity_ratio):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('data_problems_identified.png', dpi=300, bbox_inches='tight')
print("✅ 문제점 시각화 저장 완료: data_problems_identified.png")

print("\n" + "="*50)
print("모든 시각화 완료!")
print("="*50)
print("생성된 파일:")
print("1. data_distribution_analysis.png - 전체 분포 분석")
print("2. data_distribution_table.png - 상세 비교 표")
print("3. data_problems_identified.png - 문제점 시각화")
print("="*50)

