# -*- coding: utf-8 -*-
"""
论文图4：消融实验各组的F1分数与特征数量
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/ablation_results.csv")
df = df.sort_values("F1")  # 按F1升序便于绘图

x = np.arange(len(df))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10,6))
bars1 = ax1.bar(x - width/2, df["F1"], width, label="F1 Score (%)", color='skyblue')
ax1.set_ylabel("F1 Score (%)")
ax1.set_ylim(60, 101)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, df["特征数"], width, label="Feature Count", color='lightcoral')
ax2.set_ylabel("Number of Features")

ax1.set_xticks(x)
ax1.set_xticklabels(df["实验组"], rotation=25, ha='right')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Ablation Study: F1 Score vs Feature Count")
plt.tight_layout()
plt.savefig("results/ablation_figure.png", dpi=300, bbox_inches='tight')
plt.close()