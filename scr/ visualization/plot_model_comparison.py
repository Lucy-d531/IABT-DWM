# -*- coding: utf-8 -*-
"""
论文图2：模型对比的热力图、箱线图、ROC曲线、PR曲线
需要先运行 compare_models.py 得到对比结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pickle


# 示例：假设已有各模型的 y_proba 和 y_true
# y_true = ...
# probas = {"RandomForest": rf_proba, ...}

# 1. 热力图 (模型性能指标)
perf = pd.read_csv("results/table4_comparison.csv")
perf = perf.set_index("Model")[["Accuracy","Precision","Recall","F1","AUC"]]
plt.figure(figsize=(10,8))
sns.heatmap(perf, annot=True, fmt=".4f", cmap="RdYlGn", center=0.9)
plt.savefig("results/heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. 箱线图

# 3. ROC曲线
plt.figure(figsize=(8,6))
# for name, proba in probas.items():
#     fpr, tpr, _ = roc_curve(y_true, proba)
#     plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("results/roc_curves.png", dpi=300)
plt.close()

# 4. PR曲线
plt.figure(figsize=(8,6))
# for name, proba in probas.items():
#     prec, rec, _ = precision_recall_curve(y_true, proba)
#     plt.plot(rec, prec, label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("results/pr_curves.png", dpi=300)
plt.close()
