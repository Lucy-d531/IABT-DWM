# -*- coding: utf-8 -*-
"""
论文 五 可解释性分析
全局特征重要性、依赖图、决策力图（图5-7）
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os

os.makedirs("results/shap", exist_ok=True)

df = pd.read_csv("data/processed/filtered_features_dataset.csv")
X = df.drop('label', axis=1)
y = df['label']

# 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 使用论文最佳参数训练（可改为从文件中加载）
model = Pipeline([
    ('scaler', StandardScaler()),
    ('tomek', TomekLinks()),
    ('xgb', XGBClassifier(random_state=42, n_estimators=500, max_depth=5,
                          learning_rate=0.2, gamma=0.015, subsample=0.842,
                          colsample_bytree=0.5, eval_metric='logloss', verbosity=0))
])
model.fit(X_train, y_train)

# 用测试集子集加速（取200个样本）
X_sample = X_test.sample(200, random_state=42)
explainer = shap.TreeExplainer(model.named_steps['xgb'])
shap_values = explainer.shap_values(X_sample)

# 图5(b) 摘要图
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("results/shap/summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 图5(a) 条形图
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/shap/bar_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 图6 依赖图（前4重要特征）
mean_abs = np.abs(shap_values).mean(0)
top4_idx = np.argsort(mean_abs)[-4:][::-1]
for idx in top4_idx:
    feat = X_sample.columns[idx]
    plt.figure(figsize=(8,5))
    shap.dependence_plot(idx, shap_values, X_sample, show=False)
    plt.title(f"SHAP Dependence: {feat}")
    plt.tight_layout()
    plt.savefig(f"results/shap/dependence_{feat}.png", dpi=300)
    plt.close()

# 图7 力图（选择一个暗网节点和一个正常节点）
y_pred = model.predict(X_test)
dark_idx = np.where((y_test == 1) & (y_pred == 1))[0][0]
normal_idx = np.where((y_test == 0) & (y_pred == 0))[0][0]

for idx, name in [(dark_idx, "darknet"), (normal_idx, "normal")]:
    shap.force_plot(explainer.expected_value, shap_values[0,:] if name=="darknet" else shap_values[1,:],
                    X_test.iloc[idx], matplotlib=True, show=False)
    plt.savefig(f"results/shap/force_plot_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()