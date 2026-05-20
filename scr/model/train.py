# -*- coding: utf-8 -*-
"""
论文 三(六)(八) 及 四(四)
XGBoost模型 + Tomek Links + 贝叶斯超参数优化
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle
import os

os.makedirs("results", exist_ok=True)

# 加载筛选后数据
df = pd.read_csv("data/processed/filtered_features_dataset.csv")
X = df.drop('label', axis=1)
y = df['label']

# 三级划分（训练60%，验证20%，测试20%）
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

# Pipeline: 标准化 → Tomek Links → XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('tomek', TomekLinks()),
    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0))
])

# 贝叶斯搜索空间（论文表5）
search_spaces = {
    'xgb__n_estimators': Integer(100, 500),
    'xgb__max_depth': Integer(3, 7),
    'xgb__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'xgb__gamma': Real(0, 0.3),
    'xgb__subsample': Real(0.6, 0.9),
    'xgb__colsample_bytree': Real(0.5, 0.8)
}

# 贝叶斯优化（30次迭代，5折CV，优化F1）
opt = BayesSearchCV(
    pipeline, search_spaces, n_iter=30, scoring='f1', cv=StratifiedKFold(5),
    n_jobs=-1, random_state=42, verbose=1
)
opt.fit(X_train, y_train)

print("最佳参数:", opt.best_params_)
print("最佳CV F1:", opt.best_score_)

# 训练最终模型（训练+验证集）
X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])
best_pipeline = opt.best_estimator_
best_pipeline.fit(X_final, y_final)

# 测试集评估
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print(f"测试集: Acc={accuracy_score(y_test, y_pred):.4f}, "
      f"Prec={precision_score(y_test, y_pred):.4f}, "
      f"Rec={recall_score(y_test, y_pred):.4f}, "
      f"F1={f1_score(y_test, y_pred):.4f}, "
      f"AUC={roc_auc_score(y_test, y_proba):.4f}")

# 保存模型和参数
with open("results/best_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)