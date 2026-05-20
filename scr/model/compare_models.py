# -*- coding: utf-8 -*-
"""
论文 四(三) 表4
对比 IABT-DW 与 11 种主流机器学习模型
所有模型均使用默认参数 + Tomek Links + 标准化
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/processed/filtered_features_dataset.csv")
X = df.drop('label', axis=1)
y = df['label']

# 三级划分（与训练脚本一致）
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)
X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])

# 定义模型列表（论文表4）
models = {
    "BernoulliNB": BernoulliNB(),
    "Ridge": Ridge(),
    "GaussianNB": GaussianNB(),
    "SGD": SGDClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "KNeighbors": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "IABT-DW": XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
}

results = []
for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('tomek', TomekLinks()),
        ('clf', model)
    ])
    pipe.fit(X_final, y_final)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5
    })

df_res = pd.DataFrame(results).sort_values("F1", ascending=False)
df_res.to_csv("results/table4_comparison.csv", index=False)
print(df_res.to_string(index=False))