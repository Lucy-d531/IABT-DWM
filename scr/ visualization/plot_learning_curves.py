# -*- coding: utf-8 -*-
"""
论文图3：IABT-DW 与 DecisionTree, RandomForest, ExtraTrees 的学习曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv("data/processed/filtered_features_dataset.csv")
X = df.drop('label', axis=1)
y = df['label']

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "IABT-DW": XGBClassifier(random_state=42, n_estimators=500, max_depth=5,
                              learning_rate=0.2, gamma=0.015, subsample=0.842,
                              colsample_bytree=0.5, eval_metric='logloss', verbosity=0)
}

plt.figure(figsize=(8,6))
for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('tomek', TomekLinks()), ('clf', model)])
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=5, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, test_mean, 'o-', label=name)

plt.xlabel("Training examples")
plt.ylabel("F1 score")
plt.legend()
plt.grid(True)
plt.savefig("results/learning_curves.png", dpi=300)
plt.close()