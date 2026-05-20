# -*- coding: utf-8 -*-
"""
论文 四(六) 消融实验
移除结构特征、衍生特征、金额时间特征等5种设置
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import time

df = pd.read_csv("data/processed/filtered_features_dataset.csv")
X = df.drop('label', axis=1)
y = df['label']
all_features = list(X.columns)

# 特征分组（论文定义）
structural = ['Avg_Outdegree', 'Pearson_Correlation', 'Average_Path_Length',
              'Diameter', 'Closeness_Centrality', 'Betweenness_Centrality']
derived = ['Transaction_Regularity', 'Time_Stability', 'Activity_Intensity',
           'Activity_Coefficient_of_Variation', 'Transaction_Volatility',
           'Transaction_Amount_Coefficient_of_Variation', 'Avg_Transaction_Interval_x_PageRank',
           'Degree_Concentration', 'Activity_Peak_Ratio', 'Avg_Transaction_Interval_x_Closeness_Centrality',
           'Betweenness_Centrality_x_Avg_Transaction_Interval', 'Average_Transaction_Amount']
amount_time = ['Min_Sent_Tokens', 'Lifetime', 'Active_Days',
               'Min_Transaction_Interval', 'Std_Active_Instances', 'Std_Transaction_Interval']

groups = {
    "Full": all_features,
    "-Structural": [f for f in all_features if f not in structural],
    "-Derived": [f for f in all_features if f not in derived],
    "-Amount_Time": [f for f in all_features if f not in amount_time],
    "-Structural_Derived": [f for f in all_features if f not in structural and f not in derived]
}

def evaluate_subset(feature_list):
    X_sub = X[feature_list]
    # 三级划分
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_sub, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    X_final = pd.concat([X_train, X_val])
    y_final = pd.concat([y_train, y_val])
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('tomek', TomekLinks()),
        ('xgb', XGBClassifier(random_state=42, n_estimators=500, max_depth=5,
                              learning_rate=0.2, gamma=0.015, subsample=0.842,
                              colsample_bytree=0.5, eval_metric='logloss', verbosity=0))
    ])
    start = time.time()
    pipe.fit(X_final, y_final)
    t = time.time() - start
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return f1_score(y_test, y_pred), roc_auc_score(y_test, y_proba), t

results = []
for name, feats in groups.items():
    f1, auc, t = evaluate_subset(feats)
    results.append({"实验组": name, "特征数": len(feats), "F1": f1*100, "AUC": auc*100, "时间(s)": t})

res_df = pd.DataFrame(results)
res_df.to_csv("results/ablation_results.csv", index=False)
print(res_df)