# -*- coding: utf-8 -*-
"""
论文 三(四) 特征工程模块
构建35个原始特征 + 27个衍生特征，两阶段筛选后输出24个特征
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

# 创建目录
os.makedirs("data/processed", exist_ok=True)

# 读取原始数据（附录表1，用户需放置 data.xlsx 于 data/raw/）
df = pd.read_excel("data/raw/data.xlsx")
print(f"原始数据: {df.shape}")

# ------------------------------------------------------------
# 1. 原始特征列表（35个，论文附录表2）
original_features = [
    'Total_Received_Tokens', 'Total_Sent_Tokens', 'Min_Received_Tokens',
    'Max_Received_Tokens', 'Min_Sent_Tokens', 'Max_Sent_Tokens',
    'Std_Received_Tokens', 'Std_Sent_Tokens', 'Std_All_Tokens',
    'In_degree', 'Out_degree', 'Total_Degree', 'Lifetime', 'Active_Days',
    'Max_Active_Instances', 'Min_Active_Instances', 'Avg_Active_Instances',
    'Std_Active_Instances', 'Max_Transaction_Interval', 'Min_Transaction_Interval',
    'Avg_Transaction_Interval', 'Std_Transaction_Interval', 'Avg_Indegree',
    'Avg_Outdegree', 'Avg_Total_Degree', 'Max_Indegree', 'Max_Outdegree',
    'Max_Total_Degree', 'Pearson_Correlation', 'Betweenness_Centrality',
    'Average_Path_Length', 'Diameter', 'Closeness_Centrality', 'PageRank', 'Density'
]

# ------------------------------------------------------------
# 2. 构建衍生特征（27个，论文附录表2）
# 交易活跃度
df['Total_Transaction_Amount'] = df['Total_Received_Tokens'] + df['Total_Sent_Tokens']
df['Transaction_Amount_Ratio'] = df['Total_Received_Tokens'] / (df['Total_Sent_Tokens']+1e-10)
df['Average_Transaction_Amount'] = df['Total_Transaction_Amount'] / (df['Total_Degree']+1e-10)
df['Transaction_Volatility'] = df['Std_All_Tokens'] / (df['Average_Transaction_Amount']+1e-10)

# 网络结构
df['In_Out_Degree_Ratio'] = df['In_degree'] / (df['Out_degree']+1e-10)
df['Degree_Concentration'] = df['Total_Degree'] / (df['Max_Total_Degree']+1e-10)

# 时间模式
df['Activity_Intensity'] = df['Active_Days'] / (df['Lifetime']+1e-10)
df['Transaction_Regularity'] = 1 / (df['Std_Transaction_Interval']+1e-10)

# 异常检测
df['Transaction_Amount_Range'] = df['Max_Received_Tokens'] - df['Min_Received_Tokens']
df['Transaction_Amount_Coefficient_of_Variation'] = df['Std_All_Tokens'] / (df['Total_Transaction_Amount']+1e-10)
df['Activity_Coefficient_of_Variation'] = df['Std_Active_Instances'] / (df['Avg_Active_Instances']+1e-10)

# 交互特征（介数中心性、交易间隔、PageRank、接近中心性）
important = ['Betweenness_Centrality', 'Avg_Transaction_Interval', 'PageRank', 'Closeness_Centrality']
for i in range(len(important)):
    for j in range(i+1, len(important)):
        df[f"{important[i]}_x_{important[j]}"] = df[important[i]] * df[important[j]]

# 比值特征
df['Receive_Extreme_Ratio'] = df['Max_Received_Tokens'] / (df['Min_Received_Tokens']+1e-10)
df['Send_Extreme_Ratio'] = df['Max_Sent_Tokens'] / (df['Min_Sent_Tokens']+1e-10)
df['Activity_Peak_Ratio'] = df['Max_Active_Instances'] / (df['Avg_Active_Instances']+1e-10)

# 偏差特征（标准化后的差值）
df['In_Degree_Deviation'] = df['In_degree'] - df['Avg_Indegree']
df['Out_Degree_Deviation'] = df['Out_degree'] - df['Avg_Outdegree']
df['Total_Degree_Deviation'] = df['Total_Degree'] - df['Avg_Total_Degree']

# 其他衍生
df['Transaction_Frequency'] = df['Total_Degree'] / (df['Lifetime']+1e-10)
df['Network_Relative_Importance'] = df['Betweenness_Centrality'] / (df['Max_Indegree']+1e-10)
df['Time_Stability'] = 1 - (df['Std_Transaction_Interval'] / (df['Avg_Transaction_Interval']+1e-10))

print(f"衍生特征数: {df.shape[1] - 1 - len(original_features)}")

# ------------------------------------------------------------
# 3. 特征筛选（两阶段）
# 阶段一：保留与标签相关系数绝对值 > 0.05 且 p < 0.05
X = df.drop('label', axis=1)
y = df['label']
keep = []
corr_info = []
for col in X.columns:
    corr, p = pearsonr(X[col], y)
    if abs(corr) > 0.05 and p < 0.05:
        keep.append(col)
        corr_info.append((col, corr, p))

print(f"阶段一保留特征数: {len(keep)}")

# 阶段二：删除特征间相关系数 > 0.85 的冗余对中与标签相关性较弱者
X_sub = X[keep]
corr_matrix = X_sub.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = set()
for col in upper.columns:
    high_corr = upper[col][upper[col] > 0.85].index
    for hc in high_corr:
        corr_col = abs(pearsonr(X_sub[col], y)[0])
        corr_hc = abs(pearsonr(X_sub[hc], y)[0])
        if corr_col < corr_hc:
            to_drop.add(col)
        else:
            to_drop.add(hc)

final_features = [f for f in keep if f not in to_drop]
print(f"最终保留特征数: {len(final_features)}   # 应为24个，与论文附录表3一致")

# ------------------------------------------------------------
# 4. Z-score 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[final_features])
X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)

# 输出最终数据集
final_df = pd.concat([X_scaled_df, y], axis=1)
final_df.to_csv("data/processed/filtered_features_dataset.csv", index=False)
print("特征工程完成，保存至 data/processed/filtered_features_dataset.csv")