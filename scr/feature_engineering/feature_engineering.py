import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr


os.makedirs("data/processed", exist_ok=True)

df = pd.read_excel("data/raw/data.xlsx")
print(f"raw data: {df.shape}")

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

# Trading activity
df['Total_Transaction_Amount'] = df['Total_Received_Tokens'] + df['Total_Sent_Tokens']
df['Transaction_Amount_Ratio'] = df['Total_Received_Tokens'] / (df['Total_Sent_Tokens']+1e-10)
df['Average_Transaction_Amount'] = df['Total_Transaction_Amount'] / (df['Total_Degree']+1e-10)
df['Transaction_Volatility'] = df['Std_All_Tokens'] / (df['Average_Transaction_Amount']+1e-10)

# Network structure
df['In_Out_Degree_Ratio'] = df['In_degree'] / (df['Out_degree']+1e-10)
df['Degree_Concentration'] = df['Total_Degree'] / (df['Max_Total_Degree']+1e-10)

# time 
df['Activity_Intensity'] = df['Active_Days'] / (df['Lifetime']+1e-10)
df['Transaction_Regularity'] = 1 / (df['Std_Transaction_Interval']+1e-10)

# Anomaly detection
df['Transaction_Amount_Range'] = df['Max_Received_Tokens'] - df['Min_Received_Tokens']
df['Transaction_Amount_Coefficient_of_Variation'] = df['Std_All_Tokens'] / (df['Total_Transaction_Amount']+1e-10)
df['Activity_Coefficient_of_Variation'] = df['Std_Active_Instances'] / (df['Avg_Active_Instances']+1e-10)

# Interactive features 
important = ['Betweenness_Centrality', 'Avg_Transaction_Interval', 'PageRank', 'Closeness_Centrality']
for i in range(len(important)):
    for j in range(i+1, len(important)):
        df[f"{important[i]}_x_{important[j]}"] = df[important[i]] * df[important[j]]


df['Receive_Extreme_Ratio'] = df['Max_Received_Tokens'] / (df['Min_Received_Tokens']+1e-10)
df['Send_Extreme_Ratio'] = df['Max_Sent_Tokens'] / (df['Min_Sent_Tokens']+1e-10)
df['Activity_Peak_Ratio'] = df['Max_Active_Instances'] / (df['Avg_Active_Instances']+1e-10)


df['In_Degree_Deviation'] = df['In_degree'] - df['Avg_Indegree']
df['Out_Degree_Deviation'] = df['Out_degree'] - df['Avg_Outdegree']
df['Total_Degree_Deviation'] = df['Total_Degree'] - df['Avg_Total_Degree']

df['Transaction_Frequency'] = df['Total_Degree'] / (df['Lifetime']+1e-10)
df['Network_Relative_Importance'] = df['Betweenness_Centrality'] / (df['Max_Indegree']+1e-10)
df['Time_Stability'] = 1 - (df['Std_Transaction_Interval'] / (df['Avg_Transaction_Interval']+1e-10))

print(f"Number of derived features: {df.shape[1] - 1 - len(original_features)}")

# ------------------------------------------------------------
X = df.drop('label', axis=1)
y = df['label']
keep = []
corr_info = []
for col in X.columns:
    corr, p = pearsonr(X[col], y)
    if abs(corr) > 0.05 and p < 0.05:
        keep.append(col)
        corr_info.append((col, corr, p))

print(f"Number of retained features in stage one: {len(keep)}")

#Stage 2
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
print(f"The final number of retained features: {len(final_features)}   

# ------------------------------------------------------------
# Z-scor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[final_features])
X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)


final_df = pd.concat([X_scaled_df, y], axis=1)
final_df.to_csv("data/processed/filtered_features_dataset.csv", index=False)
print("Feature engineering completed，Save to data/processed/filtered_features_dataset.csv")
