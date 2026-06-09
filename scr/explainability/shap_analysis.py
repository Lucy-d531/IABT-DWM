"""
Paper 5 Interpretability Analysis
Global feature importance, dependency graph, decision-making power graph (Figure 5-7)
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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('tomek', TomekLinks()),
    ('xgb', XGBClassifier(random_state=42, n_estimators=500, max_depth=5,
                          learning_rate=0.2, gamma=0.015, subsample=0.842,
                          colsample_bytree=0.5, eval_metric='logloss', verbosity=0))
])
model.fit(X_train, y_train)

X_sample = X_test.sample(200, random_state=42)
explainer = shap.TreeExplainer(model.named_steps['xgb'])
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("results/shap/summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/shap/bar_plot.png", dpi=300, bbox_inches='tight')
plt.close()


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


y_pred = model.predict(X_test)
dark_idx = np.where((y_test == 1) & (y_pred == 1))[0][0]
normal_idx = np.where((y_test == 0) & (y_pred == 0))[0][0]

for idx, name in [(dark_idx, "darknet"), (normal_idx, "normal")]:
    shap.force_plot(explainer.expected_value, shap_values[0,:] if name=="darknet" else shap_values[1,:],
                    X_test.iloc[idx], matplotlib=True, show=False)
    plt.savefig(f"results/shap/force_plot_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()
