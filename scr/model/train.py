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

df = pd.read_csv("data/processed/filtered_features_dataset.csv")
X = df.drop('label', axis=1)
y = df['label']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('tomek', TomekLinks()),
    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0))
])

search_spaces = {
    'xgb__n_estimators': Integer(100, 500),
    'xgb__max_depth': Integer(3, 7),
    'xgb__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'xgb__gamma': Real(0, 0.3),
    'xgb__subsample': Real(0.6, 0.9),
    'xgb__colsample_bytree': Real(0.5, 0.8)
}


opt = BayesSearchCV(
    pipeline, search_spaces, n_iter=30, scoring='f1', cv=StratifiedKFold(5),
    n_jobs=-1, random_state=42, verbose=1
)
opt.fit(X_train, y_train)

print("best parameters:", opt.best_params_)

X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])
best_pipeline = opt.best_estimator_
best_pipeline.fit(X_final, y_final)

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print(f"test set: Acc={accuracy_score(y_test, y_pred):.4f}, "
      f"Prec={precision_score(y_test, y_pred):.4f}, "
      f"Rec={recall_score(y_test, y_pred):.4f}, "
      f"F1={f1_score(y_test, y_pred):.4f}, "
      f"AUC={roc_auc_score(y_test, y_proba):.4f}")


with open("results/best_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)
