This repository contains the replication code for the paper "Bitcoin Dark Web Marketplace Transaction Node Identification Model Based on Complex Network Features and Explainable AI".

## Project Overview
This repository implements the IABT-DWM model proposed in the paper, which is designed to identify dark web transaction nodes in the Bitcoin network. Key features include:

- Constructing a multi-level complex network feature system (amount, time, degree, topology: 35 original features + 27 derived features)
- Using Tomek Links undersampling to address class imbalance
- Training a high-performance classifier based on XGBoost with Bayesian optimization
- Providing full-stack SHAP explainability analysis (global feature importance, feature interaction effects, individual sample decision paths)

On the dataset (13,861 dark web nodes, 53,082 normal nodes), the model achieves:

- Accuracy: 99.87%
- Precision: 99.57%
- Recall: 99.78%
- F1-score: 99.68%
- AUC: 99.98%

## Data Preparation
All raw data used in this experiment are from public sources; details are provided in Table 7 of the paper.

## Repository Structure
IABT-DWM/
├── src/
│   ├── feature_engineering/
│   │   └── build_features.py         
│   ├── model/
│   │   ├── train.py                 
│   │   └── compare_models.py         
│   ├── explainability/
│   │   └── shap_analysis.py         
│   └── visualization/
│       ├── plot_model_comparison.py  
│       ├── plot_learning_curves.py   
│       └── plot_ablation.py          
├── experiments/
│   └── ablation_study.py             
└── README.md
