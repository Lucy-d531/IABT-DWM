本仓库是论文《照透暗网的复杂与隐秘：基于复杂网络特征与可解释AI的比特币暗网交易节点识别模型》的复现代码，运行环境为 Ubuntu 22.04 + Python 3.13.5。

**项目简介**
本仓库实现了论文中提出的 IABT-DW模型，用于识别比特币网络中的暗网交易节点。主要特点包括：
构建多层次的复杂网络特征体系（金额、时间、度、拓扑结构，共35个原始特征 + 27个衍生特征）
使用 Tomek Links欠采样处理类别不平衡问题
基于 XGBoost和贝叶斯优化训练高性能分类器
提供SHAP全流程可解释性分析（全局特征重要性、特征交互效应、样本决策路径）

在数据集（暗网节点13,861，正常节点53,082）上，模型达到：
准确率：99.87%、精确率、99.57%、召回率、99.78%、F1分数、99.68%和AUC：99.98%

**数据准备**
本实验使用的原始数据均来自公开渠道，详见论文附录表1。

**仓库结构**
IABT-DW/
├── src/
│   ├── feature_engineering/
│   │   └── build_features.py         # 特征工程（35原始+27衍生，两阶段筛选）
│   ├── model/
│   │   ├── train.py                  # IABT-DW模型训练（贝叶斯优化+ XGBoost）
│   │   └── compare_models.py         # 与主流机器学习模型对比（表4）
│   ├── explainability/
│   │   └── shap_analysis.py          # SHAP可解释性分析（图5-7）
│   └── visualization/
│       ├── plot_model_comparison.py  # 图2：热力图/箱线图/ROC/PR曲线
│       ├── plot_learning_curves.py   # 图3：学习曲线
│       └── plot_ablation.py          # 图4：消融实验条形图
├── experiments/
│   └── ablation_study.py             # 消融实验
├── results                          # 输出结果
└── README.md
