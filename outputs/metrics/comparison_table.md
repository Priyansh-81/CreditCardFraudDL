# Consolidated Model Evaluation & Comparison Table

> **Project**: Deep Learning for Credit Card Fraud Detection (ICT-4442)
> **Evaluation Protocol**: Chronological Split (70% Train, 15% Val, 15% Test). Decision thresholds are chosen strictly to maximize $F_1$-score on the Validation partition and evaluated once on the unseen Test partition (42,722 transactions with 52 frauds, base rate 0.1217%).

| Architecture Family | Model Name | PR-AUC (Primary) | ROC-AUC | F1-Score | Precision | Recall | Optimal Threshold |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Supervised Feed-Forward (MLP) | **MLP Baseline** | **0.7751** | 0.9547 | 0.7907 | 1.0000 | 0.6538 | `0.9993` |
| Supervised 1-D CNN | **1-D CNN** | **0.7272** | 0.9686 | 0.7273 | 0.8889 | 0.6154 | `0.9948` |
| Attention Autoencoder | **Attention Autoencoder** | **0.1978** | 0.9050 | 0.3621 | 0.3281 | 0.4038 | `2.0078` |

### Key Findings & Architectural Observations
- **Primary Performance Driver (PR-AUC)**: In highly skewed fraud detection (0.12% positive rate), Precision-Recall AUC is the primary evaluation metric because ROC-AUC is flattered by the 42,670 true negatives.
- **Supervised Inductive Bias**: Supervised representations with positive loss reweighting (`pos_weight ≈ 518.177`) explicitly optimize fraud discriminability, while the Attention Autoencoder learns legitimate transaction manifold geometry.
- **1-D CNN Regularization**: 1D convolutions across feature axes capture localized feature interactions with weight sharing, providing strong inductive bias against overfitting to small positive fraud sample sizes.
