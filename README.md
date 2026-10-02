# Comparative Evaluation of Deep Learning Architectures for Credit Card Fraud Detection under Extreme Class Imbalance

**Course:** ICT-4442 Deep Learning  
**Institution:** Manipal Institute of Technology  
**Student:** Priyansh Nandan (Reg No: 230953450)  

---

## 1. Project Objective

Credit card fraud detection is characterized by extreme class imbalance (typically < 0.2% fraud) and temporal transaction dynamics. The objective of this project is to conduct a **controlled comparative evaluation** across four deep learning architecture families on the exact same benchmark dataset:

1. **MLP Baseline**
2. **1-D CNN**
3. **LSTM**
4. **Attention-based Autoencoder** (this module)

To ensure scientific validity, all models share a common preprocessing pipeline, identical chronological data splits, the exact same scaling transformations (fitted strictly on the training partition), and a unified threshold selection and evaluation protocol. The primary experimental variable is strictly the **model architecture**.

---

## 2. Dataset

The project uses the standard benchmark **ULB Credit Card Fraud Detection dataset**:
- **Total transactions:** 284,807
- **Fraudulent transactions:** 492 (fraud rate: ~0.172%)
- **Timeframe:** Two consecutive days of European cardholder transactions (September 2013)
- **Features (30 input features + 1 target):**
  - `Time`: Elapsed seconds since the first recorded transaction (0s to 172,792s)
  - `V1` to `V28`: Principal components obtained via PCA (anonymized for confidentiality)
  - `Amount`: Transaction amount in Euros
  - `Class`: Ground-truth label (`0` = legitimate, `1` = fraudulent)

### Expected Dataset Path
The raw dataset is not committed to version control due to file size. Place the uncompressed CSV at:
```
data/raw/creditcard.csv
```
Source: [Kaggle ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 3. Project Structure

```
CreditCardFraudDL/
├── data/
│   ├── raw/                  # Contains creditcard.csv (git-ignored)
│   └── processed/            # Processed splits and saved scaler
├── notebooks/                # Exploratory and demonstration notebooks
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized hyperparameter and path definitions
│   ├── preprocessing.py      # Common preprocessing, splitting, and scaling
│   ├── dataset.py            # PyTorch Dataset and DataLoader abstractions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention_autoencoder.py  # Tabular Transformer Autoencoder
│   │   └── mlp.py                    # Supervised MLP baseline
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_autoencoder.py      # Reconstruction training pipeline
│   │   └── train_mlp.py              # Supervised MLP training (pos_weight BCE)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_autoencoder.py   # Anomaly scoring and threshold selection
│   │   ├── evaluate_supervised.py    # Shared evaluation for MLP / 1D-CNN / LSTM
│   │   └── error_analysis.py         # FP/FN profiling and report figures
│   └── utils.py              # Logging, seeding, directory management
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py      # 10 unit and integration tests
│   └── test_mlp.py           # 4 MLP baseline tests
├── outputs/
│   ├── models/               # Checkpoints (best_attention_autoencoder.pt)
│   ├── metrics/              # Machine-readable JSON metrics and test scores
│   ├── logs/                 # Training logs and history
│   └── figures/              # Error-analysis figures (git-ignored)
├── run_pipeline.py           # Command-line entrypoint for the full pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore
```

---

## 4. Team Responsibilities

| Contributor | Primary Responsibilities |
|---|---|
| **Priyansh Nandan (230953450)** | **1.** Common preprocessing & chronological split pipeline<br>**2.** Attention-based Autoencoder architecture<br>**3.** Autoencoder training & threshold selection<br>**4.** Output artifacts for final model comparison |
| **Pranav Kasliwal (230911284)** | **1.** MLP baseline & supervised dataloader/evaluation harness<br>**2.** Error analysis across models<br>**3.** Comparative result visualisations |

---

## 5. Common Preprocessing Pipeline

To guarantee zero data leakage and a controlled experimental comparison across all 4 architectures:

1. **Chronological Splitting (No Shuffling):**
   - Transactions are ordered chronologically by `Time`.
   - **Split ratios:** 70% Train, 15% Validation, 15% Test.
   - Strict temporal boundaries are verified:
     $$\max(\text{Time}_{\text{train}}) \le \min(\text{Time}_{\text{val}}) \quad \text{and} \quad \max(\text{Time}_{\text{val}}) \le \min(\text{Time}_{\text{test}})$$
   - Random shuffling is prohibited because future transactions must never leak into training data.

2. **Feature Scaling Fitted Strictly on Training Set:**
   - Features (`Time`, `Amount`, `V1`–`V28`) are scaled using `RobustScaler` (or `StandardScaler`).
   - The scaler is fitted **only** on $X_{\text{train}}$:
     ```python
     scaler.fit(X_train)
     X_train = scaler.transform(X_train)
     X_val   = scaler.transform(X_val)
     X_test  = scaler.transform(X_test)
     ```
   - Fitted scaler parameters are persisted to `data/processed/robust_scaler.joblib`.

3. **Imbalance Handling Protocol:**
   - No synthetic oversampling (e.g., SMOTE) or undersampling is applied, ensuring that the empirical data distribution remains identical across models.
   - **For the Autoencoder specifically:**
     The model is trained **only on legitimate transactions** ($Class = 0$) from the training partition. Fraudulent transactions in the training partition are discarded from reconstruction training.
     Validation and test partitions retain both legitimate and fraudulent samples to evaluate anomaly detection performance.

---

## 6. Attention Autoencoder Architecture

### Tabular Self-Attention Design
Tabular features do not form a natural language sequence. To defensibly apply multi-head self-attention across features (following tabular transformer principles like FT-Transformer, NeurIPS 2021):

```text
Input Transaction x in R^{30}
         │
         ▼
[Feature Tokenizer]
Projects each scalar feature x_j into R^{d_model} with learned column embeddings
Output: T in R^{B x 30 x d_model}
         │
         ▼
[Transformer Encoder (2 Layers, 4 Heads)]
Self-attention captures non-linear pairwise interactions across features
Output: H in R^{B x 30 x d_model}
         │
         ▼
[Bottleneck Compression Layer]
Flatten -> Linear -> LayerNorm -> GELU -> Linear
Output: Latent vector z in R^{B x 8}
         │
         ▼
[Non-linear MLP Decoder]
8 -> 32 -> LayerNorm -> GELU -> 64 -> LayerNorm -> GELU -> Linear
Output: Reconstructed transaction x_hat in R^{B x 30}
```

### Architecture Specifications (Phase 1 Synopsis)
- **Input Dimension:** 30 features (`Time`, `V1`–`V28`, `Amount`)
- **Embedding Dimension ($d_{\text{model}}$):** 32
- **Transformer Encoder Layers:** 2
- **Self-Attention Heads:** 4
- **Feed-Forward Dimension:** 64
- **Bottleneck Dimension:** 8 (strictly 8-unit latent space)
- **Loss Function:** Mean Squared Error (MSE) between original input and reconstruction:
  $$\mathcal{L}_{\text{MSE}}(x, \hat{x}) = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$$

---

## 7. Training & Threshold Selection Protocol

1. **Reconstruction Training:**
   - Trained on legitimate training samples ($N \approx 199,000$).
   - Optimizer: AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$) with `ReduceLROnPlateau`.
   - Early stopping on validation reconstruction loss (patience: 6 epochs).
   - Best checkpoint saved to `outputs/models/best_attention_autoencoder.pt`.

2. **Validation-Based Threshold Selection:**
   - Per-sample reconstruction error $\text{MSE}(x_i, \hat{x}_i)$ acts as the anomaly score.
   - Threshold $\tau^*$ is selected **strictly on the validation set** by searching the precision-recall curve to maximize the $F_1$-score:
     $$\tau^* = \arg\max_{\tau} F_1(\tau; X_{\text{val}}, y_{\text{val}})$$
   - The test set is **never** used during threshold selection.

3. **Held-Out Test Evaluation:**
   - The chosen threshold $\tau^*$ is applied **once** to the held-out test partition:
     $$\hat{y}_{\text{test}} = \mathbb{I}(\text{error}(x_{\text{test}}) \ge \tau^*)$$

---

## 8. Evaluation Metrics

Because fraud accounts for only ~0.172% of transactions, accuracy is uninformative (a naive model predicting all zeros attains 99.83% accuracy). The evaluation focuses on:
- **PR-AUC (Primary Metric):** Area Under the Precision-Recall Curve (Average Precision).
- **ROC-AUC:** Area Under the Receiver Operating Characteristic.
- **Precision, Recall, F1-Score:** Evaluated at the validation-selected decision threshold.
- **Confusion Matrix:** True Negatives, False Positives, False Negatives, True Positives.

---

## 9. How to Run

### Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests
The test suite includes 10 unit and integration tests verifying dataset validation, chronological split integrity, zero leakage, scaling correctness, autoencoder forward pass, output shapes, and threshold application:
```bash
pytest tests/ -v
```

### Execute Pipeline
Once `data/raw/creditcard.csv` is placed in the project:
```bash
# Run complete end-to-end pipeline (preprocess -> train -> evaluate)
python run_pipeline.py --stage all

# Or run individual stages:
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage train --epochs 40 --batch-size 512
python run_pipeline.py --stage evaluate

# Select the model with --model (autoencoder | mlp | cnn1d | lstm | all; default: autoencoder)
python run_pipeline.py --model mlp --stage all
```

The MLP baseline (30 -> 128 -> 64 -> 32 -> 1, BatchNorm + ReLU + Dropout 0.3) trains on the full
labelled training partition with `BCEWithLogitsLoss(pos_weight = 198980 / 384 ≈ 518.18)` instead of
resampling, uses AdamW (lr 1e-3, weight decay 1e-4) with `ReduceLROnPlateau` on validation PR-AUC, and
early-stops on validation PR-AUC (patience 6). The decision threshold maximizes F1 on the validation
PR curve and is applied once to the test set.

---

## 10. Expected Output Files

Upon execution, the following artifacts are produced:
- `outputs/models/best_attention_autoencoder.pt`: PyTorch checkpoint of the best model.
- `outputs/logs/training_history.json`: Epoch-wise train and validation loss history.
- `outputs/metrics/attention_autoencoder_metrics.json`: Complete validation selection diagnostics and test set metrics.
- `outputs/metrics/model_comparison_entry.json`: Machine-readable summary for comparison across all 4 architectures:
  ```json
  {
      "model": "Attention Autoencoder",
      "pr_auc": null,
      "roc_auc": null,
      "accuracy": null,
      "precision": null,
      "recall": null,
      "f1": null,
      "threshold": null
  }
  ```
- `outputs/metrics/test_predictions.npz`: Test sample anomaly scores, ground-truth labels, and binary predictions.

MLP baseline (`--model mlp`):
- `outputs/models/best_mlp.pt`: Best checkpoint by validation PR-AUC.
- `outputs/logs/mlp_training_history.json`: Per-epoch train loss, validation PR-AUC / F1 and learning rate.
- `outputs/metrics/mlp_metrics.json`: Validation threshold selection and held-out test metrics.
- `outputs/metrics/mlp_test_predictions.npz`: Test probabilities, labels, and binary predictions.
- `outputs/metrics/error_analysis_summary.json`: TP/TN/FP/FN profiles (Amount, PCA feature shifts).
- `outputs/figures/confusion_matrix.png`, `precision_recall_curve.png`, `feature_residuals.png`.
