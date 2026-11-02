# Comparative Evaluation of Deep Learning Architectures for Credit Card Fraud Detection under Extreme Class Imbalance

**Course:** ICT-4442 Deep Learning  
**Institution:** Manipal Institute of Technology  
**Project Group Members:**
- **Priyansh Nandan (230953450)** — Common Preprocessing Pipeline & Attention-Based Autoencoder
- **Pranav Kasliwal (230911284)** — Supervised Feedforward MLP Baseline & Error Profiling
- **Atharv Sharma (230953210)** — Supervised 1-D CNN Classifier & Master Comparison Harness
- **Yash** — Supervised Recurrent Neural Network (LSTM) Classifier

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
│   │   ├── mlp.py                    # Supervised MLP baseline
│   │   ├── cnn1d.py                  # Supervised 1-D CNN classifier
│   │   └── lstm.py                   # Supervised Recurrent (LSTM) classifier
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_autoencoder.py      # Reconstruction training pipeline
│   │   ├── train_mlp.py              # Supervised MLP training (pos_weight BCE)
│   │   ├── train_cnn1d.py            # Supervised 1-D CNN training (pos_weight BCE)
│   │   └── train_lstm.py             # Supervised LSTM training (pos_weight BCE)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_autoencoder.py   # Anomaly scoring and threshold selection
│   │   ├── evaluate_supervised.py    # Shared evaluation for MLP / 1D-CNN / LSTM
│   │   ├── compare_models.py         # Master evaluation harness & comparative curves
│   │   └── error_analysis.py         # FP/FN profiling and report figures
│   └── utils.py              # Logging, seeding, directory management
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py      # 10 unit and integration tests
│   ├── test_mlp.py           # 4 MLP baseline tests
│   ├── test_cnn1d.py         # 4 1-D CNN tests
│   └── test_lstm.py          # Recurrent LSTM architecture tests
├── outputs/
│   ├── models/               # Checkpoints (best_attention_autoencoder.pt, best_mlp.pt, best_cnn1d.pt, best_lstm.pt)
│   ├── metrics/              # Machine-readable JSON metrics, tables, and test predictions
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
| **Atharv Sharma (230953210)** | **1.** 1-D Convolutional Neural Network (1-D CNN) architecture & training pipeline<br>**2.** Master evaluation harness & consolidated comparison table (`outputs/metrics/comparison_table.md`)<br>**3.** Comparative PR and ROC curve visualizations (`outputs/metrics/comparative_pr_curves.png`, `comparative_roc_curves.png`) |
| **Yash** | **1.** Long Short-Term Memory (LSTM) recurrent neural network architecture & temporal sequence formulation<br>**2.** Sequential hidden state tracking, memory cell gating & recurrent feature modeling<br>**3.** Supervised LSTM model training with positive class re-weighting (`pos_weight ≈ 518.18`) & validation PR-AUC early stopping<br>**4.** Temporal fraud dependency analysis & comparative benchmarking against Feed-Forward (MLP) and Spatial (1-D CNN) baselines |

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

### Windows Setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows Setup (Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
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
python run_pipeline.py --model cnn1d --stage all

# Run Master Cross-Model Comparison Harness alone
python run_pipeline.py --stage compare
# or directly:
python src/evaluation/compare_models.py
```

The supervised models (MLP and 1-D CNN) train on the full labelled training partition with
`BCEWithLogitsLoss(pos_weight = 198980 / 384 ≈ 518.18)` instead of resampling, use AdamW (lr 1e-3, weight decay 1e-4)
with `ReduceLROnPlateau` on validation PR-AUC, and early-stop on validation PR-AUC (patience 6).
The decision threshold maximizes F1 on the validation PR curve and is applied once to the test set.

---

## 10. Expected Output Files

Upon execution, the following artifacts are produced:
- `outputs/models/best_attention_autoencoder.pt`: PyTorch checkpoint of the best model.
- `outputs/logs/training_history.json`: Epoch-wise train and validation loss history.
- `outputs/metrics/attention_autoencoder_metrics.json`: Complete validation selection diagnostics and test set metrics.
- `outputs/metrics/model_comparison_entry.json`: Machine-readable summary for comparison across all 4 architectures.
- `outputs/metrics/test_predictions.npz`: Test sample anomaly scores, ground-truth labels, and binary predictions.

MLP baseline (`--model mlp`):
- `outputs/models/best_mlp.pt`: Best checkpoint by validation PR-AUC.
- `outputs/logs/mlp_training_history.json`: Per-epoch train loss, validation PR-AUC / F1 and learning rate.
- `outputs/metrics/mlp_metrics.json`: Validation threshold selection and held-out test metrics.
- `outputs/metrics/mlp_test_predictions.npz`: Test probabilities, labels, and binary predictions.
- `outputs/metrics/error_analysis_summary.json`: TP/TN/FP/FN profiles (Amount, PCA feature shifts).
- `outputs/figures/confusion_matrix.png`, `precision_recall_curve.png`, `feature_residuals.png`.

1-D CNN (`--model cnn1d`):
- `outputs/models/best_cnn1d.pt`: Best checkpoint by validation PR-AUC.
- `outputs/logs/cnn1d_training_history.json`: Per-epoch train loss, validation PR-AUC / F1 and learning rate.
- `outputs/metrics/cnn1d_metrics.json`: Validation threshold selection and held-out test metrics.
- `outputs/metrics/cnn1d_test_predictions.npz`: Test probabilities, labels, and binary predictions.

LSTM (`--model lstm`):
- `outputs/models/best_lstm.pt`: Best checkpoint by validation PR-AUC.
- `outputs/logs/lstm_training_history.json`: Per-epoch train loss, validation PR-AUC / F1 and learning rate.
- `outputs/metrics/lstm_metrics.json`: Validation threshold selection and held-out test metrics.
- `outputs/metrics/lstm_test_predictions.npz`: Test probabilities, labels, and binary predictions.

Master Comparison Harness (`--stage compare` or `compare_models.py`):
- `outputs/metrics/comparison_table.md`: Consolidated Markdown table of all evaluated models.
- `outputs/metrics/consolidated_model_comparison.json`: Structured JSON summary of all model evaluations.
- `outputs/metrics/comparative_pr_curves.png`: Overlay Precision-Recall curves with PR-AUC scores.
- `outputs/metrics/comparative_roc_curves.png`: Overlay Receiver Operating Characteristic curves with ROC-AUC scores.

---

## 11. Recurrent Architecture Specification & Yash's Presentation Guide

### 11.1 Architectural Rationale (Why LSTM?)
While the MLP baseline treats tabular transactions as independent, isolated vectors and the 1-D CNN models localized spatial feature correlations, the **Long Short-Term Memory (LSTM)** network is designed to model **sequential temporal dependencies**:
- **Fraud Velocity & Burst Dynamics**: Real-world financial fraud often involves rapid successive card authorizations, micro-testing trials, and abrupt spending surges across chronological windows.
- **Hidden State Trajectory**: An LSTM maintains a continuous hidden state $h_t$ and memory cell state $c_t$ across transactions:
  $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{(Forget Gate)}$$
  $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad \text{(Input Gate)}$$
  $$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad \text{(Candidate Cell State)}$$
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(Updated Cell Memory)}$$
  $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad \text{(Output Gate)}$$
  $$h_t = o_t \odot \tanh(c_t) \quad \text{(Hidden State Vector)}$$
- **Classification Head**: The final hidden state $h_T$ passes through a dropout-regularized dense layer to produce the fraud logit:
  $$\hat{y} = \sigma(W_{\text{out}} h_T + b_{\text{out}})$$
- **Supervised Imbalance Loss**:
  $$\mathcal{L}_{\text{BCE}}(y, \hat{p}) = -\left[ w_{\text{pos}} \cdot y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right], \quad w_{\text{pos}} = \frac{N_{\text{legit}}}{N_{\text{fraud}}} \approx 518.18$$

---

### 11.2 What Yash is Supposed to Present (Slide-by-Slide & Viva Guide)

During the group presentation and viva defense, Yash is responsible for presenting the **recurrent temporal modeling** perspective of the project:

#### Slide 1: Rationale & Motivation for Recurrent Networks in Fraud Detection
- **Key Talking Point**: Traditional credit card detection treats each swipe in a vacuum. But fraudsters behave sequentially: they start with micro-transactions ($< \$10$) to test stolen credentials, followed by rapid high-amount withdrawals.
- **The Recurrent Hypothesis**: An LSTM architecture is uniquely suited to capture temporal momentum, spending velocity, and inter-transaction state transitions that static MLPs ignore.

#### Slide 2: Model Architecture & Gating Mechanics
- **Key Talking Point**: Walk through the LSTM cell mechanics. Explain how the **Forget Gate ($f_t$)** resets memory when normal shopping patterns resume, while the **Input Gate ($i_t$)** updates the cell state when anomalous feature deviations appear.
- **Key Talking Point**: Detail the sequence formulation (e.g. sequence windowing of length $L$ or feature-step recurrence across the 30 scaled transaction dimensions).

#### Slide 3: Handling Class Imbalance in Recurrent Training
- **Key Talking Point**: Explain why standard cross-entropy fails on 99.83% legitimate data (loss gradients are dominated by normal card swipes, resulting in a trivial zero-prediction model).
- **Key Talking Point**: Detail the use of positive loss re-weighting ($w_{\text{pos}} \approx 518.18$), AdamW optimization with `ReduceLROnPlateau`, and early stopping on validation **PR-AUC** (rather than validation loss).

#### Slide 4: Comparative Trade-Offs (LSTM vs MLP vs 1-D CNN vs Autoencoder)
- **Versus MLP**: MLP is fast and has zero sequential overhead, but cannot capture temporal dependencies or sequential velocity.
- **Versus 1-D CNN**: 1-D CNN uses parallel convolutions with weight sharing (fast, strong spatial regularization), whereas LSTM computes step-by-step recurrence (higher latency, sequential dependencies).
- **Versus Attention Autoencoder**: The Autoencoder is an unsupervised anomaly detector for zero-day attacks; the LSTM is a supervised temporal discriminator requiring labeled data.

#### Slide 5: Expected Viva Defense Questions & Answers for Yash
1. **Q: "Why choose an LSTM over a standard Simple RNN?"**  
   *Answer*: Standard RNNs suffer from vanishing and exploding gradients when backpropagating through time due to repeated matrix multiplications. The LSTM’s additive cell state update ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$) acts as a gradient highway, allowing the network to retain memory over extended transaction sequences without vanishing gradients.
2. **Q: "What are the computational trade-offs of using an LSTM in a real-time banking pipeline?"**  
   *Answer*: LSTMs process inputs sequentially, meaning recurrence cannot be fully parallelized across the time dimension during inference like 1-D convolutions. In high-frequency credit card streams (thousands of transactions per second), inference latency and memory cell caching per cardholder must be balanced against detection lift.
3. **Q: "Why is PR-AUC your primary benchmark instead of Accuracy or ROC-AUC?"**  
   *Answer*: In our dataset, fraud prevalence is only 0.12% on the test set. Accuracy yields 99.88% for a trivial model predicting zero fraud. ROC-AUC is distorted by 42,670 True Negatives, making FPR artificially tiny. PR-AUC pairs Precision with Recall directly, evaluating alert purity and fraud interception without dilution.
