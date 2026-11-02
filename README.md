# Deep Learning for Credit Card Fraud Detection under Extreme Class Imbalance

**Course:** ICT-4442 Deep Learning  
**Institution:** Manipal Institute of Technology, Manipal  

### Project Group Members
| Contributor | Registration Number | Primary Architectural Role |
|:---|:---:|:---|
| **Priyansh Nandan** | **230953450** | Common Preprocessing Pipeline & Attention-Based Autoencoder |
| **Pranav Kasliwal** | **230911284** | Supervised Feedforward MLP Baseline & Error Profiling |
| **Atharv Sharma** | **230953210** | Supervised 1-D CNN Classifier & Master Comparison Harness |
| **Yash** | **230953292** | Supervised Recurrent Neural Network (LSTM) Classifier |

---

## 1. Team Member Contribution Table

| Contributor | Primary Responsibilities & Module Deliverables |
|:---|:---|
| **Priyansh Nandan (230953450)** | **1.** Common preprocessing pipeline & chronological split verification (zero temporal leakage)<br>**2.** Tabular Feature-Token Transformer Attention Autoencoder architecture design<br>**3.** Unsupervised reconstruction training & validation-based decision thresholding<br>**4.** Master cross-model evaluation notebook and data profiling |
| **Pranav Kasliwal (230911284)** | **1.** Multilayer Perceptron (MLP) baseline architecture design & training pipeline<br>**2.** Supervised dataloader abstractions with class imbalance loss weighting (`pos_weight ≈ 518.18`)<br>**3.** Error analysis across models (False Positive / False Negative profiling)<br>**4.** Model result visualisations and metric exports |
| **Atharv Sharma (230953210)** | **1.** 1-D Convolutional Neural Network (1-D CNN) architecture & training pipeline<br>**2.** Spatial feature interaction modeling & localized convolution kernel tuning<br>**3.** Master evaluation harness & consolidated comparison table (`outputs/metrics/comparison_table.md`)<br>**4.** Comparative Precision-Recall (PR) and ROC curve generation (`outputs/metrics/`) |
| **Yash (230953292)** | **1.** Long Short-Term Memory (LSTM) recurrent neural network architecture design<br>**2.** Sequential hidden state tracking, memory cell gating & recurrent feature modeling<br>**3.** Supervised LSTM model training with positive class re-weighting & validation PR-AUC early stopping<br>**4.** Temporal fraud dependency analysis & comparative benchmarking against Feed-Forward and Spatial baselines |

---

## 2. Input: Dataset Specifications

The project evaluates architectures on the standard benchmark **ULB Credit Card Fraud Detection dataset**:
* **Total Transactions:** 284,807 transactions recorded over two days (September 2013).
* **Fraudulent Transactions:** 492 cases (**0.172% fraud rate** — extreme class imbalance).
* **Feature Schema (30 Input Features + 1 Target):**
  * `Time`: Seconds elapsed from the first recorded transaction ($0.0\text{s}$ to $172,792.0\text{s}$, $\approx 48.0\text{ hours}$).
  * `V1` to `V28`: Anonymized principal component features obtained via PCA.
  * `Amount`: Transaction amount in Euros (ranges from €$0.00$ to €$25,691.16$, median: €$22.00$).
  * `Class`: Binary ground-truth target label (`0` = legitimate transaction, `1` = fraudulent transaction).

### Dataset Placement
Place the uncompressed raw dataset CSV at:
```
data/raw/creditcard.csv
```
Source: [Kaggle ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 3. Processing: Common Preprocessing Pipeline & Architectures

To ensure scientific rigor, all models share a common preprocessing pipeline and identical chronological data splits:

```text
Raw Transaction Stream (data/raw/creditcard.csv)
   │
   ▼
[ Chronological Sorting & Splitting (No Shuffling) ]
   ├── Train Set (70%): First ~37 hours (199,364 rows, 384 frauds)
   ├── Validation Set (15%): Next ~5.5 hours (42,721 rows, 56 frauds)
   └── Held-Out Test Set (15%): Final ~5.5 hours (42,722 rows, 52 frauds)
   │
   ▼
[ Leakage-Free Scaling ]
   └── RobustScaler (Median centering, IQR scaling) fitted STRICTLY on X_train
   │
   ├──▶ Supervised Classifiers (MLP, 1-D CNN, LSTM): Train on full labelled X_train with pos_weight = 518.18
   │
   └──▶ Attention Autoencoder: All 384 training frauds withheld; trains ONLY on legitimate samples (198,980 rows)
```

### Evaluated Model Architectures:
1. **MLP Baseline**: Deep feedforward network ($30 \to 128 \to 64 \to 32 \to 1$) with BatchNorm, ReLU, and Dropout.
2. **1-D CNN**: Convolutions across feature space (Channels: $1 \to 32 \to 64 \to 128$, Kernel Size: 3, AdaptiveAvgPool1d, Dense head) for spatial regularisation.
3. **LSTM Sequence Classifier**: Multi-layer LSTM ($2$ layers, hidden dimension $64$) modeling temporal hidden state trajectories across chronological sequences.
4. **Attention-Based Autoencoder**: Feature Tokenizer projecting $30$ features into continuous tokens $\mathbb{R}^{30 \times 32} \to 2$-layer Transformer Encoder ($4$ heads) $\to 8$-dimensional latent bottleneck $\to$ Non-linear Decoder reconstructing input features.

---

## 4. Output: Evaluation Metrics & Benchmark Results

### Primary Evaluation Metric (PR-AUC)
Due to extreme class imbalance ($0.1217\%$ positive rate on the test set), **Accuracy is deceptive** (a dummy classifier predicting zero fraud achieves $99.88\%$ accuracy). **ROC-AUC is inflated** ($>0.90$) by $42,670$ True Negatives.  
**Precision-Recall AUC (PR-AUC)** is the primary benchmark, directly evaluating fraud detection efficacy relative to the random guessing baseline ($P = 0.0012$, $0.12\%$).

### Consolidated Test Set Benchmarks (Held-Out Test Set: 42,722 Transactions, 52 Frauds)
| Architecture Family | Model Name | PR-AUC (Primary) | ROC-AUC | F1-Score | Precision | Recall | Optimal Threshold ($\tau^*$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Supervised Feed-Forward | **MLP Baseline** | **0.7751** | 0.9547 | 0.7907 | 100.0% | 65.38% | `0.9993` |
| Supervised Recurrent | **LSTM Sequence Classifier** | **0.7600** | 0.9747 | 0.7865 | 94.59% | 67.31% | `0.9995` |
| Supervised 1-D CNN | **1-D CNN** | **0.7272** | 0.9686 | 0.7273 | 88.89% | 61.54% | `0.9948` |
| Attention Autoencoder | **Attention Autoencoder** | **0.1978** *(162.5× lift)* | 0.9050 | 0.3621 | 32.81% | 40.38% | `2.0078` |

### Expected Output Files:
* `outputs/models/`: Checkpoints (`best_attention_autoencoder.pt`, `best_mlp.pt`, `best_cnn1d.pt`, `best_lstm.pt`).
* `outputs/metrics/`: 
  * Metrics JSON files (`attention_autoencoder_metrics.json`, `mlp_metrics.json`, `cnn1d_metrics.json`, `lstm_metrics.json`).
  * Test prediction arrays (`test_predictions.npz`, `mlp_test_predictions.npz`, `lstm_test_predictions.npz`).
  * Comparative tables and curve plots (`comparison_table.md`, `consolidated_model_comparison.json`, `comparative_pr_curves.png`, `comparative_roc_curves.png`).
* `notebooks/fraud_detection_metrics_and_analysis.ipynb`: Master evaluation and metrics notebook.

---

## 5. Running Instructions for Unix & Windows

### 5.1 Environment Setup

#### Unix / macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Windows (Git Bash):
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

---

### 5.2 Running the Test Suite
Verify pipeline integrity, data splitting boundaries, forward passes, and thresholding:
```bash
pytest tests/ -v
```

---

### 5.3 Executing the Pipeline
Ensure `creditcard.csv` is located at `data/raw/creditcard.csv`.

```bash
# 1. Run complete end-to-end pipeline (preprocess -> train -> evaluate -> compare)
python run_pipeline.py --stage all

# 2. Run preprocessing only:
python run_pipeline.py --stage preprocess

# 3. Train or evaluate individual models:
python run_pipeline.py --model autoencoder --stage evaluate
python run_pipeline.py --model mlp --stage evaluate
python run_pipeline.py --model cnn1d --stage evaluate
python run_pipeline.py --model lstm --stage evaluate

# 4. Generate master cross-model comparison tables and curves:
python run_pipeline.py --stage compare
```

---

### 5.4 Running and Viewing the Analysis Notebook
Open the notebook in VS Code, Cursor, or JupyterLab:
```bash
jupyter lab notebooks/fraud_detection_metrics_and_analysis.ipynb
```
*(All cell outputs, tables, and figures are pre-rendered and runnable top-to-bottom).*
