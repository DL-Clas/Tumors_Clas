# Efficient Brain Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python)](https://www.python.org/)

PyTorch implementation of **BTNet-TS**, a novel architecture designed to achieve an optimal balance between accuracy and efficiency in brain MRI tumor diagnosis.

## Overview

![Framework Diagram of BTNet-TS](results/arc.png)

Brain tumors present a significant threat to human life and health. Current deep learning models for brain MRI diagnosis often struggle with insufficient feature representation and limited balancing capabilities, impairing diagnostic accuracy, especially in clinical scenarios with severe class imbalance.

To address these issues, we propose the **BTNet-TS** architecture, which consists of three core components:

1. **ER-3DA (Efficient Residual blocks fused with 3-dimensional tensor attention):** Utilizes 3D tensors to capture concurrent multidimensional semantic features of brain MRI lesions without unnecessary computational bloat.
2. **MFS-GD (Multi-scale Feature Fusion Strategy based on Group DenseNet):** Fuses underlying fine-grained features with deep semantic information, inherently mitigating extreme class imbalance by preserving minority lesion details.
3. **GCF-2S (Graph Convolutional Feature extraction based on Superpixel Segmentation):** Extracts and analyzes lesion-local correlation features to provide complementary topological priors.

**Key Results (5-fold Cross-Validation):**

| Dataset | Accuracy | Notes |
|---------|----------|-------|
| BTD-4  | **98.28%** | 4-class brain tumor MRI |
| BTD-3  | **98.69%** | 3-class brain tumor MRI |
| BTD-44 | **96.03%** | 44-class fine-grained |

The model architecture is defined in `net/MyDiagX.py`, with the default model `MyDiag21` exported as `create_model` across all scripts. Multiple architecture variants are available: `MyDiag21`, `MyDiag37`, `MyDiag53`, and their `_tiny` counterparts.

---

## Environment Setup

Our standardized experimental environment ensures exact reproducibility. The model was trained and evaluated on Ubuntu 20.04 using an NVIDIA RTX 4090 GPU.

**Prerequisites:**

- Python >= 3.8
- PyTorch == 2.1.0
- CUDA == 11.3+ (for GPU acceleration; MPS supported on Apple Silicon)

**Installation:**

```bash
git clone https://github.com/DL-Clas/Tumors_Clas.git
cd Tumors_Clas

conda create -n btnet python=3.8 -y
conda activate btnet

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

**Key Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.1.0 | Deep learning framework |
| torchvision | 0.16.0 | Image transforms and datasets |
| DGL | 2.1.0 | Graph convolution (GCF-2S) |
| scikit-learn | 1.3.2 | Evaluation metrics |
| scikit-image | 0.21.0 | SLIC superpixel segmentation |
| matplotlib | 3.5.2 | Visualization |
| seaborn | 0.12.2 | Statistical visualizations |
| thop | >=0.1.1 | FLOPs/MACs profiling |
| opencv-python | 4.10.0 | Image processing |
| timm | 0.9.16 | PyTorch image models |
| einops | 0.7.0 | Tensor operations |

---

## Datasets & Data Split Protocols

To ensure complete transparency and prevent data leakage, we rigorously standardized our preprocessing and data splitting methodologies. Fold split files (`fold_1.txt` ... `fold_5.txt`) are provided under each dataset directory.

Please download the datasets from their public repositories or shared data links and place them in the `data/` folder:

### 1. BTD-4 — Brain Tumor MRI Dataset (4-class)

- **Source:** [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Details:** 7,023 aggregated human brain MRIs
- **Classes:** Glioma, Meningioma, Pituitary, Normal
- **Split:** 5-fold cross-validation, performed **strictly at the patient level and image-wise with rigorous deduplication** to prevent near-duplicate slice leakage
- **Processed:** [download](https://www.kaggle.com/datasets/cvlearning0616/btd-4-dataset)

### 2. BTD-3 — Brain Tumor Dataset (3-class)

- **Source:** [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- **Details:** 3,064 contrast-enhanced T1 images across 233 patients
- **Classes:** Glioma, Meningioma, Pituitary
- **Split:** 5-fold cross-validation, performed **strictly at the patient level and image-wise with rigorous deduplication** to prevent near-duplicate slice leakage
- **Processed:** [download](https://www.kaggle.com/datasets/cvlearning0616/btd-3-dataset)

### 3. BTD-44 — Brain Tumor MRI Images 44 Classes

- **Source:** [Kaggle](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)
- **Details:** 4,479 highly imbalanced brain tumor images across 44 fine-grained sub-categories (14 tumor types × 3 MRI modalities: T1, T1C+, T2)
- **Split:** 5-fold cross-validation, performed **strictly at the patient level and image-wise with rigorous deduplication** to prevent near-duplicate slice leakage
- **Processed:** [download](https://www.kaggle.com/datasets/cvlearning0616/btd-44-dataset)

### 4. BTD-7 — Brain Tumor Multi-Sequence MRI Dataset (7-class)

- **Source:** [Kaggle](https://www.kaggle.com/datasets/cvlearning0616/btd-7-dataset)
- **Details:** Brain tumor MRI dataset used for robustness  and out-of-distribution (OOD) detection evaluation
- **Purpose:** Serves as the OOD and robustness dataset for uncertainty quantification — models trained on BTD-4 (in-distribution) are evaluated on BTD-7 to measure OOD detection performance
- **Split:** Follows the same fold file format (`fold_1.txt` ... `fold_5.txt`) and `class_mapping.txt` structure


### Image Deduplication

To eliminate near-duplicate slices that could cause data leakage, use the provided deduplication utility:

```bash
python "data/image deduplication.py"
```

The script performs global pHash-based deduplication with configurable Hamming distance threshold. Duplicate images are moved to an archive folder (not deleted) for manual verification.

---

## Training

All models are trained under a strictly matched protocol with standardized normalization, identical augmentation (random resized crop + horizontal flip), and a fixed global random seed (Seed = 42) for exact replication. Xavier Uniform initialization is applied to all convolution and linear layers.

```bash
python train.py \
    --data_root ./data/BTD-44 \
    --weights_dir ./weights \
    --batch_size 16 \
    --epochs 15 \
    --lr 1e-3 \
    --min_lr 1e-5 \
    --k_folds 5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `./data/BTD-44` | Root directory of the dataset (must contain `fold_*.txt` and `class_mapping.txt`) |
| `--weights_dir` | `./weights` | Directory to save model weights |
| `--batch_size` | 16 | Training batch size |
| `--lr` | 1e-3 | Initial learning rate (Adam optimizer) |
| `--min_lr` | 1e-5 | Minimum learning rate for cosine annealing scheduler |
| `--k_folds` | 5 | Number of cross-validation folds |

**Training details:**

- **Optimizer:** Adam
- **Scheduler:** Cosine annealing (`T_max = epochs`, `eta_min = min_lr`)
- **Early stopping:** Patience = 10 epochs (monitors validation loss)
- **Model:** `MyDiag21` (auto-detected from `net/MyDiagX.py`)
- **Device priority:** CUDA > MPS (Apple Silicon) > CPU
- **Output:** Weights saved as `weights/MyDiag21/Fold{1..5}_Best.pth`

To train on a different dataset, change `--data_root`:

```bash
python train.py --data_root ./data/BTD-4
python train.py --data_root ./data/BTD-3
```

---

## Testing and Evaluation

### Standard Evaluation (`test.py`)

Runs 5-fold evaluation with per-fold metrics (Accuracy, Precision, Recall, F1-Score), 95% confidence intervals, confusion matrices (raw + normalized), and patient-level bootstrap statistics.

```bash
python test.py \
    --data_root ./data/BTD-44 \
    --weights_dir ./weights \
    --results_dir ./results \
    --batch_size 16 \
    --k_folds 5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `./data/BTD-44` | Root directory of the dataset |
| `--weights_dir` | `./weights` | Directory containing model weights |
| `--results_dir` | `./results` | Directory to save evaluation results |
| `--batch_size` | 16 | Evaluation batch size |
| `--k_folds` | 5 | Number of cross-validation folds |

**Outputs:**

- `results/MyDiag21/fold_level_raw_results.csv` — Per-fold metrics
- `results/MyDiag21/cm_raw.png` — Raw confusion matrix heatmap (300 DPI)
- `results/MyDiag21/cm_normalized.png` — Normalized confusion matrix heatmap (300 DPI)
- Console: Mean ± 95% CI for each metric, patient-level bootstrap accuracy & F1

### Evaluation with Confidence Scores (`test_prof.py`)

Same as `test.py` but additionally computes softmax confidence probabilities for each prediction. Results are cached for reuse, skipping model inference on subsequent runs.

```bash
python test_prof.py \
    --data_root ./data/BTD-44 \
    --weights_dir ./weights \
    --results_dir ./results
```

**Cache location:** `weights/MyDiag21/results/predictions_with_conf.txt`

**Cache format:**

```
Fold_ID    True_Label    Pred_Label    Confidence    Image_Path
```

### Evaluation with Per-Sample Recording (`test_record.py`)

Same as `test.py` but caches all fold predictions (without confidence) for downstream sensitivity analyses. Skips model inference if cache exists.

```bash
python test_record.py \
    --data_root ./data/BTD-44 \
    --weights_dir ./weights \
    --results_dir ./results
```

**Cache location:** `weights/MyDiag21/results/predictions_B44_BT.txt`

**Cache format:**

```
Fold_ID    True_Label    Pred_Label    Image_Path
```

---

## Efficiency Evaluation

Profiles hardware efficiency metrics: parameter count, MACs/FLOPs, inference latency, throughput, and peak GPU memory.

```bash
python efficiency_eval.py
```

**Requires GPU** for accurate CUDA-event-based latency measurement. Outputs:

| Metric | Description |
|--------|-------------|
| Parameters | Total learnable parameters |
| MACs | Multiply-accumulate operations (via `thop`) |
| Inference Latency | Mean ± std ms/image |
| Throughput | Images/second (batch_size=1) |

---

## Sensitivity Analyses

> **Note:** All sensitivity analysis scripts are located in the `sensitivity/` directory and should be run from the project root.

### Class-Prior Sensitivity (`sensitivity/sen_prior.py`)

Analyzes the relationship between class sample size (prior) and per-class recall. Produces a scatter plot with 95% CI regression fit.

```bash
python sensitivity/sen_prior.py
```

- **Input:** `results/predictions_B44_BT.txt`
- **Output:** `class_prior_sensitivity_BTD44.png` (300 DPI)

### Rare-Class Confusion Analysis (`sensitivity/sen_rare.py`)

Automatically identifies the 8 rarest classes (by sample count) in BTD-44 and generates a confusion sub-matrix heatmap to analyze inter-class confusion patterns among underrepresented categories.

```bash
python sensitivity/sen_rare.py
```

- **Input:** `results/predictions_B44_BT.txt`
- **Output:** `rare_class_confusion_BTD44.png` (300 DPI)

### Calibration and ECE Analysis (`sensitivity/sen_grop.py`)

Computes Expected Calibration Error (ECE) and plots reliability diagrams for the 5 majority and 5 minority classes. Evaluates whether model confidence scores are well-calibrated across class frequency groups.

```bash
python sensitivity/sen_grop.py
```

- **Input:** `weights/MyDiag21/results/predictions_with_conf.txt`
- **Output:** `reliability_diagrams_optimized.pdf` / `.png` (600 DPI)

---

## Statistical Testing

Performs rigorous statistical significance testing of BTNet-TS against baseline models using three complementary methods:



```bash
python t-test.py
```

---

## Superpixel Analysis

The `superpiexl/` module contains tools for evaluating SLIC superpixel segmentation quality against ground-truth tumor masks. This validates the superpixel preprocessing pipeline used by the GCF-2S module.

### Superpixel Segmentation Validation

Runs SLIC superpixel segmentation and computes three quantitative metrics against a ground-truth mask:

```bash
python test_superpixel.py
```

**Metrics:**

| Metric | Description |
|--------|-------------|
| **DSC (Dice Similarity Coefficient)** | Region overlap between superpixel and ground-truth: `2×\|S ∩ G\| / (\|S\| + \|G\|)` |
| **UE (Under-Segmentation Error)** | Superpixel pixels outside ground-truth region: `\|S - G\| / \|G\|` |
| **BR (Boundary Recall)** | Fraction of ground-truth boundary pixels within tolerance of superpixel boundary |

The script applies morphological smoothing and relaxed boundary tolerance to account for the 16-block SLIC grid resolution.

---

## Heatmap Visualization

The `visualization/` module provides quantitative interpretability evaluation tools for model attention heatmaps.

### Interpretability Evaluation

Evaluates the quality of Grad-CAM or attention heatmaps using pointing game and IoU metrics:

```python
cd visualization
python heatmap_eval.py
```

**Edit the file to set paths:**

```python
HEATMAP_PATH = "2v.png"   # Input: jet colormap overlay
MASK_PATH = "2l.png"      # Input: ground-truth tumor mask
```

**Metrics:**

| Metric | Description |
|--------|-------------|
| **Pointing Game** | Binary hit (1) / miss (0) — whether the peak activation falls inside the tumor region |
| **Attention-Mask IoU** | Intersection-over-Union between thresholded heatmap (top 50%) and ground-truth mask |
| **Sanity Check** | Monte Carlo simulation (1000 runs) of random heatmap to establish baseline expectation |


**Example images** (`1l.png`, `1o.png`, `1v.png`) are provided in the directory for testing.

---

## Comprehensive Analysis Module

The `comprehensive/` package provides in-depth model analysis tools for understanding internal representations and fusion dynamics.

### Representational Similarity

Computes Centered Kernel Alignment (CKA) and Singular Value CCA (SVCCA) between the representations of core BTNet-TS modules.

```python
from comprehensive import compute_cka, compute_svcca

# CKA between two representation matrices
cka_score = compute_cka(er3da_features, mfs_features)
# SVCCA with 99% variance truncation
svcca_score = compute_svcca(er3da_features, gcf_features)
```


### Fusion Analysis (`comprehensive/fusion_analysis.py`)

Analyzes multi-branch fusion dynamics of the ER-3DA modules.

```python
from comprehensive import (
    compute_gradient_norms_per_branch,
    compute_forward_activation_variance,
    compute_spatial_entropy,
    compute_branch_cka_similarity,
    branch_wise_masking_analysis,
)
```

### Superpixel Validation

Quantitative validation of SLIC superpixel segmentation quality with DSC, UE, and BR metrics.

```python
from comprehensive import (
    dice_coefficient,
    under_segmentation_error,
    boundary_recall,
    validate_superpixel_segmentation,
)
```

---

## Uncertainty Quantification

The `uncertainty/` package provides comprehensive uncertainty estimation methods for BTNet-TS, including epistemic (model) and aleatoric (data) uncertainty decomposition.

### MC Dropout

Wraps BTNet-TS with Monte Carlo Dropout for epistemic uncertainty estimation. Dropout (rate=0.2) is injected into the MFS-GD module's GFF layers. At inference, multiple stochastic forward passes are aggregated.

```python
from uncertainty import MCDropoutModel

model = MyDiag21(num_classes=4)
mc_model = MCDropoutModel(model, p_retain=0.8, n_forward=50)
result = mc_model.predict_with_uncertainty(x)
```

**Parameters:**

| Argument | Default | Description |
|----------|---------|-------------|
| `p_retain` | 0.8 | Retention probability (1 - dropout rate) |
| `n_forward` | 50 | Number of stochastic forward passes |

### Evidential Deep Learning

Replaces the final Softmax with a Softplus evidence activation, producing a Dirichlet distribution over class probabilities. Provides full uncertainty decomposition.

```python
from uncertainty import EvidentialModel, EvidentialLoss

ev_model = EvidentialModel(model)
result = ev_model.predict(x)
criterion = EvidentialLoss(annealing_step=10)
loss = criterion(evidence, targets, epoch=current_epoch)
```


### Bayesian Attention 

Replaces the deterministic 3D tensor attention parameters (P_xy, P_zx, P_zy) in ER-3DA blocks with variational Bayesian layers using the reparameterization trick.

```python
from uncertainty import BayesianAttentionModel

bayes_model = BayesianAttentionModel(model, in_channels=512)
output = bayes_model(x)
kl_loss = bayes_model.compute_kl_loss()
```

### Deep Ensemble

Trains an ensemble of BTNet-TS models with different random seeds and aggregates predictions via mean softmax probabilities.

```python
from uncertainty import EnsembleModel, train_ensemble

ensemble = EnsembleModel([model1, model2, model3, model4, model5])
result = ensemble.predict_with_uncertainty(x)
```

### OOD Detection

Evaluates out-of-distribution detection using BTD-4 (in-distribution) vs BTD-7 (OOD).

```python
from uncertainty import evaluate_ood_detection, compute_ood_metrics

metrics = evaluate_ood_detection(
    model, id_loader, ood_loader, device,
    score_type='confidence'
)
```

### Selective Prediction

Computes selective prediction metrics: Area Under the Risk-Coverage curve (AURC) and retained accuracy at specified coverage levels.

```python
from uncertainty import compute_aurc, coverage_analysis

probs = torch.softmax(model(x), dim=1).numpy()
pred_class = probs.argmax(axis=1)
confidences = probs[np.arange(len(probs)), pred_class]
errors = (pred_class != targets).astype(np.float32)

aurc = compute_aurc(confidences, errors)
cov_results = coverage_analysis(confidences, errors, coverage_levels=[0.85, 0.90, 0.95, 1.0])
```

### Calibration Metrics

Computes standard calibration and scoring metrics.

```python
from uncertainty import compute_ece, compute_nll, compute_brier_score

ece = compute_ece(probabilities, targets, n_bins=15)   
nll = compute_nll(probabilities, targets)                 
brier = compute_brier_score(probabilities, targets)      
```

---

## Project Structure

```
Tumors_Clas/
├── net/
│   └── MyDiagX.py                  # Core model (MyDiag21, ER-3DA, MFS-GD, GCF-2S)
├── data/
│   ├── BTD-3/                      # 3-class dataset (fold files, class_mapping)
│   ├── BTD-4/                      # 4-class dataset (fold files, class_mapping)
│   ├── BTD-44/                     # 44-class dataset (fold files, class_mapping)
│   ├── BTD-7/                      # 7-class dataset (fold files, class_mapping)
│   └── image deduplication.py      # pHash-based deduplication utility
├── weights/
│   ├── MyDiag21/                   # Trained fold weights (Fold{1..5}_Best.pth)
│   │   └── results/                # Cached predictions from test_prof/test_record
│   └── weights.txt                 # Download links for pretrained weights (Baidu Pan)
├── results/
│   ├── MyDiag21/                   # Evaluation results per model
│   │   ├── fold_level_raw_results.csv
│   │   ├── cm_raw.png
│   │   └── cm_normalized.png
│   ├── arc.png                     # Architecture diagram
│   ├── predictions_B44_BT.txt      # Cached predictions (from test_record)
│   └── predictions_with_conf.txt   # Cached predictions with confidence (from test_prof)
├── sensitivity/
│   ├── sen_prior.py                # Class-prior sensitivity analysis
│   ├── sen_rare.py                 # Rare-class confusion analysis
│   └── sen_grop.py                 # Calibration & ECE analysis
├── comprehensive/
│   ├── __init__.py                 # Public API exports
│   ├── fusion_analysis.py          # Branch gradient/activation/CKA/masking analysis
│   ├── representational_similarity.py  # CKA and SVCCA between modules
│   └── superpixel_validation.py    # DSC/UE/BR metrics for SLIC superpixels
├── uncertainty/
│   ├── __init__.py                 # Public API exports
│   ├── mc_dropout.py               # MC Dropout uncertainty estimation
│   ├── evidential.py               # Evidential deep learning
│   ├── bayesian_attention.py       # Variational Bayesian attention layers
│   ├── ensemble.py                 # Deep ensemble training and inference
│   ├── ood_detection.py            # OOD detection metrics (AUROC, AUPR, FPR)
│   ├── selective_prediction.py     # AURC and coverage analysis
│   └── utils.py                    # ECE, NLL, Brier score computation
├── superpiexl/
│   ├── test_superpixel.py          # SLIC superpixel evaluation (DSC/UE/BR)
│   ├── 3_mask.png                  # Example ground-truth mask
│   └── 3_slic.png                  # Example SLIC segmentation output
├── visualization/
│   ├── heatmap_eval.py             # Pointing Game, IoU, randomization check
│   ├── 1l.png                      # Example label/ground-truth
│   ├── 1o.png                      # Example overlay
│   └── 1v.png                      # Example attention heatmap
├── train.py                        # 5-fold CV training
├── test.py                         # Standard evaluation
├── test_prof.py                    # Evaluation with confidence scores
├── test_record.py                  # Evaluation with per-sample recordin
├── efficiency_eval.py              # Hardware efficiency profiling
├── t-test.py                       # Statistical significance testing
├── requirements.txt
```

---

## Contact

For any questions regarding the code, data splits, or methodology, please open an issue in this repository or contact the corresponding author.
