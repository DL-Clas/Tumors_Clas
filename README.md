# BTNet-TS: Efficient Brain Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features

[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **BTNet-TS**, a novel architecture designed to achieve an optimal balance between accuracy and efficiency in brain MRI tumor diagnosis.

## 📖 Overview

![Framework Diagram of BTNet-TS](docs/framework_diagram.png) *(Please place your diagram image in a `docs` folder or update this path)*

Brain tumors present a significant threat to human life and health. Current deep learning models for brain MRI diagnosis often struggle with insufficient feature representation and limited balancing capabilities, impairing diagnostic accuracy, especially in clinical scenarios with severe class imbalance. 

To address these issues, we propose the **BTNet-TS** architecture, which consists of three core components:
1. **ER-3DA (Efficient Residual blocks fused with 3-dimensional tensor attention):** Utilizes 3D tensors to capture concurrent multidimensional semantic features of brain MRI lesions without unnecessary computational bloat.
2. **MFS-GD (Multi-scale Feature Fusion Strategy based on Group DenseNet):** Fuses underlying fine-grained features with deep semantic information, inherently mitigating extreme class imbalance by preserving minority lesion details.
3. **GCF-2S (Graph Convolutional Feature extraction based on Superpixel Segmentation):** Extracts and analyzes lesion-local correlation features to provide complementary topological priors.

**Key Results:** BTNet-TS achieves a state-of-the-art accuracy of **98.28%** on the BTD-4 dataset. Comprehensive 5-fold cross-validation proves the model's robustness against patient and data variability, demonstrating a highly favorable accuracy-efficiency balance compared to both heavy vision transformers and compact CNNs.

---

## ⚙️ Environment Setup

Our standardized experimental environment ensures exact reproducibility. The model was trained and evaluated on Ubuntu 20.04 using an NVIDIA RTX 4090 GPU.

**Prerequisites:**
* Python >= 3.8
* PyTorch == 1.11.0
* CUDA == 11.3

**Installation:**
```bash
# Clone the repository
git clone [https://github.com/DL-Clas/Tumors_Clas.git](https://github.com/DL-Clas/Tumors_Clas.git)
cd Tumors_Clas

# Create a conda virtual environment
conda create -n btnet python=3.8 -y
conda activate btnet

# Install PyTorch and dependencies
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
