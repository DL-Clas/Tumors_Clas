# Efficient Brain Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features

## Overview

![Framework Diagram of BTNet-TS](Clas-archive/results/arc.png)

Brain tumors present a significant threat to human life and health. However, current deep learning models for brain MRI diagnosis often struggle with insufficient feature representation and limited balancing capabilities, which impairs diagnostic accuracy. To address these issues, we propose the Efficient Brain MRI Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features (BTNet-TS).

First, BTNet-TS incorporates Efficient Residual blocks fused with 3-dimensional tensor attention (ER-3DA), which utilize 3D tensors to capture multidimensional semantic features of brain MRI lesions. Then, a Multi-scale Feature Fusion Strategy based on Group DenseNet (MFS-GD) is introduced to fuse underlying fine-grained features with deep semantic information. Finally, a Graph Convolutional Feature extraction method based on Superpixel Segmentation (GCF-2S) is developed to extract and analyze lesion-local correlation features. 

The experimental results demonstrate that the proposed method achieves an optimal balance between accuracy and efficiency across three brain MRI tumor datasets, reaching a maximum accuracy of 98.28% on the BTD-4 dataset. Additionally, rather than relying on explicit data-level rebalancing, BTNet-TS effectively preserves fine-grained details of minority lesions, inherently mitigating extreme class imbalance. Currently, BTNet-TS serves as a promising research architecture for public-dataset brain MRI tumor classification.

The code, detailed hyperparameters, and dataset split protocols are available at https://github.com/DL-Clas/Tumors_Clas.git.

## Construction
*(Placeholder for environment construction details)*

## Testing with BTNet-TS
Copy and paste your images into `data/` or `data2/` folder, and:
*(Placeholder for testing execution commands)*

## Training with BTNet-TS
Check the configurations of the training in `train.py`. All baseline models and the BTNet-TS architecture are trained under a fully matched protocol to ensure reproducibility. This includes standardized preprocessing, fixed normalization schemes, consistent augmentation operations, and strict random-seed handling. 

To ensure complete transparency and prevent data leakage, exact dataset split files are provided in the repository.

## Brain Tumor Dataset dataset (BTD-3)
The 8:2 train/test split for this dataset was performed strictly at the patient level to prevent near-duplicate slices.
Here the link
Citation

## Brain Tumor MRI Dataset (BTD-4)
This aggregated dataset was processed with rigorous image deduplication to ensure no near-overlap exists between training and testing sets.
Here the link
Citation

## Brain Tumor MRI Images 44 Classes (BTD-44)
This aggregated dataset was also rigorously deduplicated at the image level.
Here the link
Citation

## About
Code repository for "Efficient Brain Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features."
