# Efficient Brain Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features

## Overview
[cite_start]Brain tumors present a significant threat to human life and health[cite: 1878]. [cite_start]However, current deep learning models for brain MRI diagnosis often struggle with insufficient feature representation and limited balancing capabilities, which impairs diagnostic accuracy[cite: 1879]. [cite_start]To address these issues, we propose the Efficient Brain MRI Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features (BTNet-TS)[cite: 1556].

[cite_start]First, BTNet-TS incorporates Efficient Residual blocks fused with 3-dimensional tensor attention (ER-3DA) [cite: 1821][cite_start], which utilize 3D tensors to capture multidimensional semantic features of brain MRI lesions[cite: 1881]. [cite_start]Then, a Multi-scale Feature Fusion Strategy based on Group DenseNet (MFS-GD) is introduced to fuse underlying fine-grained features with deep semantic information[cite: 1882]. [cite_start]Finally, a Graph Convolutional Feature extraction method based on Superpixel Segmentation (GCF-2S) is developed to extract and analyze lesion-local correlation features[cite: 1883]. 

[cite_start]The experimental results demonstrate that the proposed method achieves an optimal balance between accuracy and efficiency across three brain MRI tumor datasets [cite: 1800, 1820][cite_start], reaching a maximum accuracy of 98.28% on the BTD-4 dataset[cite: 1563]. [cite_start]Additionally, rather than relying on explicit data-level rebalancing, BTNet-TS effectively preserves fine-grained details of minority lesions, inherently mitigating extreme class imbalance[cite: 1789, 1790, 1791]. [cite_start]Currently, BTNet-TS serves as a promising research architecture for public-dataset brain MRI tumor classification[cite: 1825].

[cite_start]The code, detailed hyperparameters, and dataset split protocols are available at https://github.com/DL-Clas/Tumors_Clas.git[cite: 1837, 1886].

## Construction
*(Placeholder for environment construction details)*

## Testing with BTNet-TS
Copy and paste your images into `data/` or `data2/` folder, and:
*(Placeholder for testing execution commands)*

## Training with BTNet-TS
[cite_start]Check the configurations of the training in `train.py`[cite: 1887]. [cite_start]All baseline models and the BTNet-TS architecture are trained under a fully matched protocol to ensure reproducibility[cite: 1582]. [cite_start]This includes standardized preprocessing, fixed normalization schemes, consistent augmentation operations, and strict random-seed handling[cite: 1836]. 

[cite_start]To ensure complete transparency and prevent data leakage, exact dataset split files are provided in the repository[cite: 1566, 1837].

## Brain Tumor Dataset dataset (BTD-3)
[cite_start]The 8:2 train/test split for this dataset was performed strictly at the patient level to prevent near-duplicate slices[cite: 1569].
Here the link
Citation

## Brain Tumor MRI Dataset (BTD-4)
[cite_start]This aggregated dataset was processed with rigorous image deduplication to ensure no near-overlap exists between training and testing sets[cite: 1570].
Here the link
Citation

## Brain Tumor MRI Images 44 Classes (BTD-44)
[cite_start]This aggregated dataset was also rigorously deduplicated at the image level[cite: 1570].
Here the link
Citation

## About
Code repository for "Efficient Brain Tumor Diagnosis Networks by Fusing Tensor Residual Attention and Superpixel Map Features."
