import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, balanced_accuracy_score, 
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Import the core model
from net.MyDiagX import MyDiag21 as create_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Preprocessing protocols for the validation set during alignment training
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the test dataset
    assert os.path.exists(args.data_dir), f"Dataset path '{args.data_dir}' does not exist."
    test_dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Retrieve category mapping information
    class_list = test_dataset.classes
    num_classes = len(class_list)
    print(f"Loaded {len(test_dataset)} test images across {num_classes} classes.")

    # 3. Initialize the model and load the weights
    model = create_model(num_classes=num_classes)
    assert os.path.exists(args.weights), f"Weights file '{args.weights}' does not exist."
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # 4. Execute inference
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating", leave=True)
        for images, labels in test_bar:
            outputs = model(images.to(device))
            preds = torch.max(outputs, dim=1)[1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculation of Key Performance Indicators
    acc = accuracy_score(all_labels, all_preds)
    # Use the macro average to ensure that the weights of each category are consistent, thereby accurately reflecting the performance of minority categories.
    mac_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Introducing MCC and Balanced Accuracy: Essential Metrics for Medical Imaging
    mcc = matthews_corrcoef(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)

    print("\n" + "="*50)
    print("COMPREHENSIVE TEST METRICS (For Manuscript Tables)")
    print("="*50)
    print(f"Accuracy (Overall)  : {acc*100:.2f}%")
    print(f"Balanced Accuracy   : {b_acc*100:.2f}%")
    print(f"Macro Precision     : {mac_prec*100:.2f}%")
    print(f"Macro Recall        : {mac_rec*100:.2f}%")
    print(f"Macro F1-Score      : {mac_f1*100:.2f}%")
    print(f"MCC                 : {mcc:.4f}")
    print("="*50)

    # Print a detailed report containing all Recall/F1 categories
    print("\nPer-Class Performance Report:")
    report = classification_report(all_labels, all_preds, target_names=class_list, digits=4, zero_division=0)
    print(report)

    # 6. Generate and save a high-resolution confusion matrix (automatically adapts to the number of categories)
    os.makedirs('results', exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Dynamically adjust image size and whether to display values based on the number of categories (for multi-category scenarios such as BTD-44)
    fig_size = (10, 8) if num_classes <= 10 else (20, 18)
    annot = True if num_classes <= 10 else False
    
    plt.figure(figsize=fig_size)
    sns.heatmap(cm, annot=annot, cmap='Blues', fmt='g', xticklabels=class_list, yticklabels=class_list)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Export high-resolution images (300 DPI)
    plt.savefig('results/confusion_matrix.png', dpi=300)
    print("\n[Success] High-resolution confusion matrix saved to 'results/confusion_matrix.png'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate BTNet-TS model comprehensively.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset directory (./data/BTD-44/test)')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    args = parser.parse_args()
    main(args)
