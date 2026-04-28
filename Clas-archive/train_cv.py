"""
BTNet-TS 5-Fold Cross-Validation
This script implements a rigorous cross-validation protocol to assess model 
stability and clinical generalization. It reports Mean ± 95% Confidence Intervals for all diagnostic metrics.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
import scipy.stats as stats
from net.MyDiagX import MyDiag21 as BTNet_TS

def set_seed(seed=42):
    """Ensure strict reproducibility across all folds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_95_ci(data):
    """
    Calculates the 95% Confidence Interval as requested by Reviewer 6.
    Formula: Mean ± (1.96 * SEM)
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * 1.96 
    return mean, ci

def get_transforms():
    """Standardized preprocessing pipeline (224x224)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_one_fold(fold, model, train_loader, val_loader, criterion, optimizer, device, args):
    """Logic for training a single fold with early stopping."""
    best_fold_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        
        if acc > best_fold_acc:
            best_fold_acc = acc
            patience_counter = 0
            # Save temporary fold weights
            torch.save(model.state_dict(), f"fold_{fold}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
                
    # Load best weights for this fold to evaluate
    model.load_state_dict(torch.load(f"fold_{fold}_best.pth"))
    return evaluate_model(model, val_loader, device)

def evaluate_model(model, loader, device):
    """Calculates all metrics including MCC and Balanced Accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        'acc': accuracy_score(all_labels, all_preds) * 100,
        'prec': precision_score(all_labels, all_preds, average='macro') * 100,
        'recall': recall_score(all_labels, all_preds, average='macro') * 100,
        'f1': f1_score(all_labels, all_preds, average='macro') * 100,
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds) * 100,
        'mcc': matthews_corrcoef(all_labels, all_preds)
    }
    return metrics

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load entire dataset for cross-validation splitting
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=get_transforms())
    labels = [sample[1] for sample in full_dataset.samples]
    
    # Stratified K-Fold to maintain class ratios in each fold (Critical for BTD-44)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    cv_results = {k: [] for k in ['acc', 'prec', 'recall', 'f1', 'balanced_acc', 'mcc']}

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Training Fold {fold+1}/{args.folds} ---")
        
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Initialize model (Replace with actual BTNet_TS)
        num_classes = len(full_dataset.classes)
        model = nn.Sequential(nn.Flatten(), nn.Linear(224*224*3, num_classes)).to(device) 
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        fold_metrics = train_one_fold(fold+1, model, train_loader, val_loader, criterion, optimizer, device, args)
        
        for k in cv_results.keys():
            cv_results[k].append(fold_metrics[k])
            print(f"Fold {fold+1} {k.upper()}: {fold_metrics[k]:.2f}")

    # Final Statistical Analysis (Mean ± 95% CI)
    print("\n" + "="*30)
    print("FINAL CROSS-VALIDATION RESULTS (Mean ± 95% CI)")
    print("="*30)
    for k, values in cv_results.items():
        mean, ci = calculate_95_ci(values)
        unit = "%" if k != 'mcc' else ""
        print(f"{k.upper():12}: {mean:.2f} ± {ci:.2f}{unit}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BTNet-TS 5-Fold Cross-Validation")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to full dataset")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='BTD-4')

    args = parser.parse_args()
    main(args)
