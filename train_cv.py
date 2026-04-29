import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import scipy.stats as stats
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.optim import lr_scheduler

# Import the core model
from net.MyDiagX import MyDiag21 as create_model

def set_seed(seed=42):
    """Fix the global random seed to ensure the reproducibility of each fold in the cross-validation"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SubsetDataset(Dataset):
    """Custom Subset, used to dynamically assign different data augmentation strategies to the train and validation sets during cross-validation"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        # The original dataset should not include any transforms.
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)

def calculate_95_ci(data):
    """Calculate the 95% confidence interval (Mean ± 95% CI)"""
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)
    return mean, ci

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameter Configuration
    batch_size = 16
    epochs = 100
    lr = 0.001
    k_folds = 5
    dataset_name = "BTD-4"
    
    # Data augmentation
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # Load the entire dataset
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./data"))
    image_path = os.path.join(data_root, dataset_name) 
    assert os.path.exists(image_path), f"Path '{image_path}' does not exist."
    
    full_dataset = datasets.ImageFolder(root=image_path, transform=None)
    targets = full_dataset.targets
    class_list = full_dataset.class_to_idx
    num_classes = len(class_list)

    print(f"Total images loaded for {k_folds}-Fold CV: {len(full_dataset)}")

    # Initialization of K-fold cross-validation with stratification
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Used to track the best metrics for each fold
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'-'*30}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'-'*30}")

        # Dynamically applying different transforms
        train_dataset = SubsetDataset(full_dataset, train_idx, transform=data_transforms["train"])
        val_dataset = SubsetDataset(full_dataset, val_idx, transform=data_transforms["val"])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Each fold requires the model and optimizer to be reinitialized to prevent data leakage and weight inheritance.
        net = create_model(num_classes=num_classes)
        net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        best_metrics = {}

        # Early stop counter for each fold
        patience = 10
        counter = 0

        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout, leave=False)
            for images, labels in train_bar:
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            net.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout, leave=False)
                for images, labels in val_bar:
                    labels = labels.to(device)
                    outputs = net(images.to(device))
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = torch.max(outputs, dim=1)[1]
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculating the multi-category composite index (Macro average)
            current_acc = accuracy_score(all_labels, all_preds)
            current_pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            current_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            current_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f"[Fold {fold+1} Epoch {epoch+1}] Acc: {current_acc:.4f} | F1: {current_f1:.4f}")
            scheduler.step()

            # Save the best metrics for the current trade and the early exit logic
            if current_acc > best_val_acc:
                best_val_acc = current_acc
                best_metrics = {
                    'accuracy': current_acc,
                    'precision': current_pre,
                    'recall': current_rec,
                    'f1': current_f1
                }
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Record the best result for the current fold
        fold_results['accuracy'].append(best_metrics['accuracy'])
        fold_results['precision'].append(best_metrics['precision'])
        fold_results['recall'].append(best_metrics['recall'])
        fold_results['f1'].append(best_metrics['f1'])
        print(f"Fold {fold+1} Best Accuracy: {best_metrics['accuracy']:.4f}")

    # Mean ± 95% CI
    print(f"\n{'='*40}")
    print("5-FOLD CROSS VALIDATION RESULTS (Mean ± 95% CI)")
    print(f"{'='*40}")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean_val, ci_val = calculate_95_ci(fold_results[metric])
        print(f"{metric.capitalize():<10}: {mean_val*100:.2f}% ± {ci_val*100:.2f}%")

if __name__ == '__main__':
    main()
