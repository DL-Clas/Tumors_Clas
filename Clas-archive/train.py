"""
BTNet-TS Training Script
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net.MyDiagX import MyDiag21 as BTNet_TS


def set_seed(seed=42):
    """
    Ensures exact replication by fixing all random seeds.
    Required for strict reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to {seed}")

def xavier_init_weights(m):
    """
    Standardized Xavier initialization strategy applied to all models.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def get_data_loaders(data_dir, batch_size):
    """
    Implements the standard preprocessing and augmentation pipeline.
    Input size is strictly fixed to 224x224 pixels.
    """
    # Standard spatial transformations for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        # Standardized normalization scheme
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # Strict normalization without spatial augmentation for validation/testing
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, len(train_dataset.classes)

def main(args):
    # 1. Setup Environment
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 2. Prepare Data
    # The split metadata (8:2 ratio, deduplicated) should be organized in args.data_dir
    train_loader, val_loader, num_classes = get_data_loaders(args.data_dir, args.batch_size)
    print(f"[Info] Loaded dataset with {num_classes} classes.")

    # 3. Initialize Model (Placeholder for BTNet-TS)
    # model = BTNet_TS(num_classes=num_classes).to(device)
    
    # For demonstration, we use a generic placeholder here. Replace with BTNet_TS.
    model = nn.Sequential(nn.Flatten(), nn.Linear(224*224*3, num_classes)).to(device) 
    
    # Apply standardized Xavier Initialization
    model.apply(xavier_init_weights)

    # 4. Standardized Optimizer and Loss Function
    # Cross-entropy loss inherently supports our model's mitigation of class imbalance
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer with initial LR 1e-3
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (Decays from 1e-3 to 1e-5 over 100 epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 5. Training Loop with Early Stopping Policy
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total

        # Step the scheduler
        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Early Stopping: Monitoring on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model_{args.dataset}.pth'))
            print(f"[Info] Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Info] Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BTNet-TS Standardized Training Script")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset splits")
    parser.add_argument('--batch_size', type=int, default=16, help="Standardized batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Maximum training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--patience', type=int, default=15, help="Early stopping patience")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory to save models")

    args = parser.parse_args()
    main(args)
