import os
import argparse
import math
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import the core model
from net.MyDiagX import MyDiag21 as create_model

class AddGaussianNoise(object):
    """
    Custom Gaussian noise injection module.
    The input `variance` represents the variance (0.01, 0.05, 0.10).
    The standard deviation `std` is calculated as `sqrt(variance)`.
    """
    def __init__(self, variance=0.01):
        self.std = math.sqrt(variance)
        
    def __call__(self, tensor):
        # Ensure that the tensor is within the range [0, 1]
        noise = torch.randn(tensor.size()) * self.std
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0.0, 1.0)

class RobustnessTestDataset(Dataset):
    """
    Dynamic noise data set wrapper.
    Allows us to dynamically inject specific levels of noise or blurring into the data in memory without modifying the original image on the hard drive.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.base_dataset.samples[index]
        from PIL import Image
        img = Image.open(img).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.base_dataset)

def evaluate_condition(model, loader, device, condition_name):
    """Evaluating model performance under specific interference conditions"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        bar = tqdm(loader, desc=f"Testing {condition_name:<20}", leave=False)
        for images, labels in bar:
            images = images.to(device)
            outputs = model(images)
            preds = torch.max(outputs, dim=1)[1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    return acc, prec, rec, f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load the base dataset
    assert os.path.exists(args.data_dir), f"Dataset path '{args.data_dir}' does not exist."
    base_dataset = datasets.ImageFolder(root=args.data_dir)
    num_classes = len(base_dataset.classes)

    # Initialize the model
    model = create_model(num_classes=num_classes)
    assert os.path.exists(args.weights), f"Weights file '{args.weights}' does not exist."
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    
    # Define standard normalization operations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Define the experimental matrix
    experiments = [
        {"type": "None (Baseline)", "level": "0", "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])},
        {"type": "Gaussian Noise", "level": "var = 0.01", "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            AddGaussianNoise(variance=0.01),
            normalize
        ])},
        {"type": "Gaussian Noise", "level": "var = 0.05", "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            AddGaussianNoise(variance=0.05),
            normalize
        ])},
        {"type": "Gaussian Noise", "level": "var = 0.10", "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            AddGaussianNoise(variance=0.10),
            normalize
        ])},
        {"type": "Fuzzification", "level": "k = 3x3", "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            normalize
        ])},
        {"type": "Fuzzification", "level": "k = 5x5", "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
            normalize
        ])},
        {"type": "Fuzzification", "level": "k = 7x7", "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=7),
            transforms.ToTensor(),
            normalize
        ])}
    ]

    print("="*85)
    print(f"{'Perturbation Type':<20} | {'Intensity Level':<15} | {'Acc (%)':<8} | {'P (%)':<8} | {'R (%)':<8} | {'F1 (%)':<8}")
    print("="*85)

    results_log = []

    # Run all robustness tests in a loop
    for exp in experiments:
        test_dataset = RobustnessTestDataset(base_dataset, transform=exp["transform"])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        condition_name = f"{exp['type']} ({exp['level']})"
        acc, prec, rec, f1 = evaluate_condition(model, test_loader, device, condition_name)
        
        print(f"{exp['type']:<20} | {exp['level']:<15} | {acc:>7.2f}  | {prec:>7.2f}  | {rec:>7.2f}  | {f1:>7.2f}")
        results_log.append((exp['type'], exp['level'], acc, prec, rec, f1))

    print("="*85)
    print("Evaluation Complete. You can copy the above rows directly into Table 11 of your manuscript.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Robustness of BTNet-TS against Noise and Blur.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset (./data/BTD-4/test)')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    main(args)
