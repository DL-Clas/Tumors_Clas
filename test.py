import os
import csv
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import scipy.stats as stats

from net.MyDiagX import MyDiag21 as create_model
# ==================================================

class KFoldDataset(Dataset):
    def __init__(self, data_root, txt_file, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        txt_path = os.path.join(data_root, txt_file)
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.rsplit(maxsplit=1) 
                if len(parts) == 2:
                    img_rel_path = parts[0]
                    label = int(parts[1])
                    self.samples.append((os.path.join(data_root, img_rel_path), label))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: 
            image = self.transform(image)
        return image, label, img_path

def get_class_mapping(data_root):
    mapping_path = os.path.join(data_root, "class_mapping.txt")
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        mapping[int(parts[0])] = parts[1].strip()
    return mapping

def plot_and_save_confusion_matrix(cm, classes, save_dir, title, filename, normalize=False):
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    fmt = '.2f' if normalize else 'd'
    cmap = "Blues"
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
    
    plt.title(title, fontsize=15, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def compute_95_ci_margin(data):
    """
    Calculate 95% confidence interval based on t distribution Margin。
    Return Mean ± Margin
    """
    a = 1.0 * np.array(data)
    n = len(a)
    mean, se = np.mean(a), stats.sem(a)
    # For n=5 and degree of freedom df=4, the error bound of 95% CI is calculated.
    ci_margin = se * stats.t.ppf((1 + 0.95) / 2., n-1) if n > 1 else 0
    return mean, ci_margin

def patient_level_bootstrap(patient_results, n_bootstraps=1000):
    """
    Perform patient-level Bootstrap resampling.
    Returns Mean and 95% cimark directly (based on normal approximation).
    """
    y_true_patient = []
    y_pred_patient = []
    
    for pid, data in patient_results.items():
        y_true_patient.append(data['true_labels'][0]) 
        preds = data['pred_labels']
        pred_vote = max(set(preds), key=preds.count)
        y_pred_patient.append(pred_vote)

    n_patients = len(y_true_patient)
    accs, f1s = [], []
    
    np.random.seed(42)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_patients, n_patients, replace=True)
        sample_y_true = [y_true_patient[i] for i in indices]
        sample_y_pred = [y_pred_patient[i] for i in indices]
        
        accs.append(accuracy_score(sample_y_true, sample_y_pred) * 100)
        f1s.append(f1_score(sample_y_true, sample_y_pred, average='macro', zero_division=0) * 100)

    # The mean value is represented by true distribution, and the error limit is 1.96 times (95% CI) of Bootstrap standard deviation.
    base_acc = accuracy_score(y_true_patient, y_pred_patient) * 100
    base_f1 = f1_score(y_true_patient, y_pred_patient, average='macro', zero_division=0) * 100
    
    acc_margin = 1.96 * np.std(accs, ddof=1)
    f1_margin = 1.96 * np.std(f1s, ddof=1)
    
    return base_acc, acc_margin, base_f1, f1_margin

def extract_patient_id(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) >= 2 and parts[1].startswith('P'):
        return parts[1]
    return filename 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Dynamic identification model and result path
    model_name = create_model.__name__
    model_weights_dir = os.path.join(args.weights_dir, model_name)
    model_results_dir = os.path.join(args.results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    print(f"{'='*50}")
    print(f"🔬 [Evaluation] Target Model: {model_name}")
    print(f"📂 [Weights Path] {model_weights_dir}")
    print(f"📊 [Results Path] {model_results_dir}")
    print(f"💻 [Device] Using device: {device}")
    print(f"{'='*50}")
    
    class_mapping = get_class_mapping(args.data_root)
    num_classes = len(class_mapping) if class_mapping else 3
    class_names = [class_mapping.get(i, f"Class {i}") for i in range(num_classes)]
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    fold_metrics = []
    global_all_preds = []
    global_all_labels = []
    patient_results = {}

    print(f"\n{'='*50}\n   FOLD-LEVEL RAW EVALUATION\n{'='*50}")

    for fold in range(1, args.k_folds + 1):
        weight_path = os.path.join(model_weights_dir, f'Fold{fold}_Best.pth')
        if not os.path.exists(weight_path):
            print(f"❌ Weight file not found.: {weight_path}")
            print(f"💡 Tip: Please make sure that you have run train.py and successfully generated the weights of the model.")
            return

        test_dataset = KFoldDataset(args.data_root, f"fold_{fold}.txt", val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Dynamic assembly architecture
        try:
            net = create_model(weights=None)
        except:
            net = create_model()
            
        if hasattr(net, 'fc'): 
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif hasattr(net, 'classifier'):
            if isinstance(net.classifier, nn.Sequential):
                in_features = net.classifier[-1].in_features
                net.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = net.classifier.in_features
                net.classifier = nn.Linear(in_features, num_classes)
                
        net.load_state_dict(torch.load(weight_path, map_location=device))
        net.to(device)
        net.eval()

        fold_preds, fold_labels = [], []

        with torch.no_grad():
            for images, labels, paths in test_loader:
                outputs = net(images.to(device))
                preds = torch.max(outputs, dim=1)[1]
                
                preds_cpu = preds.cpu().numpy()
                labels_cpu = labels.cpu().numpy()
                
                fold_preds.extend(preds_cpu)
                fold_labels.extend(labels_cpu)
                
                for i in range(len(paths)):
                    pid = extract_patient_id(paths[i])
                    if pid not in patient_results:
                        patient_results[pid] = {'true_labels': [], 'pred_labels': []}
                    patient_results[pid]['true_labels'].append(labels_cpu[i])
                    patient_results[pid]['pred_labels'].append(preds_cpu[i])

        global_all_preds.extend(fold_preds)
        global_all_labels.extend(fold_labels)

        acc = accuracy_score(fold_labels, fold_preds) * 100
        prec = precision_score(fold_labels, fold_preds, average='macro', zero_division=0) * 100
        rec = recall_score(fold_labels, fold_preds, average='macro', zero_division=0) * 100
        f1 = f1_score(fold_labels, fold_preds, average='macro', zero_division=0) * 100

        fold_metrics.append({'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
        print(f"Fold {fold} | Acc: {acc:.2f}% | Prec: {prec:.2f}% | Rec: {rec:.2f}% | F1: {f1:.2f}%")

    # ==========================================================
    # Data export (automatically stored in the corresponding model results directory)
    # ==========================================================
    csv_path = os.path.join(model_results_dir, "fold_level_raw_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        writer.writeheader()
        for i, m in enumerate(fold_metrics):
            row = {'Fold': i+1}
            row.update(m)
            writer.writerow(row)
    print(f"\n✅ The original evaluation result of folding has been exported to: {csv_path}")

    # ==========================================================
    # Generate Mean ± 95% CI (Margin)
    # ==========================================================
    print(f"\n{'='*50}\n   JOURNAL REPORT: 5-FOLD SUMMARY (Mean ± 95% CI)\n{'='*50}")
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        mean, ci_margin = compute_95_ci_margin(values)
        print(f"{key:>12}: {mean:.2f}% ± {ci_margin:.2f}%")
    print(f"   * The ± value represents the 95% Confidence Interval margin.")

    # ==========================================================
    # Generation of confusion matrix (automatically stored in the corresponding model results directory)
    # ==========================================================
    cm_raw = confusion_matrix(global_all_labels, global_all_preds)
    cm_norm = confusion_matrix(global_all_labels, global_all_preds, normalize='true')

    plot_and_save_confusion_matrix(cm_raw, class_names, model_results_dir, 
                                   f"Global Confusion Matrix (Raw) - {model_name}", "cm_raw.png")
    plot_and_save_confusion_matrix(cm_norm, class_names, model_results_dir, 
                                   f"Global Confusion Matrix (Normalized) - {model_name}", "cm_normalized.png", normalize=True)
    print(f"\n✅ The objective heat map evaluation has been completed and stored in the {model_results_dir} directory.")

    # ==========================================================
    # Patient-level Bootstrap statistical test (also output mean 95% ci)
    # ==========================================================
    print(f"\n{'='*50}\n   PATIENT-LEVEL BOOTSTRAP (B=1000, Mean ± 95% CI)\n{'='*50}")
    p_acc, p_acc_margin, p_f1, p_f1_margin = patient_level_bootstrap(patient_results, n_bootstraps=1000)
    
    print(f"Total number of independent patients: {len(patient_results)}")
    print(f"Patient-Level Accuracy : {p_acc:.2f}% ± {p_acc_margin:.2f}%")
    print(f"Patient-Level F1-Score : {p_f1:.2f}% ± {p_f1_margin:.2f}%")
    print('='*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_root', type=str, default=os.path.join(BASE_DIR, "data", "BTD-44"))
    parser.add_argument('--weights_dir', type=str, default=os.path.join(BASE_DIR, "weights"))
    parser.add_argument('--results_dir', type=str, default=os.path.join(BASE_DIR, "results"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k_folds', type=int, default=5)
    
    main(parser.parse_args())
