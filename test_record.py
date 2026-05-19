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
    set_fontsize = 4
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    fmt = '.2f' if normalize else 'd'
    cmap = "Greens"
    
    ax = sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": set_fontsize})
    
    # Gets the right color bar object.
    cbar = ax.collections[0].colorbar
    # Sets the font size of color bar scale.
    cbar.ax.tick_params(labelsize=set_fontsize)

    # plt.title(title, fontsize=25, pad=25)
    plt.ylabel('True Label', fontsize=set_fontsize)
    plt.xlabel('Predicted Label', fontsize=set_fontsize)
    plt.xticks(rotation=45, ha='right', fontsize=set_fontsize)
    plt.yticks(rotation=0, fontsize=set_fontsize)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=900, bbox_inches='tight')
    plt.close()

def compute_95_ci_margin(data):
    a = 1.0 * np.array(data)
    n = len(a)
    mean, se = np.mean(a), stats.sem(a)
    ci_margin = se * stats.t.ppf((1 + 0.95) / 2., n-1) if n > 1 else 0
    return mean, ci_margin

def patient_level_bootstrap(patient_results, n_bootstraps=1000):
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
    
    model_name = create_model.__name__
    model_weights_dir = os.path.join(args.weights_dir, model_name)
    model_results_dir = os.path.join(args.results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    print(f"{'='*50}")
    print(f"🔬 [Evaluation] Target Model: {model_name}")
    print(f"📂 [Weights Path] {model_weights_dir}")
    print(f"📊 [Results Path] {model_results_dir}")
    print(f"{'='*50}")
    
    class_mapping = get_class_mapping(args.data_root)
    num_classes = len(class_mapping) if class_mapping else 3
    class_names = [class_mapping.get(i, f"Class {i}") for i in range(num_classes)]
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pre-allocated memory for unified calculation of indicators
    fold_data = {i: {'labels': [], 'preds': []} for i in range(1, args.k_folds + 1)}
    global_all_preds = []
    global_all_labels = []
    patient_results = {}

    # ==========================================================
    # Separation of caching mechanism and model reasoning
    # ==========================================================
    cache_file = os.path.join(model_weights_dir, "./results/predictions_B44_BT.txt")

    if os.path.exists(cache_file):
        print(f"🚀 Inference result cache file detected.: {cache_file}")
        print(f"⏭️ Skip data set loading and model reasoning, and directly read the cache for index evaluation....\n")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    fold_idx, true_l, pred_l, img_path = parts
                    fold_idx, true_l, pred_l = int(fold_idx), int(true_l), int(pred_l)
                    
                    # Load memory
                    fold_data[fold_idx]['labels'].append(true_l)
                    fold_data[fold_idx]['preds'].append(pred_l)
                    global_all_labels.append(true_l)
                    global_all_preds.append(pred_l)
                    
                    pid = extract_patient_id(img_path)
                    if pid not in patient_results:
                        patient_results[pid] = {'true_labels': [], 'pred_labels': []}
                    patient_results[pid]['true_labels'].append(true_l)
                    patient_results[pid]['pred_labels'].append(pred_l)
    else:
        print(f"⚠️ No cache file detected, will load the model for inference.")
        print(f"💾 Inference results will be saved in real-time: {cache_file}\n")
        
        # Open the file for writing to the cache.
        f_cache = open(cache_file, 'w', encoding='utf-8')
        
        for fold in range(1, args.k_folds + 1):
            weight_path = os.path.join(model_weights_dir, f'Fold{fold}_Best.pth')
            if not os.path.exists(weight_path):
                print(f"❌ Weight file not found.: {weight_path}")
                f_cache.close()
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
            
            print(f"Inferencing Fold {fold}...")

            with torch.no_grad():
                for images, labels, paths in test_loader:
                    outputs = net(images.to(device))
                    preds = torch.max(outputs, dim=1)[1]
                    
                    preds_cpu = preds.cpu().numpy()
                    labels_cpu = labels.cpu().numpy()
                    
                    for i in range(len(paths)):
                        true_l = labels_cpu[i]
                        pred_l = preds_cpu[i]
                        img_p = paths[i]
                        
                        # Stored in memory for this evaluation.
                        fold_data[fold]['labels'].append(true_l)
                        fold_data[fold]['preds'].append(pred_l)
                        global_all_labels.append(true_l)
                        global_all_preds.append(pred_l)
                        
                        pid = extract_patient_id(img_p)
                        if pid not in patient_results:
                            patient_results[pid] = {'true_labels': [], 'pred_labels': []}
                        patient_results[pid]['true_labels'].append(true_l)
                        patient_results[pid]['pred_labels'].append(pred_l)
                        
                        f_cache.write(f"{fold}\t{true_l}\t{pred_l}\t{img_p}\n")
                        
        f_cache.close()
        print(f"✅  Inference is over, and all results have been successfully cached in txt file.\n")

    # ==========================================================
    # Unified index calculation
    # ==========================================================
    print(f"{'='*50}\n   FOLD-LEVEL RAW EVALUATION\n{'='*50}")
    fold_metrics = []
    
    for fold in range(1, args.k_folds + 1):
        f_labels = fold_data[fold]['labels']
        f_preds = fold_data[fold]['preds']
        
        if not f_labels:
            continue
            
        acc = accuracy_score(f_labels, f_preds) * 100
        prec = precision_score(f_labels, f_preds, average='macro', zero_division=0) * 100
        rec = recall_score(f_labels, f_preds, average='macro', zero_division=0) * 100
        f1 = f1_score(f_labels, f_preds, average='macro', zero_division=0) * 100

        fold_metrics.append({'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
        print(f"Fold {fold} | Acc: {acc:.2f}% | Prec: {prec:.2f}% | Rec: {rec:.2f}% | F1: {f1:.2f}%")

    # ==========================================================
    # Data export and visualization module
    # ==========================================================
    csv_path = os.path.join(model_results_dir, "fold_level_raw_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        writer.writeheader()
        for i, m in enumerate(fold_metrics):
            row = {'Fold': i+1}
            row.update(m)
            writer.writerow(row)
    print(f"\n✅ The original evaluation result of folding has been exported to.: {csv_path}")

    print(f"\n{'='*50}\n   JOURNAL REPORT: 5-FOLD SUMMARY (Mean ± 95% CI)\n{'='*50}")
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        mean, ci_margin = compute_95_ci_margin(values)
        print(f"{key:>12}: {mean:.2f}% ± {ci_margin:.2f}%")
    print(f"   * The ± value represents the 95% Confidence Interval margin.")

    cm_raw = confusion_matrix(global_all_labels, global_all_preds)
    cm_norm = confusion_matrix(global_all_labels, global_all_preds, normalize='true')

    plot_and_save_confusion_matrix(cm_raw, class_names, model_results_dir, 
                                   f"Global Confusion Matrix (Raw) - {model_name}", "cm_raw.png")
    plot_and_save_confusion_matrix(cm_norm, class_names, model_results_dir, 
                                   f"Global Confusion Matrix (Normalized) - {model_name}", "cm_normalized.png", normalize=True)
    print(f"\n✅ Objective heatmap evaluation completed, and results have been saved to {model_results_dir} directory.")

    print(f"\n{'='*50}\n   PATIENT-LEVEL BOOTSTRAP (B=1000, Mean ± 95% CI)\n{'='*50}")
    p_acc, p_acc_margin, p_f1, p_f1_margin = patient_level_bootstrap(patient_results, n_bootstraps=1000)
    
    print(f"Independent patient count: {len(patient_results)}")
    print(f"Patient-Level Accuracy : {p_acc:.2f}% ± {p_acc_margin:.2f}%")
    print(f"Patient-Level F1-Score : {p_f1:.2f}% ± {p_f1_margin:.2f}%")
    print('='*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_root', type=str, default=os.path.join(BASE_DIR, "data", "BTD-44"))
    parser.add_argument('--weights_dir', type=str, default=os.path.join(BASE_DIR, "weights"))
    parser.add_argument('--results_dir', type=str, default=os.path.join(BASE_DIR, "results"))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--k_folds', type=int, default=5)
    
    main(parser.parse_args())
