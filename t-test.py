import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================= 1. Core parameter configuration =================
PROPOSED_MODEL_FILE = 'predictions_B3_BT.txt' 
BOOTSTRAP_ITERS = 1000  
RANDOM_SEED = 42        

# ================= 2. Data processing and statistical algorithms =================
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, names=['Fold', 'TrueLabel', 'PredLabel', 'Path'])
    df['PatientID'] = df['Path'].str.extract(r'_(P\d+)_')
    return df

def get_fold_accuracies(df):
    folds = sorted(df['Fold'].unique())
    return [accuracy_score(df[df['Fold'] == f]['TrueLabel'], df[df['Fold'] == f]['PredLabel']) for f in folds]

def calculate_patient_bootstrap_ci(df, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    patients = df['PatientID'].unique()
    grouped = df.groupby('PatientID')
    p_data = {p: (grouped.get_group(p)['TrueLabel'].values, grouped.get_group(p)['PredLabel'].values) for p in patients}
    
    boot_metrics = {'Acc': [], 'P': [], 'R': [], 'F1': []}
    
    for _ in range(n_bootstraps):
        sampled_patients = np.random.choice(patients, size=len(patients), replace=True)
        u_patients, counts = np.unique(sampled_patients, return_counts=True)
        y_true, y_pred = [], []
        for p, c in zip(u_patients, counts):
            y_true.extend(p_data[p][0].tolist() * c)
            y_pred.extend(p_data[p][1].tolist() * c)
            
        boot_metrics['Acc'].append(accuracy_score(y_true, y_pred))
        boot_metrics['P'].append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        boot_metrics['R'].append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        boot_metrics['F1'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
    results = {}
    for metric_name, values in boot_metrics.items():
        mean_val = np.mean(values)
        lower, upper = np.percentile(values, 2.5), np.percentile(values, 97.5)
        margin = (upper - lower) / 2.0
        results[metric_name] = f"{mean_val*100:.2f} ± {margin*100:.2f}"
    return results

def format_p_value(p_val):
    """
    Format p-values for SCI journal submission:
    If p >= 0.001, keep three decimal places.
    If p < 0.001, use scientific notation (e.g., 1.2e-04 -> 1.20 x 10^-4)
    """
    if p_val < 0.001:
        base, exp = f"{p_val:.2e}".split('e')
        exp = int(exp) # Convert to integer to remove leading zeros
        return f"{base}e{exp}" # Console print format
    else:
        return f"{p_val:.3f}"

# ================= 3. 自动化制表主流程 =================
if __name__ == "__main__":
    all_files = glob.glob('predictions_B3_*.txt')
    if PROPOSED_MODEL_FILE not in all_files:
        raise FileNotFoundError(f" Target file not found. {PROPOSED_MODEL_FILE}, please check the path or file name!")
        
    df_proposed = load_and_preprocess(PROPOSED_MODEL_FILE)
    proposed_fold_accs = get_fold_accuracies(df_proposed)
    
    print("\n" + "="*90)
    print(f"{'Methods':<15} | {'Accuracy (%)':<15} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'Macro-F1 (%)':<15} | {'p-value':<12}")
    print("-" * 90)
    
    baseline_files = [f for f in all_files if f != PROPOSED_MODEL_FILE]
    
    for file in tqdm(baseline_files + [PROPOSED_MODEL_FILE], desc="Processing Models"):
        model_name = os.path.basename(file).replace('predictions_B3_', '').replace('.txt', '')
        df = load_and_preprocess(file)
        metrics = calculate_patient_bootstrap_ci(df, n_bootstraps=BOOTSTRAP_ITERS)
        
        if file == PROPOSED_MODEL_FILE:
            p_val_str = "-"
            model_name = "**" + model_name + " (Ours)**"
        else:
            baseline_fold_accs = get_fold_accuracies(df)
            t_stat, p_val = stats.ttest_rel(proposed_fold_accs, baseline_fold_accs)
            p_val_str = format_p_value(p_val) # Call the new formatting function
            
        print(f"{model_name:<15} | {metrics['Acc']:<15} | {metrics['P']:<15} | {metrics['R']:<15} | {metrics['F1']:<15} | {p_val_str:<12}")
        
    print("="*90)