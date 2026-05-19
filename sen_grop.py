import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 15      
plt.rcParams['axes.labelsize'] = 14      
plt.rcParams['xtick.labelsize'] = 12     
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

def calculate_ece(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def get_calibration_curve(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mean_confs = []
    accs = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            mean_confs.append(confidences[in_bin].mean())
            accs.append(accuracies[in_bin].mean())
            
    return np.array(mean_confs), np.array(accs)

# ==========================================
# 1. Read data
# ==========================================
file_path = '/Users/brosion/Documents/Code/Tumors_Clas-main/updated_predictions_with_conf.txt' 

df = pd.read_csv(file_path, sep='\t', header=None, 
                 names=['fold', 'true_label', 'pred_label', 'confidence', 'path'])

df['class_name'] = df['path'].apply(lambda x: x.split('/')[-2])
class_counts = df['true_label'].value_counts()
label_to_name = df.drop_duplicates(subset=['true_label'])[['true_label', 'class_name']].set_index('true_label').to_dict()['class_name']

majority_classes = class_counts.nlargest(5).index.tolist()
minority_classes = class_counts.nsmallest(5).index.tolist()
selected_classes = majority_classes + minority_classes

# ==========================================
# 2. Drawing partial optimization
# ==========================================
# Resize the canvas to ensure that there is enough space to keep the subgraph square.
fig = plt.figure(figsize=(18, 8.5))
fig.suptitle('Reliability Diagrams by Class (Majority vs. Minority)', fontsize=22, fontweight='bold', y=1.02)

results = []

for i, cls_label in enumerate(selected_classes):
    cls_name = label_to_name[cls_label]
    # Clean up the long class name to prevent the title from being crowded.
    short_cls_name = cls_name.split(' ')[0] if len(cls_name.split(' ')) > 1 else cls_name
    if 'NORMAL' in cls_name.upper(): short_cls_name = cls_name
    
    is_majority = cls_label in majority_classes
    mask = (df['pred_label'] == cls_label)
    
    if mask.sum() == 0:
        continue
        
    cls_confs = df.loc[mask, 'confidence'].values
    cls_accs = (df.loc[mask, 'true_label'] == cls_label).astype(float).values
    
    ece = calculate_ece(cls_confs, cls_accs, n_bins=10)
    results.append({'Class': cls_name, 'Type': 'Majority' if is_majority else 'Minority', 'ECE': ece})
    
    mean_confs, accs = get_calibration_curve(cls_confs, cls_accs, n_bins=10)
    
    # Create subgraph
    ax = plt.subplot(2, 5, i + 1)
    
    # Set equal aspect ratio to make the plot square
    ax.set_aspect('equal', adjustable='box')
    
    # Draw the diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, zorder=1, label='Perfect' if i==0 else "")
    
    color = '#1f77b4' if is_majority else '#d62728' 
    marker = 's' if is_majority else 'D' 
    
    # Plot the reliability curve
    ax.plot(mean_confs, accs, marker='o', markersize=7, color=color, linewidth=2.5, 
             markeredgecolor='white', markeredgewidth=1, zorder=2, label='Model' if i==0 else "")
    
    # Set ticks and limits
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # Add grid for better readability
    ax.grid(True, linestyle=':', alpha=0.5, zorder=0)
    
    # Set the title (integrate ECE into the title to save space in the drawing)
    type_str = "Majority" if is_majority else "Minority"
    ax.set_title(f"{short_cls_name} ({type_str})\nECE = {ece:.3f}", fontsize=13, pad=10)
    
    if i % 5 == 0:
        ax.set_ylabel('Accuracy')
    
    if i >= 5:
        ax.set_xlabel('Confidence')


plt.tight_layout()
plt.savefig('reliability_diagrams_optimized.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.savefig('reliability_diagrams_optimized.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

# Print table results
results_df = pd.DataFrame(results)
print("\n=== Class-wise Expected Calibration Error (ECE) ===")
print(results_df.sort_values(by=['Type', 'ECE']))