import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def process_and_plot_rare_confusion(file_path, num_rare_classes=8):
    """
    Read the prediction file, automatically identify the rare categories with the least sample size, and draw the confusion matrix heat map between them.

    Parameters:
    File_path (str): Predict the file path.
    Num_rare_classes (int): the number of rare classes to be extracted (8 classes with the least sample size are extracted by default).
    """
    data = []
    
    print(f"正在读取文件: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Check if the first line is a header
        start_idx = 1 if any(keyword in lines[0].lower() for keyword in ['fold', 'label', 'path']) else 0
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
                
            # Compatible with tab or comma separation
            parts = line.replace('\t', ',').split(',')
            
            if len(parts) >= 4:
                try:
                    true_label = int(parts[1].strip())
                    pred_label = int(parts[2].strip())
                    f_path = parts[3].strip()
                    
                    # Extract the class name from the file path
                    normalized_path = f_path.replace('\\', '/')
                    class_name = os.path.basename(os.path.dirname(normalized_path))
                    
                    if not class_name:
                        class_name = f"Class_{true_label}"
                        
                    data.append({
                        'True_Label': true_label,
                        'Pred_Label': pred_label,
                        'True_Name': class_name
                    })
                except ValueError:
                    continue

    if not data:
        print("Failed to parse data successfully, please check the column delimiter of txt file.")
        return

    df = pd.DataFrame(data)
    
    # Create a mapping from label to class name
    label_to_name = df.drop_duplicates('True_Label').set_index('True_Label')['True_Name'].to_dict()
    # Add a new column for predicted class names based on the predicted labels
    df['Pred_Name'] = df['Pred_Label'].map(label_to_name)
    
    # ---------------- Count the number of samples and screen rare categories ----------------
    # Count the number of samples for each true category
    class_counts = df['True_Name'].value_counts()
    
    # Extract the N classes with the least sample size as "Rare Classes"
    rare_classes = class_counts.tail(num_rare_classes).index.tolist()
    print(f"Automatically identified {num_rare_classes} rare classes:\n{rare_classes}")
    
    # Get all labels and their corresponding class names
    all_labels = sorted(list(label_to_name.keys()))
    all_names = [label_to_name[l] for l in all_labels]
    
    # Calculate the complete confusion matrix
    cm_full = confusion_matrix(df['True_Label'], df['Pred_Label'], labels=all_labels)
    cm_df = pd.DataFrame(cm_full, index=all_names, columns=all_names)
    
    # Extract the confusion sub-matrix between rare classes (True vs Pred)
    cm_rare = cm_df.loc[rare_classes, rare_classes]

    # ---------------- Plot Academic-Level Heatmap ----------------
    # Set plotting style
    sns.set_theme(style="white", context="paper", font_scale=1.1)
    plt.figure(figsize=(8, 7))

    # Plot the heatmap with annotations
    ax = sns.heatmap(
        cm_rare, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar_kws={'label': 'Prediction Count'},
        linewidths=.5,
        linecolor='gray'
    )

    # Set titles and labels
    plt.title('Confusion Matrix: Rare-Class Boundary Analysis', fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
    plt.ylabel('True Category', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()

    # Save as high-quality PNG
    output_filename = 'rare_class_confusion_BTD44.png'
    plt.savefig(output_filename, dpi=300, format='png')
    print(f"Chart successfully saved as: {output_filename}")
    plt.show()

if __name__ == "__main__":
   
    process_and_plot_rare_confusion('/Users/brosion/Documents/Code/Tumors_Clas-main/weights/efficientnet_b0/predictions_B44_BT.txt', num_rare_classes=8)