import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_predictions_and_plot(file_path, y_min=85):
    """
    Read the forecast file, count the sample number and recall rate of each category, and draw a scatter regression diagram.
    Cancel the calculation of p-value, and move the text box to the lower right corner.

    Parameters:
    File_path (str): Predict the path of the file.
    Y _ min (float/int): the minimum value of the y axis, which is set to 85 by default.
    """
    data = []
    
    print(f"Processing file: {file_path} ...")
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
                    
                    normalized_path = f_path.replace('\\', '/')
                    class_name = os.path.basename(os.path.dirname(normalized_path))
                    
                    if not class_name:
                        class_name = f"Class_{true_label}"
                        
                    data.append({
                        'True_Label': true_label,
                        'Pred_Label': pred_label,
                        'Class_Name': class_name
                    })
                except ValueError:
                    continue

    if not data:
        print("Failed to parse data successfully, please check the column delimiter of txt file.")
        return

    # Convert to DataFrame and calculate metrics
    df = pd.DataFrame(data)
    class_metrics = []
    
    for class_name, group in df.groupby('Class_Name'):
        total_samples = len(group)
        correct_predictions = len(group[group['True_Label'] == group['Pred_Label']])
        recall = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        
        class_metrics.append({
            'Class_Name': class_name,
            'Sample_Size': total_samples,
            'Recall': recall
        })

    metrics_df = pd.DataFrame(class_metrics)
    num_classes = len(metrics_df)
    print(f"Successfully counted data for {num_classes} classes.")

    # ---------------- Plot Academic-Level Heatmap ----------------
    sns.set_theme(style="ticks", context="paper", font_scale=1.2, font='Times New Roman')
    plt.figure(figsize=(8, 5))

    # Plot the scatter plot and regression fit line (still keep the fit line to intuitively show the trend)
    sns.regplot(
        x='Sample_Size', 
        y='Recall', 
        data=metrics_df,
        ci=95, 
        scatter_kws={'alpha': 0.7, 's': 55, 'color': '#2c7bb6', 'edgecolor': 'white'}, 
        line_kws={'color': '#d7191c', 'linestyle': '--', 'linewidth': 2}
    )

    plt.ylim(bottom=y_min, top=101) 
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  

    plt.title('Class-Prior Sensitivity Analysis (BTD-44 Dataset)', fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('Class Sample Size (Prior)', fontsize=20)
    plt.ylabel('Per-Class Sensitivity / Recall (%)', fontsize=20)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Save and display
    output_filename = 'class_prior_sensitivity_BTD44.png'
    plt.savefig(output_filename, dpi=300, format='png')
    print(f": The chart was successfully saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    process_predictions_and_plot('results/predictions_B44_BT.txt', y_min=60)
