import os
import random
import argparse
from shutil import copy
from tqdm import tqdm

def mk_file(file_path: str):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def main(args):
    # 1. Strictly fix the random seed
    random.seed(args.seed)

    # 2. Path Configuration
    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, args.dataset)
    assert os.path.exists(dataset_dir), f"Error: Dataset directory '{dataset_dir}' does not exist."

    # To protect the original data, output it to a new separate folder 
    out_dir = os.path.join(cwd, f"{args.dataset}_Split")
    train_root = os.path.join(out_dir, "train")
    val_root = os.path.join(out_dir, "val")

    # 3. Dynamically retrieve all category folders
    classes = [c for c in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, c))]
    if not classes:
        raise ValueError(f"No class directories found in {dataset_dir}. Please check your dataset format.")

    print(f"Found {len(classes)} classes in '{args.dataset}'.")
    print(f"Target Split Ratio -> Train: {1-args.val_rate:.1f} | Val: {args.val_rate:.1f}")

    # Initialize the target category folder
    for c in classes:
        mk_file(os.path.join(train_root, c))
        mk_file(os.path.join(val_root, c))

    total_train = 0
    total_val = 0

    # 4. Core Segmentation Logic
    for cla in classes:
        cla_path = os.path.join(dataset_dir, cla)
        # Read only image files and filter out any hidden files, such as .DS_Store
        images = [img for img in os.listdir(cla_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        num_images = len(images)

        val_images = set()

        if args.split_mode == 'patient':
            # =========================================================================
            # [Data Leakage Prevention Mechanism] Implementing a strict patient-wise split for BTD-3
            # =========================================================================
            patient_dict = {}
            for img in images:
                # The patient ID is included in a specific part of the filename. 
                # Modify the delimiter here according to the naming conventions of your actual dataset (by default, the first part is separated by an underscore).
                patient_id = img.split('_')[0] if '_' in img else img 
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = []
                patient_dict[patient_id].append(img)
            
            patients = list(patient_dict.keys())
            num_val_patients = max(1, int(len(patients) * args.val_rate))
            
            # Random sampling by patient
            val_patients = set(random.sample(patients, k=num_val_patients))
            
            for p in val_patients:
                for img_name in patient_dict[p]:
                    val_images.add(img_name)
                    
        else:
            # =========================================================================
            # Image-wise split: Suitable for BTD-4 / BTD-44 after deduplication and aggregation
            # =========================================================================
            num_val_images = max(1, int(num_images * args.val_rate))
            val_images = set(random.sample(images, k=num_val_images))

        # 5. Perform a physical copy operation
        for img in tqdm(images, desc=f"Processing '{cla}'", leave=False):
            src_path = os.path.join(cla_path, img)
            if img in val_images:
                dst_path = os.path.join(val_root, cla)
                total_val += 1
            else:
                dst_path = os.path.join(train_root, cla)
                total_train += 1
            
            copy(src_path, dst_path)

    # 6. Generate a partitioning report
    print("\n" + "="*50)
    print("🚀 DATA SPLIT COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Dataset Processed  : {args.dataset}")
    print(f"Split Mode Applied : {'Patient-wise (Zero Leakage)' if args.split_mode == 'patient' else 'Image-wise (Deduplicated)'}")
    print(f"Random Seed        : {args.seed}")
    print(f"Training Samples   : {total_train} images")
    print(f"Validation Samples : {total_val} images")
    print(f"Data Saved To      : {out_dir}/")
    print("="*50)
    print("Notice: Please point your train.py '--data_dir' to the new output path above.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Strict Data Splitting Protocol for BTNet-TS")
    parser.add_argument('--dataset', type=str, required=True, help="Folder name of the raw dataset (BTD-4)")
    parser.add_argument('--val_rate', type=float, default=0.2, help="Validation set ratio (Default: 0.2 for 8:2 split)")
    parser.add_argument('--split_mode', type=str, choices=['image', 'patient'], default='image', 
                        help="Use 'patient' for BTD-3 to avoid slice leakage, 'image' for BTD-4/44.")
    parser.add_argument('--seed', type=int, default=42, help="Global random seed for strict reproducibility")
    args = parser.parse_args()
    main(args)
