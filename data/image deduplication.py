import os
import shutil
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def global_deduplicate_by_folder(dataset_dir, archive_dir=None, hash_threshold=2, action='move'):
    """
    Ignoring the file naming standard, global deduplication is carried out based on the Hamming distance of image content.
    By default, it is carried out independently according to each classified subfolder to maximize the running speed.

    Parameters:
    Hash_threshold: Hamming distance threshold (0-64).
    0 means that there must be no noise at one pixel level (absolutely consistent);
    2~4 Recommended for medical images, which can tolerate extremely slight compression loss or slight noise changes;
    Greater than 10 may delete different consecutive slices by mistake.
    """
    dataset_path = Path(dataset_dir)
    
    if archive_dir is None:
        archive_path = dataset_path.parent / f"{dataset_path.name}_Duplicates_Archive"
    else:
        archive_path = Path(archive_dir)
        
    if action == 'move':
        archive_path.mkdir(parents=True, exist_ok=True)

    # Get all subfolders
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not categories:
        # If no subfolders exist, treat the root directory as the only category
        categories = [dataset_path]

    total_duplicates_found = 0

    for category_dir in categories:
        print(f"\n📂 Scanning directory: [{category_dir.name}]")
        
        # 1. Collect and calculate all the images pHash in this directory 
        image_data = []
        image_paths = [p for p in category_dir.glob('*.*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        if not image_paths:
            continue
            
        print(f"⏳ extract {len(image_paths)} images...")
        for img_path in tqdm(image_paths, desc="Hashing"):
            try:
                # Convert it into grayscale image to calculate hash, and eliminate the potential interference of color channels.
                img = Image.open(img_path).convert('L')
                h = imagehash.phash(img, hash_size=8)
                image_data.append((img_path, h))
            except Exception as e:
                print(f"⚠️  Unable to read {img_path}: {e}")

        # 2. Cross Comparison of Global N 2 Hamming Distance
        duplicates_to_remove = set()
        n = len(image_data)
        
        print(f"🧠  Executing global cross-comparison (threshold \u2264 {hash_threshold})...")
        for i in tqdm(range(n), desc="Comparing"):
            path_i, hash_i = image_data[i]
            
            # If i has already been marked as a duplicate image, skip it as a reference for matching others
            if path_i in duplicates_to_remove:
                continue
                
            for j in range(i + 1, n):
                path_j, hash_j = image_data[j]
                
                # If j has already been decided, also skipped.
                if path_j in duplicates_to_remove:
                    continue
                
                # Core: Calculate Hamming distance (how many bits are different in two 64-bit hashes)
                distance = hash_i - hash_j 
                
                if distance <= hash_threshold:
                    duplicates_to_remove.add(path_j)

        # 3. Perform physical isolation or deletion
        if duplicates_to_remove:
            print(f"🗑️ In [{category_dir.name}] found {len(duplicates_to_remove)} duplicate images, processing...")
            for dup_path in duplicates_to_remove:
                if action == 'move':
                    rel_path = dup_path.relative_to(dataset_path)
                    dest_path = archive_path / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dup_path), str(dest_path))
                elif action == 'delete':
                    os.remove(dup_path)
            total_duplicates_found += len(duplicates_to_remove)
        else:
            print("✅ No duplicate images were found.")

    print(f"\n🎉 Global deduplication task completed! Total cleaned: {total_duplicates_found} redundant slices.")

# ==========================================
# main function for testing
# ==========================================
if __name__ == '__main__':
    TARGET_DATASET = './BTD-4'
    
    global_deduplicate_by_folder(
        dataset_dir=TARGET_DATASET, 
        hash_threshold=0.65,   # Allow 2 bit hash error, specifically for dealing with same-source images that have been re-compressed or introduced slight noise
        action='move'       # Strongly recommend keeping the move mode, manually confirm the archive folder is correct before彻底 deleting
    )
