import os
import cv2
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

def calculate_phash(image_path):
    """Calculate Perceptual Hash (pHash) for fast initial slice filtering."""
    try:
        img = Image.open(image_path).convert('L')
        return str(imagehash.phash(img))
    except Exception:
        return None

def calculate_ssim(img_path1, img_path2):
    """Calculate SSIM between two images for precise adjacent slice matching."""
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    score, _ = ssim(img1, img2, full=True)
    return score

def extract_leakage_clusters(image_directory, ssim_threshold=0.98):
    """
    Groups near-duplicate MRI slices (likely from the same patient) into atomic clusters
    to ensure they are allocated to the exact same cross-validation fold.
    """
    phash_dict = defaultdict(list)
    
    # 1. Fast Hash Mapping
    for root, _, files in os.walk(image_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                img_hash = calculate_phash(filepath)
                if img_hash:
                    phash_dict[img_hash].append(filepath)

    atomic_clusters = []
    
    # 2. SSIM Validation for High-Confidence Clustering
    for hash_val, paths in phash_dict.items():
        if len(paths) > 1:
            cluster = set()
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    score = calculate_ssim(paths[i], paths[j])
                    if score >= ssim_threshold:
                        cluster.add(paths[i])
                        cluster.add(paths[j])
            if cluster:
                atomic_clusters.append(list(cluster))
                
    return atomic_clusters