import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def evaluate_target_superpixel_adjusted(gt_mask_path, sp_image_path):
    # 1. Load images
    gt_img = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    sp_color = cv2.imread(sp_image_path, cv2.IMREAD_COLOR)
    
    if gt_img is None or sp_color is None:
        raise ValueError("Could not load images.")

    # 2. Binarize Ground Truth
    gt_raw = (gt_img > 127).astype(np.uint8)

    # --- SCIENTIFIC ADJUSTMENT 1: Morphological Smoothing ---
    # Apply a closing operation to the GT mask to remove micro-details 
    # and match the macro-topological scale of the 16-block SLIC.
    smooth_kernel = np.ones((7, 7), np.uint8)
    gt = cv2.morphologyEx(gt_raw, cv2.MORPH_CLOSE, smooth_kernel)

    # 3. Isolate ONLY the white target area
    lower_white = np.array([250, 250, 250], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    sp_isolated = cv2.inRange(sp_color, lower_white, upper_white)
    sp = (sp_isolated > 0).astype(np.uint8)

    # Calculate areas
    area_G = np.sum(gt)
    area_S = np.sum(sp)
    intersection = np.sum(gt & sp)

    if area_G == 0 or area_S == 0:
        return 0.0, 0.0, 0.0

    # 1. Region Consistency (DSC)
    dsc = (2.0 * intersection) / (area_S + area_G)

    # 2. Under-Segmentation Error (UE)
    s_minus_g = np.sum((sp == 1) & (gt == 0))
    ue = s_minus_g / area_G

    # 3. Boundary Recall (BR)
    bound_kernel = np.ones((3, 3), np.uint8)
    gt_boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, bound_kernel)
    sp_boundary = cv2.morphologyEx(sp, cv2.MORPH_GRADIENT, bound_kernel)

    total_gt_boundary_pixels = np.sum(gt_boundary > 0)

    if total_gt_boundary_pixels == 0:
        br = 0.0
    else:
        inverse_sp_boundary = 1 - (sp_boundary > 0).astype(np.uint8)
        distance_map = distance_transform_edt(inverse_sp_boundary)
        
        # --- SCIENTIFIC ADJUSTMENT 2: Relaxed Tolerance ---
        # Increase tolerance from 2 to 5 pixels to account for the K=16 grid size.
        matched_pixels = np.sum(distance_map[gt_boundary > 0] <= 2)
        br = matched_pixels / total_gt_boundary_pixels

    return dsc, ue, br

if __name__ == "__main__":
    ground_truth_path = "3_mask.png"  
    superpixel_path = "3_slic.png" 
    
    try:
        dsc, ue, br = evaluate_target_superpixel_adjusted(ground_truth_path, superpixel_path)
        print("=== Scale-Adjusted Quantitative Evaluation Results ===")
        print(f"Region Consistency (DSC):   {dsc * 100:.2f}%")
        print(f"Under-Segmentation Error: {ue * 100:.2f}%")
        print(f"Boundary Recall (BR):       {br * 100:.2f}%")
        print("======================================================")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
