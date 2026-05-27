import numpy as np
import cv2
import matplotlib.pyplot as plt

def extract_activation_from_jet_overlay(overlay_path):
    """
    Extracts the underlying 2D activation array from a saved RGB heatmap overlay.
    In 'Jet' colormaps, peak heat is distinctly red, and cold/background is blue.
    We proxy the activation intensity using the difference between Red and Blue channels.
    """
    overlay_bgr = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
    if overlay_bgr is None:
        raise FileNotFoundError(f"Could not load image at {overlay_path}")
    
    # Extract B, G, R channels
    B = overlay_bgr[:, :, 0].astype(np.float32)
    R = overlay_bgr[:, :, 2].astype(np.float32)
    
    # Proxy activation: Red strongly indicates focus, Blue strongly indicates absence.
    activation = R - B 
    
    # Threshold out negative/background artifacts and normalize to [0, 1]
    activation = np.clip(activation, 0, None)
    activation = (activation - np.min(activation)) / (np.max(activation) - np.min(activation) + 1e-8)
    
    return activation

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load image at {mask_path}")
    # Binarize clearly (0 and 1)
    mask = (mask > 127).astype(np.uint8)
    return mask

def metric_pointing_game(activation_map, mask):
    """
    Returns 1 if the maximum activation falls inside the ground-truth tumor.
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(activation_map)
    max_x, max_y = max_loc
    
    # 1 if Hit, 0 if Miss
    is_hit = 1 if mask[max_y, max_x] > 0 else 0
    return is_hit, max_loc

def metric_iou(activation_map, mask, threshold=0.5):
    """
    Intersection over Union of the highly activated regions and the mask.
    """
    # Threshold the continuous heatmap to get the focal region
    binary_focus = (activation_map > threshold).astype(np.uint8)
    
    intersection = np.logical_and(binary_focus, mask).sum()
    union = np.logical_or(binary_focus, mask).sum()
    
    iou = intersection / union if union > 0 else 0.0
    return iou, binary_focus

def sanity_check_randomization(mask_shape, mask):
    """
    Simulates a randomly initialized/scrambled network output to act as a 
    baseline showing the model outperforms random chance.
    """
    random_heatmap = np.random.rand(mask_shape[0], mask_shape[1])
    rand_iou, _ = metric_iou(random_heatmap, mask, threshold=0.8) # Stricter for noise
    rand_pg, _ = metric_pointing_game(random_heatmap, mask)
    return rand_iou, rand_pg

if __name__ == "__main__":
    # --- Edit these file names to match the images you uploaded ---
    HEATMAP_PATH = "2v.png" 
    MASK_PATH = "2l.png"       

    try:
        print("=== Quantitative Interpretability Evaluation ===")
        
        # 1. Prepare data
        activation_map = extract_activation_from_jet_overlay(HEATMAP_PATH)
        mask = load_mask(MASK_PATH)
        
        # Make sure shapes align just in case
        if activation_map.shape != mask.shape:
            mask = cv2.resize(mask, (activation_map.shape[1], activation_map.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. Pointing Game
        pg_hit, peak_loc = metric_pointing_game(activation_map, mask)
        print(f"[*] Pointing Game Hit (1=Hit, 0=Miss): {pg_hit}")
        print(f"    Peak Activation Location (X, Y): {peak_loc}")

        # 3. IoU
        # Threshold chosen to represent the core 'hot' zone (top 50% activation)
        iou_score, binary_heatmap = metric_iou(activation_map, mask, threshold=0.5)
        print(f"[*] Attention-Mask IoU (Overlap): {iou_score:.4f}")

        # 4. Randomization Sanity Check (Expected to be near zero)
        num_simulations = 1000
        rand_iou_total = 0
        rand_pg_total = 0
        
        for _ in range(num_simulations):
            r_iou, r_pg = sanity_check_randomization(mask.shape, mask)
            rand_iou_total += r_iou
            rand_pg_total += r_pg
            
        print(f"[*] Sanity Check Baseline (Over {num_simulations} random runs):")
        print(f"    - Random IoU Expectation: {rand_iou_total/num_simulations:.4f}")
        print(f"    - Random Pointing Hit Rate: {rand_pg_total/num_simulations:.4f}")

    except Exception as e:
        print(f"Error executing evaluation: {str(e)}")
