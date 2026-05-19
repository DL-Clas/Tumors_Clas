import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift, rotate, gaussian_filter, binary_fill_holes
from PIL import Image
import os

# ==========================================
# 1. Real image loading and simulation model
# ==========================================
def load_real_mri(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file cannot be found: {image_path}")
    img = Image.open(image_path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0

class MockModel:
    def predict(self, img):
        feature = np.mean(img) + np.std(img) * 0.8
        return np.clip(feature, 0.1, 0.99)

# ==========================================
# 2. (Clinical Realistic) Perturbation Functions
# ==========================================
class MRIPerturbations:
    @staticmethod
    def intensity_nonuniformity(img):
        """1. Intensity Non-Uniformity: Pure multiplicative low-frequency field, mean around 1, simulating B1 field local overexposure"""
        h, w = img.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        # Gaussian distribution is generated, and the center is biased to the upper left corner.
        bias = np.exp(-((X + 0.5)**2 + (Y + 0.5)**2) / 3.0)
        # Strictly control the zoom range between [0.75, 1.3] to avoid global darkening.
        bias_field = 0.75 + 0.55 * (bias / np.max(bias))
        return np.clip(img * bias_field, 0, 1)

    @staticmethod
    def motion_artifacts(img):
        """2. Motion Artifacts: Strictly comply with MRI physics, generating Ghosting only along a single axis (phase encoding direction Y)"""
        f = np.fft.fft2(img)
        h, w = img.shape
        
        # Construct phase shifts based only on the Y-axis
        ky = np.linspace(-np.pi, np.pi, h)
        
        # Simulate 4 obvious swallowing/breathing movements, causing local phase error.
        # The amplitude is controlled at about 3 pixels, resulting in a clearly visible ghost with the same direction.
        phase_shift = np.exp(-1j * 3.0 * np.sin(4 * ky))
        
        # Broadcast one-dimensional phase offset to two-dimensional and multiply.
        f_motion = f * phase_shift[:, np.newaxis]
        
        img_motion = np.abs(np.fft.ifft2(f_motion))
        return np.clip(img_motion, 0, 1)

    @staticmethod
    def scanner_site_shifts(img):
        """3. Scanner/Site Shifts: Affine transformation simulating patient positioning inaccuracies"""
        img_rot = rotate(img, angle=5, reshape=False, mode='constant', cval=0.0)
        img_shifted = shift(img_rot, shift=(10, -8), mode='constant', cval=0.0)
        return img_shifted

    @staticmethod
    def resolution_changes(img):
        """4. Resolution Changes: Moderate K-space truncation,符合 clinical low-coil scanning experience"""
        f = np.fft.fftshift(np.fft.fft2(img))
        h, w = img.shape
        
        # Truncate the high frequency and keep 18% of the frequency information in the center.
        cy, cx = h // 2, w // 2
        dy, dx = int(h * 0.09), int(w * 0.09)
        
        mask = np.zeros_like(img)
        mask[cy-dy:cy+dy, cx-dx:cx+dx] = 1.0
        
        img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))
        return np.clip(img_low, 0, 1)

    @staticmethod
    def skull_stripping_variation(img):
        """5. Skull-stripping Variation: Simulating asymmetric algorithm "cutting over" (local excision of cortex/lesion) """
        mask = img > 0.04
        mask = binary_fill_holes(mask).astype(np.float32)
        
        # Generate an "error mask" with a serious shift to the upper left to intersect with the real structure.
        # This simulated that the BET algorithm failed to locate the center point, resulting in the lower right cerebral cortex being directly cut off.
        defect_mask = shift(mask, shift=(-20, -20), mode='constant', cval=0.0)
        
        final_mask = mask * defect_mask
        # Slightly smooth the boundary and simulate the soft cutting edge of the algorithm.
        final_mask = gaussian_filter(final_mask, sigma=1.5)
        
        return img * final_mask

    @staticmethod
    def modality_differences(img):
        """6. Modality Differences: Gaussian blurred mask combined with nonlinear contrast inversion, seamless fusion without sharp white edges"""
        # Extract rough mask
        mask = img > 0.04
        mask = binary_fill_holes(mask).astype(np.float32)
        
        # Use a larger Sigma (e.g., 4.0) to heavily blur the mask, eliminating the edge cutting sensation
        smooth_brain_mask = gaussian_filter(mask, sigma=4.0)
        
        # Invert and gamma correct the image (simulate water brightening)
        fake_t2_contrast = (1.0 - img) ** 1.6
        
        # Seamless alpha mixing using feathered mask (Alpha Blending)
        # The contrast inside the brain parenchyma is reversed, and the air outside the skeleton is darkened (multiplied by 0.3).
        fake_t2 = fake_t2_contrast * smooth_brain_mask + (img * 0.3) * (1.0 - smooth_brain_mask)
        
        return np.clip(fake_t2, 0, 1)

# ==========================================
# 3. Strict evaluation framework
# ==========================================
def evaluate_robustness(image_path):
    original_img = load_real_mri(image_path)
    model = MockModel()
    
    perturbations = {
        "Baseline (Original)": lambda x: x,
        "1. Intensity Nonuniformity": MRIPerturbations.intensity_nonuniformity,
        "2. Motion Artifacts": MRIPerturbations.motion_artifacts,
        "3. Scanner/Site Shifts": MRIPerturbations.scanner_site_shifts,
        "4. Resolution Changes": MRIPerturbations.resolution_changes,
        "5. Skull-stripping Variation": MRIPerturbations.skull_stripping_variation,
        "6. Modality Differences": MRIPerturbations.modality_differences
    }
    
    results = {}
    images_to_plot = {}
    baseline_pred = model.predict(original_img)
    
    print(f"{'Perturbation Type':<30} | {'Model Output':<15} | {'Diff vs Baseline'}")
    print("-" * 70)
    
    for name, func in perturbations.items():
        perturbed_img = func(original_img)
        images_to_plot[name] = perturbed_img
        pred = model.predict(perturbed_img)
        deviation = pred - baseline_pred
        results[name] = pred
        
        if name == "Baseline (Original)":
            print(f"{name:<30} | {pred:.4f}          | Reference value")
        else:
            print(f"{name:<30} | {pred:.4f}          | {deviation:+.4f}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Peer-Review Quality MRI Perturbations (Clinical Level)", fontsize=18, fontweight='bold')
    
    for ax, (name, img) in zip(axes.flatten(), images_to_plot.items()):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{name}\nPred: {results[name]:.3f}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
    axes[1, 3].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    IMAGE_PATH = "./data/BTD-3/Pituitary tumor/Pit_P173_1216.png" 
    evaluate_robustness(IMAGE_PATH)
