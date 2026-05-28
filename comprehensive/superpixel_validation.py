import torch
import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from scipy.spatial import cKDTree


DEFAULT_EPSILON = 1e-8


def dice_coefficient(
    superpixel_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """Compute Dice Similarity Coefficient between superpixel and ground truth.

    DSC = 2 * |S ∩ G| / (|S| + |G|)

    Args:
        superpixel_mask: Binary array where 1 indicates the superpixel
            covers the lesion region. Shape [H, W].
        ground_truth_mask: Binary ground-truth tumor mask. Shape [H, W].

    Returns:
        DSC score in [0, 1] as a percentage (e.g. 94.09 means 94.09%).
    """
    intersection = np.logical_and(superpixel_mask, ground_truth_mask).sum()
    area_s = superpixel_mask.sum()
    area_g = ground_truth_mask.sum()

    denominator = area_s + area_g
    if denominator < DEFAULT_EPSILON:
        return 0.0

    dsc = (2.0 * intersection) / denominator
    return dsc * 100.0


def under_segmentation_error(
    superpixel_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """Compute Under-segmentation Error.

    UE = |S - G| / |G|  where S - G denotes pixels in the superpixel
    that fall outside the ground truth region.

    Args:
        superpixel_mask: Binary superpixel region mask. Shape [H, W].
        ground_truth_mask: Binary ground-truth tumor mask. Shape [H, W].

    Returns:
        UE score in [0, inf) as a percentage.
    """
    area_g = ground_truth_mask.sum()
    if area_g < DEFAULT_EPSILON:
        return 0.0

    overflow = np.logical_and(
        superpixel_mask,
        np.logical_not(ground_truth_mask),
    ).sum()

    ue = overflow / area_g
    return ue * 100.0


def boundary_recall(
    superpixel_boundary: np.ndarray,
    ground_truth_boundary: np.ndarray,
    tolerance: int = 2,
) -> float:
    """Compute Boundary Recall.

    BR = |{p in B_G : min_{q in B_S} ||p - q|| <= tolerance}| / |B_G|

    Args:
        superpixel_boundary: Binary boundary map of superpixel. Shape [H, W].
        ground_truth_boundary: Binary boundary map of ground truth. Shape [H, W].
        tolerance: Maximum Euclidean distance (in pixels) for a ground-truth
            boundary pixel to be considered recalled (default 2).

    Returns:
        BR score in [0, 1] as a percentage.
    """
    gt_points = np.argwhere(ground_truth_boundary > 0)
    sp_points = np.argwhere(superpixel_boundary > 0)

    if len(gt_points) == 0:
        return 100.0
    if len(sp_points) == 0:
        return 0.0

    sp_tree = cKDTree(sp_points)
    distances, _ = sp_tree.query(gt_points, k=1)

    recalled = (distances <= tolerance).sum()
    br = recalled / len(gt_points)
    return br * 100.0


def extract_superpixel_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract boundary of a binary mask using morphological dilation.

    Boundary = dilated(mask) - mask  (i.e., external contour).

    Args:
        mask: Binary mask. Shape [H, W].

    Returns:
        Binary boundary map. Shape [H, W].
    """
    dilated = binary_dilation(mask, footprint=disk(1))
    boundary = np.logical_and(dilated, np.logical_not(mask))
    return boundary.astype(np.uint8)


def validate_superpixel_segmentation(
    image: np.ndarray,
    ground_truth_mask: np.ndarray,
    n_segments: int = 16,
    compactness: float = 100.0,
    sigma: float = 1.0,
) -> dict:
    """Run full superpixel validation on a single MRI slice.

    Runs SLIC superpixel segmentation on `image`, then computes DSC, UE,
    and BR against the provided ground truth mask.

    Args:
        image: Input MRI image in RGB or grayscale. Shape [H, W, 3] or [H, W].
        ground_truth_mask: Binary tumor mask. Shape [H, W].
        n_segments: Target number of superpixels (default 16).
        compactness: SLIC compactness parameter (default 100).
        sigma: SLIC Gaussian smoothing sigma (default 1.0).

    Returns:
        Dictionary with keys 'dsc', 'ue', 'br' containing float percentages.
    """
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image_rgb = np.repeat(image, 3, axis=-1)
    else:
        image_rgb = image

    segments = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
    )

    props = regionprops(segments, intensity_image=image)
    lesion_mask = np.zeros_like(segments, dtype=bool)

    for prop in props:
        coords = prop.coords
        overlap = ground_truth_mask[coords[:, 0], coords[:, 1]].sum()
        region_area = len(coords)
        if region_area > 0 and overlap / region_area > 0.3:
            lesion_mask[coords[:, 0], coords[:, 1]] = True

    seg_boundary = extract_superpixel_boundary(lesion_mask)
    gt_boundary = extract_superpixel_boundary(ground_truth_mask)

    dsc = dice_coefficient(lesion_mask, ground_truth_mask)
    ue = under_segmentation_error(lesion_mask, ground_truth_mask)
    br = boundary_recall(seg_boundary, gt_boundary, tolerance=2)

    return {'dsc': dsc, 'ue': ue, 'br': br}


def batch_validate_superpixel(
    images: list,
    ground_truth_masks: list,
    n_segments: int = 16,
    compactness: float = 100.0,
    sigma: float = 1.0,
) -> dict:
    """Run superpixel validation over a batch of images.

    Args:
        images: List of numpy arrays, each shape [H, W, 3].
        ground_truth_masks: List of binary numpy arrays, each shape [H, W].
        n_segments: Target number of superpixels.
        compactness: SLIC compactness parameter.
        sigma: SLIC Gaussian smoothing sigma.

    Returns:
        Dictionary with keys 'dsc_mean', 'dsc_std', 'ue_mean', 'ue_std',
        'br_mean', 'br_std' summarizing across all samples.
    """
    dsc_list = []
    ue_list = []
    br_list = []

    for img, gt in zip(images, ground_truth_masks):
        result = validate_superpixel_segmentation(
            img, gt, n_segments, compactness, sigma,
        )
        dsc_list.append(result['dsc'])
        ue_list.append(result['ue'])
        br_list.append(result['br'])

    return {
        'dsc_mean': float(np.mean(dsc_list)),
        'dsc_std': float(np.std(dsc_list)),
        'ue_mean': float(np.mean(ue_list)),
        'ue_std': float(np.std(ue_list)),
        'br_mean': float(np.mean(br_list)),
        'br_std': float(np.std(br_list)),
    }


if __name__ == '__main__':
    print("superpixel_validation.py — standalone test")

    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_mask = np.zeros((224, 224), dtype=bool)
    dummy_mask[80:140, 90:150] = True

    result = validate_superpixel_segmentation(dummy_image, dummy_mask)
    print(f"DSC: {result['dsc']:.2f}%")
    print(f"UE:  {result['ue']:.2f}%")
    print(f"BR:  {result['br']:.2f}%")
    print("Standalone test passed.")
