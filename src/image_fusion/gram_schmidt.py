import numpy as np


def gram_schmidt_fusion_rgb(multispectral, pseudo_rgb, alpha=2):
    """
    Fuse the multispectral information into a pseudo-RGB image using an intensity substitution approach,
    with an adjustable multispectral boost factor.

    Args:
        multispectral (np.ndarray): Array of shape (H, W, C_ms) from all bands.
        pseudo_rgb (np.ndarray): Array of shape (H, W, 3) composed from selected bands (e.g., R, G, B).
        alpha (float): Boost factor to increase the multispectral impact.

    Returns:
        np.ndarray: Fused pseudo-RGB image (H, W, 3) as uint8.
    """
    pseudo_rgb = pseudo_rgb.astype(np.float32)
    multispectral = multispectral.astype(np.float32)

    intensity_rgb = np.mean(pseudo_rgb, axis=-1, keepdims=True)
    intensity_ms = np.mean(multispectral, axis=-1, keepdims=True)
    # boost the multispectral intensity before computing the ratio
    ratio = (alpha * intensity_ms) / (intensity_rgb + 1e-8)  # avoid division by zero

    fused = pseudo_rgb * ratio
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused