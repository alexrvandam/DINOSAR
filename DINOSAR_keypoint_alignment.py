"""
Keypoint-based alignment module for DINO patch matching
Supports COCO JSON format with Procrustes and Homography transforms
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, List


class KeypointAligner:
    """
    Handles keypoint-based alignment between two images using COCO JSON format.
    
    COCO JSON format expected:
    {
        "images": [
            {"id": 1, "file_name": "image1.jpg"},
            {"id": 2, "file_name": "image2.jpg"}
        ],
        "annotations": [
            {
                "image_id": 1,
                "keypoints": [x1, y1, v1, x2, y2, v2, ...],  # visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                "num_keypoints": 10
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "beetle",
                "keypoints": ["head", "pronotum_left", "pronotum_right", ...]
            }
        ]
    }
    """
    
    def __init__(self, coco_json_path: str):
        """Load COCO JSON keypoint annotations."""
        with open(coco_json_path, 'r') as f:
            self.coco = json.load(f)
        
        # Build mappings
        self.image_id_to_filename = {
            str(img['id']): img['file_name'] 
            for img in self.coco.get('images', [])
        }
        self.filename_to_image_id = {
            img['file_name']: str(img['id']) 
            for img in self.coco.get('images', [])
        }
        
        # Store annotations by image_id
        self.annotations = {}
        for ann in self.coco.get('annotations', []):
            img_id = str(ann['image_id'])
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
    
    def get_keypoints(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract keypoints for an image.
        
        Returns:
            np.ndarray of shape [N, 3] with columns [x, y, visibility]
            or None if no keypoints found
        """
        # Try to match by filename
        filename = Path(image_path).name
        img_id = self.filename_to_image_id.get(filename)
        
        if img_id is None:
            # Try matching by stem (without extension)
            stem = Path(image_path).stem
            for fname, iid in self.filename_to_image_id.items():
                if Path(fname).stem == stem:
                    img_id = iid
                    break
        
        if img_id is None:
            return None
        
        # Get annotations for this image
        anns = self.annotations.get(img_id, [])
        if not anns:
            return None
        
        # Take first annotation (or could merge multiple)
        ann = anns[0]
        kpts = np.array(ann['keypoints']).reshape(-1, 3)
        
        # Filter to visible keypoints (visibility >= 1)
        visible_mask = kpts[:, 2] >= 1
        kpts_visible = kpts[visible_mask]
        
        if len(kpts_visible) == 0:
            return None
        
        return kpts_visible
    
    def get_corresponding_keypoints(self, 
                                   img1_path: str, 
                                   img2_path: str,
                                   min_points: int = 3) -> Tuple[Optional[np.ndarray], 
                                                                  Optional[np.ndarray]]:
        """
        Get corresponding keypoints between two images.
        
        Returns:
            (kpts1, kpts2): Both [N, 2] arrays with matched keypoints,
                           or (None, None) if insufficient correspondences
        """
        kpts1_full = self.get_keypoints(img1_path)
        kpts2_full = self.get_keypoints(img2_path)
        
        if kpts1_full is None or kpts2_full is None:
            return None, None
        
        # Match by index (assuming same keypoint order)
        # For more robust matching, could use keypoint names from categories
        min_len = min(len(kpts1_full), len(kpts2_full))
        
        kpts1 = kpts1_full[:min_len, :2]  # [N, 2] - just x, y
        kpts2 = kpts2_full[:min_len, :2]
        
        if len(kpts1) < min_points:
            return None, None
        
        return kpts1, kpts2


def procrustes_alignment(src_points: np.ndarray, 
                        dst_points: np.ndarray,
                        allow_scaling: bool = False,
                        allow_reflection: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Procrustes alignment: finds optimal rotation, translation (and optionally scaling)
    to align src_points to dst_points.
    
    Args:
        src_points: [N, 2] source keypoints
        dst_points: [N, 2] destination keypoints  
        allow_scaling: If True, estimate similarity transform (rot + trans + scale)
        allow_reflection: If True, allow reflection (determinant can be negative)
    
    Returns:
        affine_2x3: 2x3 affine transformation matrix [R|t] or [sR|t]
        info: dict with alignment diagnostics
    """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[1] == 2
    
    n_points = src_points.shape[0]
    
    # Centroids
    src_centroid = src_points.mean(axis=0)
    dst_centroid = dst_points.mean(axis=0)
    
    # Center the points
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Scale factors (for similarity transform)
    src_scale = np.sqrt((src_centered ** 2).sum() / n_points)
    dst_scale = np.sqrt((dst_centered ** 2).sum() / n_points)
    
    if src_scale < 1e-8 or dst_scale < 1e-8:
        # Degenerate case
        return np.eye(2, 3), {"error": "degenerate_points"}
    
    # Normalize for rotation estimation
    src_normalized = src_centered / src_scale
    dst_normalized = dst_centered / dst_scale
    
    # Compute rotation using SVD
    # R = V @ U^T where M = U @ S @ V^T = src^T @ dst
    M = src_normalized.T @ dst_normalized
    U, S, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T
    
    # Handle reflection (if determinant is negative)
    det = np.linalg.det(R)
    if det < 0 and not allow_reflection:
        # Flip the last column of V to prevent reflection
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        det = np.linalg.det(R)
    
    # Estimate scale
    if allow_scaling:
        scale = dst_scale / src_scale
    else:
        scale = 1.0
    
    # Construct transformation: dst = scale * R @ src + t
    # We need: t = dst_centroid - scale * R @ src_centroid
    t = dst_centroid - scale * R @ src_centroid
    
    # Build 2x3 affine matrix
    affine = np.zeros((2, 3))
    affine[:, :2] = scale * R
    affine[:, 2] = t
    
    # Compute alignment error (RMSE)
    transformed = (affine[:, :2] @ src_points.T).T + affine[:, 2]
    errors = np.linalg.norm(transformed - dst_points, axis=1)
    rmse = np.sqrt((errors ** 2).mean())
    
    info = {
        "method": "procrustes",
        "n_points": n_points,
        "scale": float(scale),
        "rotation_det": float(det),
        "rmse": float(rmse),
        "max_error": float(errors.max()),
        "allow_scaling": allow_scaling,
        "allow_reflection": allow_reflection
    }
    
    return affine, info


def homography_alignment(src_points: np.ndarray,
                        dst_points: np.ndarray,
                        ransac: bool = True,
                        ransac_threshold: float = 3.0) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Homography-based alignment using OpenCV.
    More flexible than Procrustes, handles perspective distortion.
    
    Args:
        src_points: [N, 2] source keypoints
        dst_points: [N, 2] destination keypoints
        ransac: Use RANSAC for robust estimation
        ransac_threshold: RANSAC inlier threshold in pixels
    
    Returns:
        homography_3x3: 3x3 homography matrix, or None if estimation fails
        info: dict with diagnostics
    """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[1] == 2
    
    n_points = src_points.shape[0]
    
    if n_points < 4:
        return None, {"error": "need_at_least_4_points", "n_points": n_points}
    
    # Estimate homography
    if ransac:
        H, mask = cv2.findHomography(
            src_points.astype(np.float32),
            dst_points.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold
        )
        inliers = mask.sum() if mask is not None else 0
    else:
        H = cv2.findHomography(
            src_points.astype(np.float32),
            dst_points.astype(np.float32),
            0  # Use all points
        )[0]
        inliers = n_points
        mask = np.ones(n_points, dtype=np.uint8)
    
    if H is None:
        return None, {"error": "homography_estimation_failed"}
    
    # Compute reprojection errors
    src_homogeneous = np.column_stack([src_points, np.ones(n_points)])
    projected = (H @ src_homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:]
    
    errors = np.linalg.norm(projected - dst_points, axis=1)
    rmse = np.sqrt((errors ** 2).mean())
    
    info = {
        "method": "homography",
        "n_points": n_points,
        "inliers": int(inliers),
        "inlier_ratio": float(inliers / n_points),
        "rmse": float(rmse),
        "max_error": float(errors.max()),
        "ransac": ransac
    }
    
    return H, info


def affine_from_homography(H: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 homography to approximate 2x3 affine.
    Takes the upper-left 2x3 part and normalizes.
    """
    # Normalize by bottom-right element
    H_norm = H / H[2, 2]
    affine = H_norm[:2, :]
    return affine


def align_images_with_keypoints(aligner: KeypointAligner,
                                img1_path: str,
                                img2_path: str,
                                method: str = 'procrustes',
                                allow_scaling: bool = False,
                                ransac: bool = True,
                                min_points: int = 3) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Complete pipeline to align images using keypoints from COCO JSON.
    
    Args:
        aligner: KeypointAligner instance with loaded COCO JSON
        img1_path: Path to source image
        img2_path: Path to destination image
        method: 'procrustes', 'similarity', or 'homography'
        allow_scaling: For Procrustes/similarity, allow scale change
        ransac: For homography, use RANSAC
        min_points: Minimum number of keypoints required
    
    Returns:
        affine_2x3: 2x3 affine transformation matrix (or from homography)
        info: dict with alignment information
    """
    # Get corresponding keypoints
    kpts1, kpts2 = aligner.get_corresponding_keypoints(
        img1_path, img2_path, min_points=min_points
    )
    
    if kpts1 is None or kpts2 is None:
        return None, {
            "error": "insufficient_keypoints",
            "img1": img1_path,
            "img2": img2_path
        }
    
    info = {
        "img1": img1_path,
        "img2": img2_path,
        "n_keypoints": len(kpts1)
    }
    
    # Choose alignment method
    if method in ['procrustes', 'similarity']:
        affine, align_info = procrustes_alignment(
            kpts1, kpts2,
            allow_scaling=(method == 'similarity' or allow_scaling),
            allow_reflection=False
        )
        info.update(align_info)
        return affine, info
    
    elif method == 'homography':
        H, align_info = homography_alignment(
            kpts1, kpts2,
            ransac=ransac,
            ransac_threshold=3.0
        )
        info.update(align_info)
        
        if H is None:
            return None, info
        
        # Convert to affine for consistency with other methods
        affine = affine_from_homography(H)
        info['homography_3x3'] = H.tolist()
        
        return affine, info
    
    else:
        return None, {"error": f"unknown_method: {method}"}


def visualize_keypoint_alignment(img1_path: str,
                                 img2_path: str,
                                 kpts1: np.ndarray,
                                 kpts2: np.ndarray,
                                 affine: np.ndarray,
                                 output_path: str):
    """
    Visualize keypoint correspondences before and after alignment.
    """
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # Apply affine to img1
    from PIL import Image as PILImage
    H, W = img1.size
    a, b, c = affine[0]
    d, e, f = affine[1]
    
    # PIL uses inverse transform
    det = a * e - b * d
    if abs(det) < 1e-9:
        print("Warning: Singular affine matrix")
        img1_aligned = img1
    else:
        inv_affine = np.array([
            [e/det, -b/det, (b*f - e*c)/det],
            [-d/det, a/det, (d*c - a*f)/det]
        ])
        a_inv, b_inv, c_inv = inv_affine[0]
        d_inv, e_inv, f_inv = inv_affine[1]
        
        img1_aligned = img1.transform(
            img1.size,
            PILImage.AFFINE,
            (a_inv, b_inv, c_inv, d_inv, e_inv, f_inv),
            resample=PILImage.BILINEAR
        )
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original images with keypoints
    axes[0].imshow(img1)
    axes[0].scatter(kpts1[:, 0], kpts1[:, 1], c='red', s=100, marker='x')
    axes[0].set_title(f'Image 1 ({len(kpts1)} keypoints)')
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].scatter(kpts2[:, 0], kpts2[:, 1], c='blue', s=100, marker='x')
    axes[1].set_title(f'Image 2 ({len(kpts2)} keypoints)')
    axes[1].axis('off')
    
    # Aligned overlay
    # Blend images
    arr1 = np.array(img1_aligned).astype(float)
    arr2 = np.array(img2).astype(float)
    blended = (0.5 * arr1 + 0.5 * arr2).astype(np.uint8)
    
    axes[2].imshow(blended)
    
    # Transform keypoints from img1
    kpts1_homo = np.column_stack([kpts1, np.ones(len(kpts1))])
    kpts1_transformed = (affine @ kpts1_homo.T).T
    
    axes[2].scatter(kpts1_transformed[:, 0], kpts1_transformed[:, 1], 
                   c='red', s=100, marker='x', label='Img1 (transformed)')
    axes[2].scatter(kpts2[:, 0], kpts2[:, 1], 
                   c='blue', s=100, marker='o', label='Img2 (target)')
    axes[2].set_title('Aligned Overlay')
    axes[2].legend()
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved keypoint alignment visualization to {output_path}")


# ============================================================================
# Convenience functions for integration into existing scripts
# ============================================================================

def try_keypoint_alignment(img1_path: str,
                          img2_path: str,
                          keypoint_json: Optional[str] = None,
                          method: str = 'procrustes',
                          fallback_to_mask: bool = True,
                          mask1: Optional[np.ndarray] = None,
                          mask2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Try keypoint alignment, fall back to mask-based PCA if keypoints unavailable.
    
    Returns:
        affine_2x3: Transformation matrix
        info: Information about the alignment method used
    """
    info = {"method": "none", "fallback": False}
    
    # Try keypoint alignment first
    if keypoint_json is not None:
        try:
            aligner = KeypointAligner(keypoint_json)
            affine, kpt_info = align_images_with_keypoints(
                aligner, img1_path, img2_path,
                method=method,
                allow_scaling=(method == 'similarity'),
                ransac=True
            )
            
            if affine is not None:
                info.update(kpt_info)
                info["success"] = True
                print(f"✓ Keypoint alignment successful: {method}")
                print(f"  Used {kpt_info.get('n_keypoints', 0)} keypoints")
                print(f"  RMSE: {kpt_info.get('rmse', 0):.2f} pixels")
                return affine, info
            else:
                print(f"✗ Keypoint alignment failed: {kpt_info.get('error', 'unknown')}")
                
        except Exception as e:
            print(f"✗ Keypoint alignment error: {e}")
    
    # Fallback to mask-based alignment
    if fallback_to_mask and mask1 is not None and mask2 is not None:
        print("→ Falling back to mask-based PCA alignment")
        from .alignment_utils import pca_rigid_align  # Import from your existing code
        affine, pca_info = pca_rigid_align(mask1, mask2)
        info.update(pca_info)
        info["fallback"] = True
        info["success"] = True
        return affine, info
    
    # No alignment
    print("→ No alignment applied (identity transform)")
    return np.eye(2, 3), {"method": "identity", "success": False}


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test keypoint alignment')
    parser.add_argument('img1', help='Source image')
    parser.add_argument('img2', help='Destination image')
    parser.add_argument('--keypoints', required=True, help='COCO JSON with keypoints')
    parser.add_argument('--method', choices=['procrustes', 'similarity', 'homography'],
                       default='procrustes')
    parser.add_argument('--output', default='keypoint_alignment_test.png',
                       help='Output visualization path')
    args = parser.parse_args()
    
    # Load keypoints
    aligner = KeypointAligner(args.keypoints)
    
    # Get corresponding keypoints
    kpts1, kpts2 = aligner.get_corresponding_keypoints(args.img1, args.img2)
    
    if kpts1 is None:
        print("ERROR: No keypoints found!")
        exit(1)
    
    print(f"Found {len(kpts1)} corresponding keypoints")
    
    # Align
    affine, info = align_images_with_keypoints(
        aligner, args.img1, args.img2,
        method=args.method
    )
    
    if affine is None:
        print(f"ERROR: Alignment failed - {info.get('error')}")
        exit(1)
    
    print(f"\nAlignment successful!")
    print(f"Method: {info['method']}")
    print(f"RMSE: {info['rmse']:.2f} pixels")
    print(f"Max error: {info['max_error']:.2f} pixels")
    
    if 'inliers' in info:
        print(f"Inliers: {info['inliers']}/{info['n_points']}")
    
    print(f"\nAffine matrix:\n{affine}")
    
    # Visualize
    visualize_keypoint_alignment(args.img1, args.img2, kpts1, kpts2, affine, args.output)
