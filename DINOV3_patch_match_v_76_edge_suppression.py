"""
Fixed DINOv3 Patch Matching with ATTENTION-BASED RANKING (v60 + edge suppression)
- Properly extracts attention maps from DINOv3
- Ranks patches by attention importance (not just similarity)
- Two sparse-match flavors: attention-ranked and DINOv3-demo (mutual/ratio/RANSAC)
- Transparent attention overlays
- Histogram/stat modes: foreground / attention_weighted / demo_inliers / bidirectional / topk_mean
- NEW: Ignore edge patches via --border-patches, mask erosion via --mask-erode-px,
       and optional attention-intersection foreground via --attn-fg
"""

import os, argparse, csv, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms as T
from scipy import stats
from typing import Optional, Tuple, Dict
from DINOSAR_keypoint_alignment import KeypointAligner, align_images_with_keypoints

# YAML run Metadata:
def save_run_metadata(args, output_dir: str, img_size: int, grid_size: int, patch_size: int):
    """Save all run parameters to YAML for reproducibility."""
    try:
        import yaml
    except ImportError:
        print("Warning: pyyaml not installed, skipping metadata file")
        return None
    
    metadata = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'script_version': 'v67',
            'command': ' '.join(sys.argv)
        },
        'input_files': {
            'image_A': str(Path(args.img1).absolute()),
            'image_B': str(Path(args.img2).absolute()),
            'keypoints_json': str(Path(args.keypoints_json).absolute()) if args.keypoints_json else None,
            'consensus_shape_db': args.consensus_shape_db
        },
        'model': {
            'checkpoint': str(Path(args.dinov3_local_ckpt).absolute()),
            'architecture': args.dinov3_arch,
            'img_size': img_size,
            'grid_size': grid_size,
            'patch_size': patch_size
        },
        'layers': {
            'feature_layers': args.layers if args.layers else [args.layer],
            'layer_aggregation': args.layer_agg,
            'attention_layers': args.attn_layers
        },
        'foreground': {
            'fg_threshold': args.fg_thresh,
            'crop_enabled': not args.no_crop,
            'mask_erode_px': args.mask_erode_px,
            'border_patches_excluded': args.border_patches,
            'attn_fg_enabled': args.attn_fg,
            'attn_fg_threshold': args.attn_fg_thresh if args.attn_fg else None
        },
        'matching': {
            'mode': args.matching_mode,
            'match_flavor': args.match_flavor,
            'score_mode': args.score_mode,
            'top_matches_visualized': args.top_matches
        },
        'tps_parameters': {
            'grid_cell_size': args.tps_grid_cell_size,
            'grid_rows': args.tps_grid_rows,
            'grid_cols': args.tps_grid_cols,
            'grid_snap_to_patch': args.tps_grid_snap
        } if args.matching_mode in ['tps_homology', 'both'] else None,
        'alignment': {
            'enabled': args.keypoints_json is not None,
            'method': args.align_method,
            'min_keypoints': args.min_kpts
        },
        'output': {
            'directory': str(Path(output_dir).absolute()),
            'hist_min_joint': args.hist_min_joint
        }
    }
    
    yaml_path = os.path.join(output_dir, 'run_metadata.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Saved run metadata: run_metadata.yaml")
    return yaml_path

def warp_image_and_mask(img_pil, mask_np, affine_2x3, out_size):
    """Warp PIL image + numpy mask with an affine, keep white background."""
    w, h = out_size, out_size
    img_np = np.array(img_pil)
    warped_img = cv2.warpAffine(img_np, affine_2x3, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
    warped_mask = cv2.warpAffine((mask_np*255).astype(np.uint8), affine_2x3, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0) > 0
    from PIL import Image
    return Image.fromarray(warped_img), warped_mask.astype(float)

def align_masks_with_keypoints(img1_path: str,
                               img2_path: str,
                               keypoints_json: str,
                               method: str,
                               maskA: np.ndarray,
                               maskB: np.ndarray,
                               out_size: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute affine via keypoints and warp masks only (no image warping).
    """
    aligner = KeypointAligner(keypoints_json)
    affine, info = align_images_with_keypoints(aligner, img1_path, img2_path, method=method)
    if affine is None:
        print("✗ Keypoint mask alignment failed; returning originals.")
        return maskA, maskB, {"success": False, "error": info.get("error", "unknown")}
    w = h = int(out_size)
    wA = cv2.warpAffine((maskA*255).astype(np.uint8), affine, (w, h),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    wB = maskB.astype(bool)
    return wA.astype(np.float32), wB.astype(np.float32), {"success": True, **info}


def transform_keypoints(keypoints: np.ndarray, 
                       img_original: Image.Image,
                       img_processed: Image.Image,
                       crop_bbox: Optional[Tuple[int, int, int, int]] = None,
                       affine_2x3: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Transform keypoints from original image space to processed image space.
    
    Args:
        keypoints: [N, 2] in original image coordinates
        img_original: Original PIL image
        img_processed: Processed PIL image (after resize/crop)
        crop_bbox: (x0, y0, x1, y1) if image was cropped
        affine_2x3: 2x3 affine matrix if image was aligned
    
    Returns:
        keypoints_transformed: [N, 2] in processed image coordinates
    """
    kpts = keypoints.copy()
    
    # Apply crop translation
    if crop_bbox is not None:
        x0, y0, x1, y1 = crop_bbox
        kpts[:, 0] -= x0
        kpts[:, 1] -= y0
    
    # Apply resize scaling
    orig_w, orig_h = img_original.size if crop_bbox is None else (crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1])
    proc_w, proc_h = img_processed.size
    
    scale_x = proc_w / orig_w
    scale_y = proc_h / orig_h
    
    kpts[:, 0] *= scale_x
    kpts[:, 1] *= scale_y
    
    # Apply affine transformation if provided
    if affine_2x3 is not None:
        # Convert to homogeneous coordinates
        ones = np.ones((len(kpts), 1))
        kpts_hom = np.hstack([kpts, ones])
        # Apply affine: [2x3] @ [Nx3].T = [2xN], then transpose
        kpts = (affine_2x3 @ kpts_hom.T).T
    
    return kpts

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_histogram_statistics(dissimilarity_values: np.ndarray) -> Dict:
    """Compute comprehensive statistics for dissimilarity histogram."""
    if len(dissimilarity_values) == 0:
        return {}
    stats_dict = {
        'mean': float(np.mean(dissimilarity_values)),
        'median': float(np.median(dissimilarity_values)),
        'std': float(np.std(dissimilarity_values)),
        'min': float(np.min(dissimilarity_values)),
        'max': float(np.max(dissimilarity_values)),
        'p25': float(np.percentile(dissimilarity_values, 25)),
        'p75': float(np.percentile(dissimilarity_values, 75)),
        'p90': float(np.percentile(dissimilarity_values, 90)),
        'p95': float(np.percentile(dissimilarity_values, 95)),
        'p99': float(np.percentile(dissimilarity_values, 99)),
        'iqr': float(np.percentile(dissimilarity_values, 75) - np.percentile(dissimilarity_values, 25)),
        'tail_ratio': float((dissimilarity_values > 0.3).sum() / len(dissimilarity_values)),
        'heavy_tail_ratio': float((dissimilarity_values > 0.5).sum() / len(dissimilarity_values)),
        'skewness': float(stats.skew(dissimilarity_values)),
        'kurtosis': float(stats.kurtosis(dissimilarity_values)),
        'right_tail_mean': float(dissimilarity_values[dissimilarity_values > np.median(dissimilarity_values)].mean()),
    }
    return stats_dict


def is_conspecific_threshold(dissim_stats: Dict, p95_threshold: float = 0.25, tail_ratio_threshold: float = 0.15) -> bool:
    p95 = dissim_stats.get('p95', 1.0)
    tail_ratio = dissim_stats.get('tail_ratio', 1.0)
    return (p95 < p95_threshold) and (tail_ratio < tail_ratio_threshold)


# ============================================================================
# DINOv3 Model Loading
# ============================================================================

def load_dinov3_local_checkpoint(ckpt_path: str, arch: str, img_size: int, device: str):
    import sys
    try:
        try:
            from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
        except ImportError:
            try:
                sys.path.insert(0, os.path.expanduser('~/models/git/dinov3'))
                from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
            except ImportError:
                from models.vision_transformer import vit_small, vit_base, vit_large
    except Exception as e:
        raise RuntimeError(
            f"Could not import DINOv3 vision_transformer. Error: {e}\nMake sure DINOv3 repo is cloned and in PYTHONPATH"
        ) from e

    arch = arch.lower()
    if arch in ('vits16', 'vit-s/16', 'vits', 'small'):
        model_fn = lambda: vit_small(patch_size=16, img_size=img_size)
        patch_size = 16
    elif arch in ('vitb16', 'vit-b/16', 'vitb', 'base'):
        model_fn = lambda: vit_base(patch_size=16, img_size=img_size)
        patch_size = 16
    elif arch in ('vitl16', 'vit-l/16', 'vitl', 'large'):
        model_fn = lambda: vit_large(patch_size=16, img_size=img_size)
        patch_size = 16
    else:
        raise ValueError(f"Unknown DINOv3 arch: {arch}")

    print(f"Building DINOv3 model: {arch}, patch_size={patch_size}")
    model = model_fn()

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {len(unexpected_keys)}")

    model = model.to(device).eval()
    print(f"Model loaded successfully on {device}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def processor(image: Image.Image):
        if image.size != (img_size, img_size):
            image = image.resize((img_size, img_size), Image.BILINEAR)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - mean) / std
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)

    return processor, model, patch_size


# ============================================================================
# Attention Map Extraction
# ============================================================================

def extract_attention_maps(model, image: Image.Image, processor, grid_size: int, patch_size: int, device: str = 'cuda') -> Dict[int, np.ndarray]:
    inputs = processor(image)
    attention_weights = []

    def recompute_attention_from_inputs(module, x_in: torch.Tensor) -> torch.Tensor:
        B, N, C = x_in.shape
        qkv = module.qkv(x_in)
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (q.shape[-1]) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        return attn.softmax(dim=-1)

    def attention_hook(module, inputs, output):
        try:
            x_in = inputs[0]
            attn = recompute_attention_from_inputs(module, x_in)
            attention_weights.append(attn.detach().cpu())
        except Exception as e:
            print(f"[extract_attention_maps] Warning: failed to recompute attn: {e}")
            attention_weights.append(None)

    hooks = []
    if hasattr(model, 'blocks'):
        for block in model.blocks:
            if hasattr(block, 'attn'):
                hooks.append(block.attn.register_forward_hook(attention_hook))

    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    attention_maps: Dict[int, np.ndarray] = {}
    num_patches = grid_size * grid_size

    for layer_idx, attn in enumerate(attention_weights):
        if attn is None:
            continue
        attn_avg = attn[0].mean(dim=0)  # [tokens, tokens]
        cls_to_tokens = attn_avg[0]
        if cls_to_tokens.numel() <= 1:
            continue
        cls_to_patches = cls_to_tokens[1:]
        patch_scores = cls_to_patches[-num_patches:].reshape(grid_size, grid_size).numpy()
        patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)
        heatmap = np.kron(patch_scores, np.ones((patch_size, patch_size)))
        attention_maps[layer_idx] = heatmap

    if len(attention_maps) == 0:
        try:
            with torch.no_grad():
                if hasattr(model, 'get_intermediate_layers'):
                    feats = model.get_intermediate_layers(inputs, n=range(len(model.blocks)), reshape=True, norm=True)
                    last = feats[-1].squeeze(0)
                    if last.ndim == 3:
                        token_grid = last
                    else:
                        token_grid = last[1:][-num_patches:].reshape(grid_size, grid_size, -1)
                    l2 = torch.norm(token_grid, dim=-1).cpu().numpy()
                    l2 = (l2 - l2.min()) / (l2.max() - l2.min() + 1e-8)
                    heatmap = np.kron(l2, np.ones((patch_size, patch_size)))
                    attention_maps[len(attention_weights)-1 if len(attention_weights)>0 else 0] = heatmap
                    print("[extract_attention_maps] Fallback synthesized attention from token norms.")
        except Exception as e:
            print(f"[extract_attention_maps] Fallback failed: {e}")

    return attention_maps


def save_attention_overlay(img: Image.Image, heatmap: np.ndarray, mask: np.ndarray, outfn: str, alpha: float = 0.5):
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    heatmap_masked = heatmap * mask
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    im = ax.imshow(heatmap_masked, cmap='jet', alpha=alpha, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    ax.set_title('Attention Heatmap (CLS → Patches)', fontsize=12, pad=10)
    fig.savefig(outfn, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)


# ============================================================================
# Patch Embeddings
# ============================================================================

def extract_patch_embeddings_dinov3(image: Image.Image, model, processor, patch_size: int, img_size: int, layer_idx: int = -1, device: str = 'cuda') -> torch.Tensor:
    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size))
    grid_size = img_size // patch_size
    num_patches = grid_size * grid_size
    inputs = processor(image)

    with torch.no_grad():
        if layer_idx < 0:
            actual_layer = len(model.blocks) + layer_idx if hasattr(model, 'blocks') else 11
        else:
            actual_layer = layer_idx
        if hasattr(model, 'get_intermediate_layers') and actual_layer >= 0:
            try:
                features = model.get_intermediate_layers(inputs, n=[actual_layer], return_class_token=False)
                if isinstance(features, (list, tuple)):
                    features = features[0]
                patch_tokens = features[0]
            except:
                features = model.forward_features(inputs)
                if isinstance(features, dict):
                    if 'x_norm_patchtokens' in features:
                        patch_tokens = features['x_norm_patchtokens'][0]
                    else:
                        seq = features['x']
                        patch_tokens = seq[0, 1:, :] if seq.shape[1] == num_patches + 1 else seq[0, -num_patches:, :]
                else:
                    patch_tokens = features[0, 1:, :] if features.shape[1] == num_patches + 1 else features[0, -num_patches:, :]
        else:
            features = model.forward_features(inputs)
            if isinstance(features, dict):
                if 'x_norm_patchtokens' in features:
                    patch_tokens = features['x_norm_patchtokens'][0]
                else:
                    seq = features['x']
                    patch_tokens = seq[0, 1:, :] if seq.shape[1] == num_patches + 1 else seq[0, -num_patches:, :]
            else:
                patch_tokens = features[0, 1:, :] if features.shape[1] == num_patches + 1 else features[0, -num_patches:, :]

    if patch_tokens.shape[0] != num_patches:
        raise RuntimeError(f"Token count mismatch: got {patch_tokens.shape[0]}, expected {num_patches}")
    return F.normalize(patch_tokens, p=2, dim=-1)


# ============================================================================
# Foreground Mask Generation
# ============================================================================

def generate_foreground_mask(img: Image.Image) -> np.ndarray:
    try:
        from rembg import remove
        fg = remove(img)
        return (np.array(fg.convert("L")) > 0).astype(float)
    except Exception as e:
        print(f"Warning: rembg failed ({e}), using threshold")
        return generate_clean_mask(img)


def generate_clean_mask(img: Image.Image, white_thresh: int = 240) -> np.ndarray:
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(gray.shape, dtype=float)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return (mask > 0).astype(float)


def erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return mask
    k = max(1, int(erode_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    eroded = cv2.erode((mask*255).astype(np.uint8), kernel)
    return (eroded > 0).astype(np.float32)

def build_attention_mask(attn_map: np.ndarray,
                         img_size: int,
                         patch_size: int,
                         thresh: float = 0.25,
                         open_px: int = 3) -> np.ndarray:
    """
    Convert a dense attention map (H x W) to a binary foreground mask.
    - Threshold in [0,1]
    - Optional morphological opening to remove speckles
    """
    # attn_map is already at image resolution in this codebase
    m = (attn_map >= float(thresh)).astype(np.uint8) * 255
    if open_px and open_px > 0:
        k = max(1, int(open_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return (m > 0).astype(np.float32)

def crop_to_bbox(img: Image.Image, mask: np.ndarray, img_size: int, margin: float = 0.07) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return img, None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    pad_x = int(margin * w)
    pad_y = int(margin * h)
    x0 = max(0, x0 - pad_x); x1 = min(mask.shape[1] - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y); y1 = min(mask.shape[0] - 1, y1 + pad_y)
    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    bbox = (x0, y0, x1 + 1, y1 + 1)
    return resize_preserve_aspect(cropped, img_size), bbox

def resize_preserve_aspect(img: Image.Image, target_size: int) -> Image.Image:
    orig_w, orig_h = img.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale); new_h = int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - new_w) // 2; offset_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))
    return canvas


def make_patch_fill(mask: np.ndarray, grid_size: int, patch_size: int) -> np.ndarray:
    return mask.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1,3)).flatten()


# ============================================================================
# Matching helpers
# ============================================================================

def match_patches_by_attention(attention_mapA: np.ndarray, attention_mapB: np.ndarray, patchesA: torch.Tensor, patchesB: torch.Tensor, maskA: np.ndarray, maskB: np.ndarray, grid_size: int, patch_size: int, top_k: int = 100, fg_thresh: float = 0.15, border: int = 2):
    patch_attn_A = attention_mapA.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1,3)).flatten()
    patch_attn_B = attention_mapB.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1,3)).flatten()
    fillA = make_patch_fill(maskA, grid_size, patch_size)
    fillB = make_patch_fill(maskB, grid_size, patch_size)
    validA = fillA > fg_thresh; validB = fillB > fg_thresh
    interior = np.ones(len(patch_attn_A), dtype=bool)
    for i in range(len(patch_attn_A)):
        r, c = divmod(i, grid_size)
        if r < border or r >= grid_size - border or c < border or c >= grid_size - border:
            interior[i] = False
    valid_mask = interior & validA
    attention_scores_A = patch_attn_A.copy(); attention_scores_A[~valid_mask] = -np.inf
    ranked_indices_A = np.argsort(-attention_scores_A)
    top_patches_A = [idx for idx in ranked_indices_A if valid_mask[idx]][:top_k]
    with torch.no_grad():
        similarities = patchesA @ patchesB.T
    matched_patches_B, match_scores = [], []
    for idx_A in top_patches_A:
        sims_to_B = similarities[idx_A].cpu().numpy()
        sims_to_B[~validB] = -np.inf
        best_match_idx = int(np.argmax(sims_to_B)); best_score = float(sims_to_B[best_match_idx])
        matched_patches_B.append(best_match_idx); match_scores.append(best_score)
    return np.array(top_patches_A), np.array(matched_patches_B), np.array(match_scores)


def _patch_centers(indices, grid, p):
    ys = (indices // grid) * p + p // 2
    xs = (indices %  grid) * p + p // 2
    return np.stack([xs, ys], axis=1).astype(np.float32)


def demo_sparse_matches(patchesA, patchesB, validA, validB, grid_size, patch_size, top_n=200, ratio=0.9, mutual=True, ransac=False, ransac_thresh=3.0, border=2):
    with torch.no_grad():
        sims = (patchesA @ patchesB.T).cpu().numpy()

    # add border suppression on both A and B
    N = sims.shape[0]; M = sims.shape[1]
    interiorA = np.ones(N, dtype=bool); interiorB = np.ones(M, dtype=bool)
    for i in range(N):
        r, c = divmod(i, grid_size)
        if r < border or r >= grid_size - border or c < border or c >= grid_size - border:
            interiorA[i] = False
    for j in range(M):
        r, c = divmod(j, grid_size)
        if r < border or r >= grid_size - border or c < border or c >= grid_size - border:
            interiorB[j] = False

    validA = validA & interiorA
    validB = validB & interiorB

    sims[~validA, :] = -np.inf
    sims[:, ~validB] = -np.inf

    idxB_for_A = np.argmax(sims, axis=1)
    s1 = sims[np.arange(sims.shape[0]), idxB_for_A]

    neg = -sims
    part = np.partition(neg, 1, axis=1)
    s1_all = -part[:, 0]
    s2_all = -part[:, 1]
    ratio_mask = s1_all >= (ratio * s2_all)

    if mutual:
        idxA_for_B = np.argmax(sims, axis=0)
        mutual_mask = (np.arange(sims.shape[0]) == idxA_for_B[idxB_for_A])
    else:
        mutual_mask = np.ones_like(ratio_mask, dtype=bool)

    keep_mask = validA & ratio_mask & mutual_mask & np.isfinite(s1)
    order = np.argsort(-s1)
    keep_idx = [i for i in order if keep_mask[i]][:top_n]

    idxA = np.array(keep_idx, dtype=int)
    idxB = idxB_for_A[idxA]
    scores = s1[idxA]

    if ransac and len(idxA) >= 3:
        ptsA = _patch_centers(idxA, grid_size, patch_size)
        ptsB = _patch_centers(idxB, grid_size, patch_size)
        M, inliers = cv2.estimateAffinePartial2D(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if inliers is not None:
            inliers = inliers.ravel().astype(bool)
            idxA, idxB, scores = idxA[inliers], idxB[inliers], scores[inliers]
            print(f"Demo flavor: RANSAC kept {inliers.sum()}/{len(inliers)} matches")
    return idxA, idxB, scores


def attention_weighted_similarity(patchesA, patchesB, attention_A, attention_B, maskA, maskB, grid_size, patch_size, fg_thresh=0.15, border=2):
    def patch_fill(mask):
        return mask.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1,3)).flatten()
    fillA = patch_fill(maskA); fillB = patch_fill(maskB)
    validA = fillA > fg_thresh; validB = fillB > fg_thresh
    interior = np.ones(grid_size * grid_size, dtype=bool)
    for i in range(grid_size * grid_size):
        r, c = divmod(i, grid_size)
        if r < border or r >= grid_size - border or c < border or c >= grid_size - border:
            interior[i] = False
    validA = validA & interior

    with torch.no_grad():
        sims = (patchesA @ patchesB.T).cpu().numpy()
    sims[:, ~validB] = -np.inf
    bestB = np.argmax(sims, axis=1)
    s_i = sims[np.arange(sims.shape[0]), bestB]

    def patch_attn(attn_map):
        return attn_map.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1,3)).flatten().astype(np.float32)
    A_attn = patch_attn(attention_A); B_attn = patch_attn(attention_B)
    w_i = np.sqrt(np.clip(A_attn, 0, 1) * np.clip(B_attn[bestB], 0, 1))
    keep = validA & np.isfinite(s_i) & (w_i > 0)
    if not np.any(keep):
        return 0.0, 1.0, {"kept": 0}
    s_kept = s_i[keep]; w_kept = w_i[keep]
    S_aw = float((s_kept * w_kept).sum() / (w_kept.sum() + 1e-8))
    D_aw = 1.0 - S_aw
    d_kept = 1.0 - s_kept
    order = np.argsort(d_kept); cumw = np.cumsum(w_kept[order])
    p95_idx = np.searchsorted(cumw, 0.95 * cumw[-1])
    d95w = float(d_kept[order[min(p95_idx, len(order)-1)]])
    return S_aw, D_aw, {"kept": int(keep.sum()), "S_aw": S_aw, "D_aw": D_aw, "wP95": d95w, "mean_d": float(d_kept.mean())}


def weighted_percentile(values, weights, q):
    if len(values) == 0:
        return np.nan
    v = np.asarray(values); w = np.asarray(weights)
    order = np.argsort(v); v, w = v[order], w[order]
    cum_w = np.cumsum(w); cut = q * cum_w[-1]
    idx = np.searchsorted(cum_w, cut); idx = min(idx, len(v)-1)
    return float(v[idx])


def foreground_patch_mask(mask, grid_size, patch_size, fg_thresh=0.15, border=2):
    fill = mask.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1,3)).flatten()
    valid = fill > fg_thresh
    interior = np.ones_like(valid, dtype=bool)
    for i in range(len(valid)):
        r, c = divmod(i, grid_size)
        if r < border or r >= grid_size - border or c < border or c >= grid_size - border:
            interior[i] = False
    return valid & interior

def extract_patch_tokens_single_layer(image: Image.Image, model, processor,
                                      patch_size: int, img_size: int,
                                      layer_idx: int, device: str) -> torch.Tensor:
    # Same behavior as your current single-layer extractor, factored for reuse
    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size))
    grid_size = img_size // patch_size
    num_patches = grid_size * grid_size
    inputs = processor(image)
    with torch.no_grad():
        if hasattr(model, 'get_intermediate_layers') and layer_idx is not None:
            try:
                feats = model.get_intermediate_layers(inputs, n=[layer_idx], return_class_token=False)
                if isinstance(feats, (list, tuple)):
                    feats = feats[0]
                patch_tokens = feats[0]  # [N, D]
            except:
                features = model.forward_features(inputs)
                if isinstance(features, dict):
                    if 'x_norm_patchtokens' in features:
                        patch_tokens = features['x_norm_patchtokens'][0]
                    else:
                        seq = features['x']
                        patch_tokens = seq[0, 1:, :] if seq.shape[1] == num_patches + 1 else seq[0, -num_patches:, :]
                else:
                    patch_tokens = features[0, 1:, :] if features.shape[1] == num_patches + 1 else features[0, -num_patches:, :]
        else:
            features = model.forward_features(inputs)
            if isinstance(features, dict):
                if 'x_norm_patchtokens' in features:
                    patch_tokens = features['x_norm_patchtokens'][0]
                else:
                    seq = features['x']
                    patch_tokens = seq[0, 1:, :] if seq.shape[1] == num_patches + 1 else seq[0, -num_patches:, :]
            else:
                patch_tokens = features[0, 1:, :] if features.shape[1] == num_patches + 1 else features[0, -num_patches:, :]

    if patch_tokens.shape[0] != num_patches:
        raise RuntimeError(f"Token count mismatch: got {patch_tokens.shape[0]}, expected {num_patches}")
    return F.normalize(patch_tokens, p=2, dim=-1)  # [N, D], unit-norm

def extract_tokens_multi(image: Image.Image, model, processor,
                         patch_size: int, img_size: int,
                         layers: list[int], device: str) -> list[torch.Tensor]:
    """
    Returns a list of L tensors, each [N, D] unit-normalized patch tokens for the requested layers.
    Supports negative indices (e.g., -1 for last block).
    """
    # Resolve negative indices against model.blocks
    if hasattr(model, 'blocks'):
        L = len(model.blocks)
        resolved = []
        for li in layers:
            if li < 0:
                resolved.append(L + li)
            else:
                resolved.append(li)
    else:
        resolved = layers

    tokens_list = []
    for li in resolved:
        tokens = extract_patch_tokens_single_layer(image, model, processor, patch_size, img_size, li, device)
        tokens_list.append(tokens)
    return tokens_list  # list of [N, D]

def fuse_tokens(tokens_list: list[torch.Tensor], method: str = 'mean') -> torch.Tensor:
    """
    Fuse multiple per-layer token sets into one [N, D] matrix.
    Each input is already unit-normalized. We average/median elementwise, then re-normalize.
    """
    if len(tokens_list) == 1:
        return tokens_list[0]
    X = torch.stack(tokens_list, dim=0)  # [L, N, D]
    if method == 'median':
        fused = torch.median(X, dim=0).values
    else:
        fused = torch.mean(X, dim=0)
    fused = F.normalize(fused, p=2, dim=-1)
    return fused  # [N, D]


def layerwise_similarities(tokensA_list: list[torch.Tensor], tokensB_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Compute cosine sim matrices per layer: returns list of [N, N] tensors.
    Assumes each tokens*_list[l] is [N, D] and unit-normalized.
    """
    sims_per_layer = []
    with torch.no_grad():
        for A, B in zip(tokensA_list, tokensB_list):
            sims_per_layer.append(A @ B.T)
    return sims_per_layer  # list of [N, N]

def fuse_similarities(sims_per_layer: list[torch.Tensor], method: str = 'mean') -> torch.Tensor:
    """
    Late fusion at similarity level. Returns a single [N, N] tensor.
    """
    if len(sims_per_layer) == 1:
        return sims_per_layer[0]
    S = torch.stack(sims_per_layer, dim=0)  # [L, N, N]
    if method == 'median':
        fused = torch.median(S, dim=0).values
    else:
        fused = torch.mean(S, dim=0)
    return fused


def export_comprehensive_dissimilarities(sims: np.ndarray, 
                                         validA: np.ndarray, 
                                         validB: np.ndarray,
                                         grid_size: int,
                                         output_dir: str,
                                         fillA: np.ndarray = None,
                                         fillB: np.ndarray = None):
    """
    Export complete dissimilarity data with bidirectional mappings.
    
    Args:
        sims: [N, M] similarity matrix (patchesA @ patchesB.T)
        validA: [N] boolean mask for valid patches in A
        validB: [M] boolean mask for valid patches in B
        grid_size: Grid size for computing spatial coordinates
        output_dir: Where to save outputs
        fillA, fillB: Optional foreground fill ratios
    
    Exports:
        1. foreground_dissimilarities_A_to_B.csv - A→B matches (foreground only)
        2. foreground_dissimilarities_B_to_A.csv - B→A matches (foreground only)
        3. bidirectional_mutual_matches.csv - Mutual best matches
        4. dissimilarity_matrix_full.npy - Full matrix for all-by-all analysis
    """
    import csv
    
    N, M = sims.shape
    
    # Compute best matches: A→B
    best_B_for_A = np.argmax(sims, axis=1)  # [N]
    sim_A_to_B = sims[np.arange(N), best_B_for_A]  # [N]
    dissim_A_to_B = 1.0 - sim_A_to_B
    
    # Compute best matches: B→A
    best_A_for_B = np.argmax(sims, axis=0)  # [M]
    sim_B_to_A = sims[best_A_for_B, np.arange(M)]  # [M]
    dissim_B_to_A = 1.0 - sim_B_to_A
    
    # Check bidirectional consistency (mutual best matches)
    mutual_AB = (best_A_for_B[best_B_for_A] == np.arange(N))  # [N] bool
    
    # Helper: get spatial coords for a patch index
    def get_coords(idx, grid):
        return (idx % grid, idx // grid)
    
    # ========================================================================
    # 1. Export A→B (foreground patches in A)
    # ========================================================================
    csv_A_to_B = os.path.join(output_dir, 'foreground_dissimilarities_A_to_B.csv')
    with open(csv_A_to_B, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ['patch_idx_A', 'x_A', 'y_A', 
                 'best_match_idx_B', 'x_B', 'y_B',
                 'cosine_sim', 'dissimilarity', 
                 'is_mutual_best', 'is_foreground_B']
        
        if fillA is not None:
            header.insert(3, 'fill_ratio_A')
        if fillB is not None:
            header.insert(-2, 'fill_ratio_B')
        
        writer.writerow(header)
        
        for i in range(N):
            if not validA[i]:
                continue  # Skip background patches in A
            
            j = int(best_B_for_A[i])
            x_A, y_A = get_coords(i, grid_size)
            x_B, y_B = get_coords(j, grid_size)
            
            row = [
                i, x_A, y_A,
            ]
            
            if fillA is not None:
                row.append(float(fillA[i]))
            
            row.extend([
                j, x_B, y_B,
                float(sim_A_to_B[i]),
                float(dissim_A_to_B[i]),
                bool(mutual_AB[i]),
                bool(validB[j])
            ])
            
            if fillB is not None:
                row.insert(-2, float(fillB[j]))
            
            writer.writerow(row)
    
    print(f"  Saved: foreground_dissimilarities_A_to_B.csv ({validA.sum()} patches)")
    
    # ========================================================================
    # 2. Export B→A (foreground patches in B)
    # ========================================================================
    csv_B_to_A = os.path.join(output_dir, 'foreground_dissimilarities_B_to_A.csv')
    with open(csv_B_to_A, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ['patch_idx_B', 'x_B', 'y_B',
                 'best_match_idx_A', 'x_A', 'y_A',
                 'cosine_sim', 'dissimilarity',
                 'is_mutual_best', 'is_foreground_A']
        
        if fillB is not None:
            header.insert(3, 'fill_ratio_B')
        if fillA is not None:
            header.insert(-2, 'fill_ratio_A')
        
        writer.writerow(header)
        
        for j in range(M):
            if not validB[j]:
                continue  # Skip background patches in B
            
            i = int(best_A_for_B[j])
            x_B, y_B = get_coords(j, grid_size)
            x_A, y_A = get_coords(i, grid_size)
            
            # Check if this is a mutual best match
            is_mutual = (best_B_for_A[i] == j)
            
            row = [
                j, x_B, y_B,
            ]
            
            if fillB is not None:
                row.append(float(fillB[j]))
            
            row.extend([
                i, x_A, y_A,
                float(sim_B_to_A[j]),
                float(dissim_B_to_A[j]),
                bool(is_mutual),
                bool(validA[i])
            ])
            
            if fillA is not None:
                row.insert(-2, float(fillA[i]))
            
            writer.writerow(row)
    
    print(f"  Saved: foreground_dissimilarities_B_to_A.csv ({validB.sum()} patches)")
    
    # ========================================================================
    # 3. Export mutual best matches only
    # ========================================================================
    mutual_csv = os.path.join(output_dir, 'bidirectional_mutual_matches.csv')
    with open(mutual_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'patch_idx_A', 'x_A', 'y_A',
            'patch_idx_B', 'x_B', 'y_B',
            'cosine_sim', 'dissimilarity',
            'both_foreground'
        ])
        
        for i in range(N):
            if not mutual_AB[i]:
                continue
            
            j = int(best_B_for_A[i])
            x_A, y_A = get_coords(i, grid_size)
            x_B, y_B = get_coords(j, grid_size)
            
            writer.writerow([
                i, x_A, y_A,
                j, x_B, y_B,
                float(sim_A_to_B[i]),
                float(dissim_A_to_B[i]),
                bool(validA[i] and validB[j])
            ])
    
    mutual_count = mutual_AB.sum()
    mutual_fg_count = (mutual_AB & validA & validB[best_B_for_A]).sum()
    print(f"  Saved: bidirectional_mutual_matches.csv ({mutual_count} total, {mutual_fg_count} foreground)")
    
    # ========================================================================
    # 4. Save full dissimilarity matrix for all-by-all analysis
    # ========================================================================
    dissim_matrix = 1.0 - sims
    np.save(os.path.join(output_dir, 'dissimilarity_matrix_full.npy'), dissim_matrix)
    print(f"  Saved: dissimilarity_matrix_full.npy [{N}x{M}]")
    
    # Also save metadata about which patches are valid
    metadata = {
        'shape': [N, M],
        'grid_size': grid_size,
        'validA_indices': np.where(validA)[0].tolist(),
        'validB_indices': np.where(validB)[0].tolist(),
        'mutual_match_count': int(mutual_count),
        'mutual_foreground_count': int(mutual_fg_count)
    }
    
    with open(os.path.join(output_dir, 'dissimilarity_matrix_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: dissimilarity_matrix_metadata.json")
    
    return {
        'A_to_B': dissim_A_to_B,
        'B_to_A': dissim_B_to_A,
        'mutual_mask': mutual_AB,
        'full_matrix': dissim_matrix
    }

"""
TPS Homology Grid Overlay (CORRECT APPROACH)
Warps a regular grid (not images!) to specimens for anatomical correspondence

The grid is defined in consensus space and warped to each specimen.
Original images remain unmodified - only the grid deforms!
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf
from typing import Tuple, Dict, List, Optional
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ============================================================================
# Procrustes for Consensus Shape (unchanged)
# ============================================================================

def procrustes_align(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Align source to target using Procrustes (similarity transform)."""
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    source_scale = np.sqrt((source_centered ** 2).sum())
    target_scale = np.sqrt((target_centered ** 2).sum())
    
    source_normalized = source_centered / (source_scale + 1e-8)
    target_normalized = target_centered / (target_scale + 1e-8)
    
    H = source_normalized.T @ target_normalized
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    aligned = (source_normalized @ R) * target_scale + target_center
    
    return aligned


def compute_consensus_shape(keypoints_list: List[np.ndarray], max_iters: int = 10) -> np.ndarray:
    """Compute mean shape using iterative Procrustes alignment."""
    consensus = keypoints_list[0].copy()
    
    for iteration in range(max_iters):
        aligned_all = []
        for kpts in keypoints_list:
            aligned = procrustes_align(kpts, consensus)
            aligned_all.append(aligned)
        
        new_consensus = np.mean(aligned_all, axis=0)
        
        if np.allclose(consensus, new_consensus, atol=1e-4):
            break
        
        consensus = new_consensus
    
    return consensus


# ============================================================================
# TPS Grid Warper (NEW - Warps grid, not images!)
# ============================================================================

class TPSGridWarper:
    """
    Warp a regular grid from consensus space to specimen space.
    Images remain ORIGINAL - only the grid deforms!
    """
    
    def __init__(self, consensus_keypoints: np.ndarray, 
                 specimen_keypoints: np.ndarray):
        """
        Args:
            consensus_keypoints: [N, 2] keypoints in consensus space
            specimen_keypoints: [N, 2] keypoints in this specimen's space
        """
        self.consensus_kpts = consensus_keypoints
        self.specimen_kpts = specimen_keypoints
        
        # Fit TPS: consensus → specimen (inverse of what I did before!)
        # This maps grid points FROM consensus TO specimen
        self.rbf_x = Rbf(
            consensus_keypoints[:, 0], 
            consensus_keypoints[:, 1], 
            specimen_keypoints[:, 0], 
            function='thin_plate', 
            smooth=0.0
        )
        self.rbf_y = Rbf(
            consensus_keypoints[:, 0], 
            consensus_keypoints[:, 1], 
            specimen_keypoints[:, 1], 
            function='thin_plate', 
            smooth=0.0
        )
    
    def warp_points(self, points_in_consensus: np.ndarray) -> np.ndarray:
        """
        Warp points from consensus space to specimen space.
        
        Args:
            points_in_consensus: [M, 2] points in consensus coordinate system
        
        Returns:
            points_in_specimen: [M, 2] same points in specimen coordinates
        """
        x_specimen = self.rbf_x(points_in_consensus[:, 0], points_in_consensus[:, 1])
        y_specimen = self.rbf_y(points_in_consensus[:, 0], points_in_consensus[:, 1])
        
        return np.stack([x_specimen, y_specimen], axis=1)


# ============================================================================
# Regular Grid in Consensus Space
# ============================================================================

def create_consensus_grid(consensus_keypoints: np.ndarray,
                          cell_size: Optional[int] = 16,
                          margin: float = 0.1,
                          rows: Optional[int] = None,
                          cols: Optional[int] = None) -> Dict:
    """
    Create regular grid in consensus space.

    You may specify either a cell_size in pixels OR rows/cols.
    If rows is set and cols is None, cols=rows.
    """
    if consensus_keypoints.ndim != 2 or consensus_keypoints.shape[1] != 2:
        raise ValueError(f"consensus_keypoints must be [N, 2], got {consensus_keypoints.shape}")

    # Bounding box around consensus shape, with margin
    mins = consensus_keypoints.min(axis=0)
    maxs = consensus_keypoints.max(axis=0)
    xmin, ymin = float(mins[0]), float(mins[1])
    xmax, ymax = float(maxs[0]), float(maxs[1])
    w, h = xmax - xmin, ymax - ymin
    xmin -= margin * w; xmax += margin * w
    ymin -= margin * h; ymax += margin * h
    w = xmax - xmin; h = ymax - ymin

    # Resolve rows/cols vs cell_size
    if rows is not None:
        if cols is None:
            cols = rows
        # +1 lines = rows+1 horizontals, cols+1 verticals
        n_rows = int(rows)
        n_cols = int(cols)
        x_lines = np.linspace(xmin, xmax, n_cols + 1)
        y_lines = np.linspace(ymin, ymax, n_rows + 1)
        eff_cell_w = w / n_cols
        eff_cell_h = h / n_rows
        eff_cell = 0.5 * (eff_cell_w + eff_cell_h)
    else:
        if cell_size is None or cell_size <= 0:
            cell_size = 16
        n_cols = max(1, int(np.ceil(w / cell_size)))
        n_rows = max(1, int(np.ceil(h / cell_size)))
        x_lines = np.linspace(xmin, xmax, n_cols + 1)
        y_lines = np.linspace(ymin, ymax, n_rows + 1)
        eff_cell = float(cell_size)

    return {
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'x_lines': x_lines, 'y_lines': y_lines,
        'n_rows': n_rows, 'n_cols': n_cols,
        'cell_size': eff_cell
    }
    return grid_info


def get_grid_cell_corners(grid_info: Dict, row: int, col: int) -> np.ndarray:
    """
    Get 4 corner points of a grid cell in consensus space.
    
    Returns:
        corners: [4, 2] array of corner points [TL, TR, BR, BL]
    """
    x_lines = grid_info['x_lines']
    y_lines = grid_info['y_lines']
    
    x0, x1 = x_lines[col], x_lines[col + 1]
    y0, y1 = y_lines[row], y_lines[row + 1]
    
    corners = np.array([
        [x0, y0],  # Top-left
        [x1, y0],  # Top-right
        [x1, y1],  # Bottom-right
        [x0, y1]   # Bottom-left
    ])
    
    return corners


def get_grid_cell_center(grid_info: Dict, row: int, col: int) -> np.ndarray:
    """Get center point of grid cell in consensus space."""
    x_lines = grid_info['x_lines']
    y_lines = grid_info['y_lines']
    
    x_center = (x_lines[col] + x_lines[col + 1]) / 2
    y_center = (y_lines[row] + y_lines[row + 1]) / 2
    
    return np.array([x_center, y_center])


# ============================================================================
# Warp Grid to Specimen
# ============================================================================

def warp_grid_to_specimen(grid_info: Dict, 
                         tps_warper: TPSGridWarper) -> Dict:
    """
    Warp grid from consensus space to specimen space.
    
    Args:
        grid_info: Grid definition in consensus space
        tps_warper: TPS warper (consensus → specimen)
    
    Returns:
        warped_grid_info: Grid in specimen space with warped coordinates
    """
    n_rows = grid_info['n_rows']
    n_cols = grid_info['n_cols']
    
    # Warp all grid line intersections
    x_lines_consensus = grid_info['x_lines']
    y_lines_consensus = grid_info['y_lines']
    
    # Create mesh of intersection points
    xx, yy = np.meshgrid(x_lines_consensus, y_lines_consensus)
    intersection_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Warp to specimen space
    warped_intersections = tps_warper.warp_points(intersection_points)
    
    # Reshape back to grid
    warped_xx = warped_intersections[:, 0].reshape(len(y_lines_consensus), len(x_lines_consensus))
    warped_yy = warped_intersections[:, 1].reshape(len(y_lines_consensus), len(x_lines_consensus))
    
    warped_grid_info = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'warped_xx': warped_xx,  # [n_rows+1, n_cols+1]
        'warped_yy': warped_yy,  # [n_rows+1, n_cols+1]
        'cell_size': grid_info['cell_size']
    }
    
    return warped_grid_info


def get_warped_cell_center(warped_grid_info: Dict, row: int, col: int) -> np.ndarray:
    """Get center of warped grid cell in specimen space."""
    xx = warped_grid_info['warped_xx']
    yy = warped_grid_info['warped_yy']
    
    # Cell corners
    x_corners = [xx[row, col], xx[row, col+1], xx[row+1, col+1], xx[row+1, col]]
    y_corners = [yy[row, col], yy[row, col+1], yy[row+1, col+1], yy[row+1, col]]
    
    # Center is mean of corners
    x_center = np.mean(x_corners)
    y_center = np.mean(y_corners)
    
    return np.array([x_center, y_center])


# ============================================================================
# Map DINOv3 Patches to Grid Cells
# ============================================================================

def map_patches_to_warped_grid(patch_centers: np.ndarray,
                               warped_grid_info: Dict,
                               img_size: int,
                               method: str = 'nearest_center') -> Dict[int, Tuple[int, int]]:
    """
    Map each DINOv3 patch to a grid cell using warped grid.
    
    Args:
        patch_centers: [N_patches, 2] DINOv3 patch center coordinates
        warped_grid_info: Warped grid in specimen space
        img_size: Image size (for bounds checking)
        method: 'nearest_center' or 'overlap'
    
    Returns:
        patch_to_cell: {patch_idx: (row, col)} mapping
    """
    n_rows = warped_grid_info['n_rows']
    n_cols = warped_grid_info['n_cols']
    
    # Get all warped cell centers
    cell_centers = []
    cell_ids = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            center = get_warped_cell_center(warped_grid_info, row, col)
            
            # Check if center is within image bounds
            if 0 <= center[0] < img_size and 0 <= center[1] < img_size:
                cell_centers.append(center)
                cell_ids.append((row, col))
    
    if len(cell_centers) == 0:
        return {}
    
    cell_centers = np.array(cell_centers)
    
    # Find nearest cell for each patch
    if method == 'nearest_center':
        distances = cdist(patch_centers, cell_centers)
        nearest_cell_idx = distances.argmin(axis=1)
        
        patch_to_cell = {}
        for patch_idx, cell_idx in enumerate(nearest_cell_idx):
            patch_to_cell[patch_idx] = cell_ids[cell_idx]
    
    else:
        raise NotImplementedError("Overlap method not yet implemented")
    
    return patch_to_cell


# ============================================================================
# Homologous Comparison
# ============================================================================

def compute_homologous_dissimilarity(
    patchesA: torch.Tensor,
    patchesB: torch.Tensor,
    patch_to_cell_A: Dict[int, Tuple[int, int]],
    patch_to_cell_B: Dict[int, Tuple[int, int]],
    maskA: Optional[np.ndarray] = None,
    maskB: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute dissimilarity between specimens using homologous grid cells.
    
    Args:
        patchesA, patchesB: [N, D] patch embeddings
        patch_to_cell_A, patch_to_cell_B: {patch_idx: (row, col)} mappings
        maskA, maskB: Optional foreground masks for patches
    
    Returns:
        dissimilarity_dict: Per-cell and aggregate dissimilarities
    """
    # Build reverse mapping: cell → patches
    cell_to_patches_A = {}
    for patch_idx, cell in patch_to_cell_A.items():
        if maskA is None or maskA[patch_idx]:
            if cell not in cell_to_patches_A:
                cell_to_patches_A[cell] = []
            cell_to_patches_A[cell].append(patch_idx)
    
    cell_to_patches_B = {}
    for patch_idx, cell in patch_to_cell_B.items():
        if maskB is None or maskB[patch_idx]:
            if cell not in cell_to_patches_B:
                cell_to_patches_B[cell] = []
            cell_to_patches_B[cell].append(patch_idx)
    
    # Find cells present in both specimens
    common_cells = set(cell_to_patches_A.keys()) & set(cell_to_patches_B.keys())
    
    if len(common_cells) == 0:
        print("Warning: No common grid cells between specimens!")
        return {'error': 'no_common_cells'}
    
    # Compute dissimilarity for each common cell
    per_cell_dissim = {}
    dissimilarities = []
    
    for cell in common_cells:
        patches_A = cell_to_patches_A[cell]
        patches_B = cell_to_patches_B[cell]
        
        # Average embeddings at this cell
        emb_A = patchesA[patches_A].mean(dim=0)
        emb_B = patchesB[patches_B].mean(dim=0)
        
        # Cosine dissimilarity
        similarity = F.cosine_similarity(emb_A.unsqueeze(0), emb_B.unsqueeze(0)).item()
        dissimilarity = 1.0 - similarity
        
        per_cell_dissim[cell] = {
            'dissimilarity': dissimilarity,
            'n_patches_A': len(patches_A),
            'n_patches_B': len(patches_B)
        }
        dissimilarities.append(dissimilarity)
    
    dissim_array = np.array(dissimilarities)
    
    return {
        'mean': float(dissim_array.mean()),
        'median': float(np.median(dissim_array)),
        'std': float(dissim_array.std()),
        'min': float(dissim_array.min()),
        'max': float(dissim_array.max()),
        'p95': float(np.percentile(dissim_array, 95)),
        'p99': float(np.percentile(dissim_array, 99)),
        'num_cells': len(common_cells),
        'per_cell': per_cell_dissim
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_warped_grid_overlay(image: np.ndarray,
                                  warped_grid_info: Dict,
                                  patch_centers: np.ndarray,
                                  patch_to_cell: Dict,
                                  keypoints: Optional[np.ndarray] = None,  # ADD THIS
                                  title: str = "Warped Grid Overlay") -> np.ndarray:
    """
    Visualize warped grid overlaid on original image.
    
    Returns:
        vis_image: Image with grid overlay
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    
    # Draw warped grid lines
    xx = warped_grid_info['warped_xx']
    yy = warped_grid_info['warped_yy']
    n_rows = warped_grid_info['n_rows']
    n_cols = warped_grid_info['n_cols']
    
    # Horizontal lines
    for row in range(n_rows + 1):
        points = []
        for col in range(n_cols + 1):
            x, y = xx[row, col], yy[row, col]
            if 0 <= x < w and 0 <= y < h:
                points.append((int(x), int(y)))
        
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(vis, points[i], points[i+1], (0, 255, 0), 1)
    
    # Vertical lines
    for col in range(n_cols + 1):
        points = []
        for row in range(n_rows + 1):
            x, y = xx[row, col], yy[row, col]
            if 0 <= x < w and 0 <= y < h:
                points.append((int(x), int(y)))
        
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(vis, points[i], points[i+1], (0, 255, 0), 1)
    
    # NEW: Draw keypoints if provided (for alignment verification)
    if keypoints is not None:
        for i, (x, y) in enumerate(keypoints):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis, (int(x), int(y)), 6, (255, 0, 255), -1)  # Magenta
                cv2.circle(vis, (int(x), int(y)), 8, (255, 255, 255), 2)  # White border
                cv2.putText(vis, str(i+1), (int(x)+10, int(y)+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw patch centers mapped to cells
    for patch_idx, cell in patch_to_cell.items():
        px, py = patch_centers[patch_idx]
        cv2.circle(vis, (int(px), int(py)), 2, (255, 0, 0), -1)
    
    # Draw cell labels
    for row in range(n_rows):
        for col in range(n_cols):
            center = get_warped_cell_center(warped_grid_info, row, col)
            cx, cy = int(center[0]), int(center[1])
            
            if 0 <= cx < w and 0 <= cy < h:
                cv2.putText(vis, f"{row},{col}", (cx-10, cy+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    return vis

def visualize_homologous_comparison(imageA: np.ndarray,
                                   imageB: np.ndarray,
                                   warped_grid_A: Dict,
                                   warped_grid_B: Dict,
                                   patch_centers_A: np.ndarray,
                                   patch_centers_B: np.ndarray,
                                   patch_to_cell_A: Dict,
                                   patch_to_cell_B: Dict,
                                   per_cell_dissim: Dict,
                                   output_path: str,  # MOVED BEFORE OPTIONAL ARGS
                                   keypointsA: Optional[np.ndarray] = None,
                                   keypointsB: Optional[np.ndarray] = None):
    """Create comprehensive visualization of homologous comparison."""
    
    # Create overlays with keypoints
    visA = visualize_warped_grid_overlay(imageA, warped_grid_A, 
                                        patch_centers_A, patch_to_cell_A,
                                        keypoints=keypointsA, title="Specimen A")
    visB = visualize_warped_grid_overlay(imageB, warped_grid_B,
                                        patch_centers_B, patch_to_cell_B,
                                        keypoints=keypointsB, title="Specimen B")
    
    # Create dissimilarity heatmap
    n_rows = warped_grid_A['n_rows']
    n_cols = warped_grid_A['n_cols']
    dissim_matrix = np.full((n_rows, n_cols), np.nan)
    
    for cell, info in per_cell_dissim.items():
        row, col = cell
        dissim_matrix[row, col] = info['dissimilarity']
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(cv2.cvtColor(visA, cv2.COLOR_BGR2RGB))
    ax1.set_title('Specimen A\n(Original + Warped Grid)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(visB, cv2.COLOR_BGR2RGB))
    ax2.set_title('Specimen B\n(Original + Warped Grid)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    im = ax3.imshow(dissim_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax3.set_title('Homologous Dissimilarity\n(Per Grid Cell)', fontsize=12, fontweight='bold')
    
    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(dissim_matrix[i, j]):
                text = f'{dissim_matrix[i, j]:.2f}'
                ax3.text(j, i, text, ha="center", va="center", 
                        color="black", fontsize=8)
    
    ax3.set_xlabel('Grid Column')
    ax3.set_ylabel('Grid Row')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization: {output_path}")


# ============================================================================
# Complete Pipeline
# ============================================================================

def tps_homology_pipeline(
    imgA: Image.Image,
    imgB: Image.Image,
    keypointsA: np.ndarray,
    keypointsB: np.ndarray,
    patchesA: torch.Tensor,
    patchesB: torch.Tensor,
    img_size: int,
    patch_size: int,
    cell_size: int = 12,
    consensus_shape: Optional[np.ndarray] = None,
    maskA: Optional[np.ndarray] = None,
    maskB: Optional[np.ndarray] = None,
    output_dir: str = '.'
) -> Dict:
    """
    Complete TPS homology pipeline with grid warping.
    
    Images stay ORIGINAL - only grid is warped!
    """
    
    print("\n" + "="*60)
    print("TPS HOMOLOGY PIPELINE (Grid Warping)")
    print("="*60)
    
    # Step 1: Compute consensus shape
    if consensus_shape is None:
        print("Computing consensus shape...")
        consensus_shape = compute_consensus_shape([keypointsA, keypointsB])
    
    # Step 2: Create regular grid in consensus space
    # Optionally snap cell size to ViT patch lattice
    chosen_cell_size = int(cell_size)
    if grid_snap and patch_size is not None and patch_size > 0:
        if chosen_cell_size % patch_size != 0:
            k = max(1, round(chosen_cell_size / patch_size))
            chosen_cell_size = int(k * patch_size)
            print(f"Snapped TPS cell size to {chosen_cell_size} (k={k} × patch_size {patch_size})")

    if tps_rows is not None:
        print(f"Creating grid with rows={tps_rows}, cols={tps_cols or tps_rows} in consensus space...")
        grid_info = create_consensus_grid(consensus_shape, rows=tps_rows, cols=tps_cols, margin=0.1)
    else:
        print(f"Creating {chosen_cell_size}×{chosen_cell_size} px grid in consensus space...")
        grid_info = create_consensus_grid(consensus_shape, cell_size=chosen_cell_size, margin=0.1)
    print(f"  Grid: {grid_info['n_rows']}×{grid_info['n_cols']} cells")

    
    # Step 3: Create TPS warpers (consensus → specimen)
    print("Fitting TPS transformations...")
    warperA = TPSGridWarper(consensus_shape, keypointsA)
    warperB = TPSGridWarper(consensus_shape, keypointsB)
    
    # Step 4: Warp grid to each specimen
    print("Warping grid to specimens...")
    warped_grid_A = warp_grid_to_specimen(grid_info, warperA)
    warped_grid_B = warp_grid_to_specimen(grid_info, warperB)
    
    # Step 5: Compute DINOv3 patch centers
    grid_dinov3 = img_size // patch_size
    patch_centers = []
    for i in range(grid_dinov3):
        for j in range(grid_dinov3):
            x = j * patch_size + patch_size // 2
            y = i * patch_size + patch_size // 2
            patch_centers.append([x, y])
    patch_centers = np.array(patch_centers)
    
    # Step 6: Map patches to warped grid cells
    print("Mapping DINOv3 patches to grid cells...")
    patch_to_cell_A = map_patches_to_warped_grid(
        patch_centers, warped_grid_A, img_size
    )
    patch_to_cell_B = map_patches_to_warped_grid(
        patch_centers, warped_grid_B, img_size
    )
    
    print(f"  Specimen A: {len(patch_to_cell_A)} patches mapped")
    print(f"  Specimen B: {len(patch_to_cell_B)} patches mapped")
    
    # Step 7: Compute homologous dissimilarity
    print("Computing homologous dissimilarities...")
    dissim_stats = compute_homologous_dissimilarity(
        patchesA, patchesB,
        patch_to_cell_A, patch_to_cell_B,
        maskA, maskB
    )
    
    if 'error' not in dissim_stats:
        print(f"\n✓ Homologous Dissimilarity:")
        print(f"    Mean: {dissim_stats['mean']:.4f}")
        print(f"    Median: {dissim_stats['median']:.4f}")
        print(f"    P95: {dissim_stats['p95']:.4f}")
        print(f"    Common cells: {dissim_stats['num_cells']}")
    
    # Step 8: Visualize
    print("Creating visualizations...")
    imgA_np = np.array(imgA.resize((img_size, img_size)))
    imgB_np = np.array(imgB.resize((img_size, img_size)))
    
    visualize_homologous_comparison(
        imgA_np, imgB_np,
        warped_grid_A, warped_grid_B,
        patch_centers, patch_centers,
        patch_to_cell_A, patch_to_cell_B,
        dissim_stats.get('per_cell', {}),
        os.path.join(output_dir, 'tps_homology_comparison.png')
    )
    
    return {
        'dissimilarity': dissim_stats,
        'warped_grid_A': warped_grid_A,
        'warped_grid_B': warped_grid_B,
        'patch_to_cell_A': patch_to_cell_A,
        'patch_to_cell_B': patch_to_cell_B,
        'consensus_shape': consensus_shape,
        'grid_info': grid_info
    }
    
def estimate_affine_from_masks(mask_src: np.ndarray,
                               mask_dst: np.ndarray,
                               max_iter: int = 1000,
                               eps: float = 1e-6) -> Optional[np.ndarray]:
    """
    Zero-shot mask alignment: estimate a 2x3 affine that warps mask_src -> mask_dst
    using ECC on distance-transforms (robust to small holes/boundaries).

    Returns:
        2x3 affine matrix, or None if estimation fails.
    """
    # Ensure uint8 single-channel
    A = (mask_src.astype(np.uint8) * 255)
    B = (mask_dst.astype(np.uint8) * 255)

    # Distance transform -> normalize -> float32
    A_dt = cv2.distanceTransform(255 - A, cv2.DIST_L2, 3)
    B_dt = cv2.distanceTransform(255 - B, cv2.DIST_L2, 3)
    A_dt = cv2.normalize(A_dt, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    B_dt = cv2.normalize(B_dt, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # ECC needs same size
    h, w = B_dt.shape
    if A_dt.shape != (h, w):
        A_dt = cv2.resize(A_dt, (w, h), interpolation=cv2.INTER_LINEAR)

    warp = np.eye(2, 3, dtype=np.float32)  # initial affine
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        cc, warp = cv2.findTransformECC(B_dt, A_dt, warp, cv2.MOTION_AFFINE, criteria)
        return warp
    except cv2.error:
        return None



# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DINOv3 ATTENTION-BASED Patch Matching (layer fusion + edge suppression)')
    parser.add_argument('img1', help='First image path')
    parser.add_argument('img2', help='Second image path')
    parser.add_argument('--dinov3-local-ckpt', type=str, required=True, help='Path to local DINOv3 checkpoint (.pth file)')
    parser.add_argument('--dinov3-arch', type=str, default='vits16', choices=['vits16', 'vitb16', 'vitl16'], help='DINOv3 architecture')
    parser.add_argument('--img-size', type=int, default=518, help='Image size (adjusted to patch_size multiple)')

    # Layer control
    parser.add_argument('--layer', type=int, default=-1, help='Single layer index (negative allowed). Ignored if --layers is set.')
    parser.add_argument('--layers', nargs='+', type=int, default=None, help='Multiple layers to aggregate (e.g., -3 -2 -1). Overrides --layer.')
    parser.add_argument('--layer-agg', type=str, default='mean', choices=['mean', 'median'], help='How to fuse across layers (token & similarity level).')
    parser.add_argument('--report-layer-std', action='store_true', help='Print simple across-layer std of per-patch best similarities.')

    # Attention maps
    parser.add_argument('--attn-layers', nargs='+', type=int, default=[9, 11], help='Layers for attention visualization')

    # Foreground & edge cleanup
    parser.add_argument('--fg-thresh', type=float, default=0.15, help='Foreground patch threshold')
    parser.add_argument('--no-crop', action='store_true', help='Skip cropping')
    parser.add_argument('--border-patches', type=int, default=2, help='Exclude this many patches from each border when forming FG/matches')
    parser.add_argument('--mask-erode-px', type=int, default=0, help='Erode binary masks by this many pixels before patching')
    parser.add_argument('--attn-fg', action='store_true', help='Intersect FG mask with attention FG from matching layer')
    parser.add_argument('--attn-fg-thresh', type=float, default=0.25, help='Threshold for attention FG (0..1)')

    # Output & viz
    parser.add_argument('--output-dir', '--outdir', dest='output_dir', default=None, help='Output directory')
    parser.add_argument('--top-matches', type=int, default=50, help='Number of top attention matches to visualize')

    # Keypoint alignment
    parser.add_argument('--keypoints-json', type=str, default=None, help='COCO JSON with keypoints for both images (optional)')
    parser.add_argument('--align-method', type=str, default='procrustes', choices=['procrustes', 'similarity', 'homography'], help='Keypoint alignment method')
    parser.add_argument('--min-kpts', type=int, default=3, help='Min keypoints to align')

    # Matching flavors
    parser.add_argument('--match-flavor', type=str, default='attention', choices=['attention', 'demo'], help='attention = rank by attention (default); demo = mutual/ratio like notebook')
    parser.add_argument('--demo-topn', type=int, default=200, help='Max matches to keep for demo flavor (after filtering)')
    parser.add_argument('--demo-ratio', type=float, default=0.9, help='Lowe ratio threshold for demo flavor')
    parser.add_argument('--demo-mutual', action='store_true', help='Require mutual nearest for demo flavor')
    parser.add_argument('--demo-verify', action='store_true', help='Geometric verification with RANSAC (affine/similarity)')
    parser.add_argument('--demo-ransac-reproj', type=float, default=3.0, help='RANSAC reprojection threshold (px)')

    # Score/Histogram modes
    parser.add_argument('--score-mode', type=str, default='foreground',
                        choices=['foreground', 'attention_weighted', 'demo_inliers', 'bidirectional', 'topk_mean'],
                        help='Which per-patch scores drive the histogram/statistics.')
    parser.add_argument('--topk', type=int, default=5, help='K for --score-mode topk_mean.')
    parser.add_argument('--hist-min-joint', type=int, default=100, help='Minimum patches required before computing stats.')

    #grid warp args
    parser.add_argument('--matching-mode', type=str, default='similarity',
                       choices=['similarity', 'tps_homology', 'both'],
                       help='Matching strategy: similarity (current default), '
                            'tps_homology (grid warping), or both')
    
    parser.add_argument('--tps-grid-cell-size', type=int, default=16,
                       help='TPS grid cell size in pixels (default: 16, meaning 16×16 px cells). '
                            'Smaller = finer anatomical detail, larger = coarser regions')
    
    parser.add_argument('--consensus-shape-db', type=str, default=None,
                       help='Path to pre-computed consensus shape .npz file (optional). '
                            'If not provided, will compute from current pair')
    
    parser.add_argument('--tps-grid-rows', type=int, default=None,
                        help='Override TPS grid by number of rows (mutually exclusive with --tps-grid-cell-size)')
    parser.add_argument('--tps-grid-cols', type=int, default=None,
                        help='Override TPS grid by number of cols (if rows is set but cols is not, cols=rows)')
    parser.add_argument('--tps-grid-snap', action='store_true',
                        help='Snap TPS grid cell size to ViT patch lattice (multiples of patch_size).')
    # ---- NEW: attention-derived FG mask controls ----
    parser.add_argument('--attn-fg-mode', type=str, default='intersection',
                        choices=['intersection', 'union', 'attn_only'],
                        help='How to combine binary foreground mask with attention mask')
    parser.add_argument('--attn-fg-thresh-mode', type=str, default='percentile',
                        choices=['absolute', 'percentile'],
                        help='Thresholding mode for attention → mask')
    parser.add_argument('--attn-fg-thresh-pct', type=float, default=85.0,
                        help='If percentile mode: keep top P%% attention per image (0–100)')
    parser.add_argument('--attn-fg-morph', type=str, default='none',
                        choices=['none', 'open', 'close', 'openclose'],
                        help='Optional morphology to clean attention FG mask')
    parser.add_argument('--attn-fg-kernel', type=int, default=3,
                        help='Morphology kernel size (odd int)')
    # Foreground strategy
    parser.add_argument('--fg-mode', type=str, default='auto',
                        choices=['auto', 'rembg', 'thresh', 'attn', 'attn_intersect'],
                        help=("Foreground source: "
                              "'auto' (try rembg -> thresh), "
                              "'rembg' (only), "
                              "'thresh' (white bg fallback), "
                              "'attn' (attention-only), "
                              "'attn_intersect' (heuristic ∩ attention)."))
    parser.add_argument('--attn-mask-layer', type=int, default=None,
                        help='If set, use this attention layer to build the FG mask; '
                             'otherwise use the matching layer.')
    # ---- NEW: alignment stage control ----
    parser.add_argument('--align-stage', type=str, default='image', choices=['none', 'image', 'mask', 'both'], help='Apply keypoint alignment to: images, masks, both, or none')
    # (optional) save debug
    parser.add_argument('--save-debug', action='store_true', help='Write extra debug PNGs for masks/attention')
    parser.add_argument('--tps-pad-cells', type=int, default=1, help='Pad the TPS bbox by this many grid cells on each side (integer).')


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model + processor
    print(f"Loading DINOv3: {args.dinov3_local_ckpt}")
    print(f"Architecture: {args.dinov3_arch}")
    processor, model, patch_size = load_dinov3_local_checkpoint(args.dinov3_local_ckpt, args.dinov3_arch, args.img_size, device)

    # Round size
    img_size = (args.img_size // patch_size) * patch_size
    if img_size != args.img_size:
        print(f"Adjusted img_size from {args.img_size} to {img_size}")
    grid_size = img_size // patch_size
    print(f"Grid: {grid_size}x{grid_size}, Patch size: {patch_size}")

    # Output dirs
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem1 = Path(args.img1).stem.replace(' ', '_')[:30]
        stem2 = Path(args.img2).stem.replace(' ', '_')[:30]
        # Add key parameters for easy identification
        mode_str = args.matching_mode
        layers_str = f"L{'_'.join(map(str, args.layers))}" if args.layers else f"L{args.layer}"
        args.output_dir = (
            f"results_{mode_str}_{ts}_"
            f"{stem1}_vs_{stem2}_"
            f"{layers_str}_fg{int(args.fg_thresh*100)}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    maps_dir = os.path.join(args.output_dir, "activation_maps")
    os.makedirs(maps_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    # Save metadata immediately
    save_run_metadata(args, args.output_dir, img_size, grid_size, patch_size)
    
    # Load & prep images
    print("Loading images...")
    imgA_raw = Image.open(args.img1).convert('RGB')
    imgB_raw = Image.open(args.img2).convert('RGB')
    imgA = resize_preserve_aspect(imgA_raw, img_size)
    imgB = resize_preserve_aspect(imgB_raw, img_size)



    # Foreground masks
    print("Generating foreground masks...")
    maskA = generate_foreground_mask(imgA)
    maskB = generate_foreground_mask(imgB)
    Image.fromarray((maskA * 255).astype(np.uint8)).save(os.path.join(maps_dir, 'maskA_initial.png'))
    Image.fromarray((maskB * 255).astype(np.uint8)).save(os.path.join(maps_dir, 'maskB_initial.png'))

    # Optional cropping
    crop_bbox_A = None
    crop_bbox_B = None
    if not args.no_crop:
        print("Cropping to foreground...")
        imgA, crop_bbox_A = crop_to_bbox(imgA, maskA, img_size)
        imgB, crop_bbox_B = crop_to_bbox(imgB, maskB, img_size)
        maskA = generate_foreground_mask(imgA)
        maskB = generate_foreground_mask(imgB)
    
    # Optional mask erosion
    if args.mask_erode_px > 0:
        maskA = erode_mask(maskA, args.mask_erode_px)
        maskB = erode_mask(maskB, args.mask_erode_px)

    # White-out background for viz
    arrA = np.array(imgA); arrB = np.array(imgB)
    arrA[maskA == 0] = 255; arrB[maskB == 0] = 255
    imgA = Image.fromarray(arrA); imgB = Image.fromarray(arrB)
    imgA.save(os.path.join(maps_dir, 'imgA_processed.png'))
    imgB.save(os.path.join(maps_dir, 'imgB_processed.png'))

    # Optional keypoint alignment
    # Optional keypoint alignment (stage-aware)
    # ─────────────────────────────────────────────────────────────────────────
    # Optional alignment (stage-aware)
    #   --align-stage none   : no alignment, affine stays None
    #   --align-stage image  : warp image + mask using keypoints (if available)
    #   --align-stage mask   : leave images alone; warp masks only
    #   --align-stage both   : same as 'image' (image+mask in one warp)
    # Also provides a zero-shot fallback for mask alignment (ECC on masks)
    # ─────────────────────────────────────────────────────────────────────────
    affine = None  # IMPORTANT: define so later code can safely check "if affine is not None"
    if args.align_stage == 'none':
        print("Alignment: disabled by --align-stage none.")

    elif args.keypoints_json is not None:
        print(f"Attempting keypoint alignment at stage: {args.align_stage} | method: {args.align_method}")
        aligner = KeypointAligner(args.keypoints_json)
        try:
            affine_kpt, info = align_images_with_keypoints(
                aligner, args.img1, args.img2, method=args.align_method
            )
        except Exception as e:
            affine_kpt, info = None, {"error": f"{e}"}

        if affine_kpt is not None:
            affine = affine_kpt.astype(np.float32)

            if args.align_stage in ('image', 'both'):
                imgA, maskA = warp_image_and_mask(imgA, maskA, affine, img_size)
                print(f"✓ Keypoint alignment applied to IMAGE (RMSE={info.get('rmse', float('nan')):.2f} px)")

                if args.align_stage == 'both':
                    # Already warped maskA together with imgA, nothing extra to do.
                    print("✓ Keypoint alignment: MASK stage implicit with image warp")

            elif args.align_stage == 'mask':
                # Warp masks only; do NOT touch images
                w = h = int(img_size)
                maskA = cv2.warpAffine((maskA * 255).astype(np.uint8), affine, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
                maskA = maskA.astype(np.float32)
                print("✓ Keypoint alignment applied to MASKS only")

            print(f"  method={info.get('method')}  inliers={info.get('inliers', 'NA')}/{info.get('n_points', 'NA')}")
        else:
            print("✗ Keypoint alignment failed; continuing without keypoint-based warp.")
            # If user asked for mask/both, try zero-shot mask alignment as a fallback
            if args.align_stage in ('mask', 'both'):
                print("→ Trying zero-shot mask alignment (ECC on distance transforms)")
                affine_ecc = estimate_affine_from_masks(maskA, maskB)
                if affine_ecc is not None:
                    affine = affine_ecc.astype(np.float32)
                    if args.align_stage in ('image', 'both'):
                        imgA, maskA = warp_image_and_mask(imgA, maskA, affine, img_size)
                        print("✓ ECC alignment applied to IMAGE+MASK as fallback")
                    else:
                        w = h = int(img_size)
                        maskA = cv2.warpAffine((maskA * 255).astype(np.uint8), affine, (w, h),
                                               flags=cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
                        maskA = maskA.astype(np.float32)
                        print("✓ ECC alignment applied to MASKS only (fallback)")
                else:
                    print("✗ ECC mask alignment also failed; proceeding without alignment.")

    else:
        # No keypoints provided
        if args.align_stage in ('mask', 'both'):
            print("No keypoints provided → attempting zero-shot mask alignment (ECC).")
            affine_ecc = estimate_affine_from_masks(maskA, maskB)
            if affine_ecc is not None:
                affine = affine_ecc.astype(np.float32)
                if args.align_stage in ('image', 'both'):
                    imgA, maskA = warp_image_and_mask(imgA, maskA, affine, img_size)
                    print("✓ ECC alignment applied to IMAGE+MASK (no keypoints)")
                else:
                    w = h = int(img_size)
                    maskA = cv2.warpAffine((maskA * 255).astype(np.uint8), affine, (w, h),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
                    maskA = maskA.astype(np.float32)
                    print("✓ ECC alignment applied to MASKS only (no keypoints)")
            else:
                print("✗ ECC mask alignment failed; proceeding without alignment.")
        else:
            print("No keypoints provided; alignment skipped.")
    
    # Attention maps (for overlays and optional attn-fg)
    print(f"Extracting attention maps from layers {args.attn_layers}...")
    attention_maps_A = extract_attention_maps(model, imgA, processor, grid_size, patch_size, device)
    attention_maps_B = extract_attention_maps(model, imgB, processor, grid_size, patch_size, device)
    print(f"Successfully extracted attention from {len(attention_maps_A)} layers")
    for layer_idx in args.attn_layers:
        if layer_idx in attention_maps_A:
            save_attention_overlay(imgA, attention_maps_A[layer_idx], maskA, os.path.join(maps_dir, f'imgA_attention_layer{layer_idx}.png'), alpha=0.5)
            print(f"  Saved attention overlay for image A, layer {layer_idx}")
        if layer_idx in attention_maps_B:
            save_attention_overlay(imgB, attention_maps_B[layer_idx], maskB, os.path.join(maps_dir, f'imgB_attention_layer{layer_idx}.png'), alpha=0.5)
            print(f"  Saved attention overlay for image B, layer {layer_idx}")

    matching_layer = args.attn_layers[-1]
    if matching_layer not in attention_maps_A:
        matching_layer = max(attention_maps_A.keys())
        print(f"Warning: Layer {args.attn_layers[-1]} not available, using layer {matching_layer}")
    attention_A = attention_maps_A[matching_layer]
    attention_B = attention_maps_B[matching_layer]

    # Optional: attention∩mask foreground
    # ------------------------------------------------------------------------
    # Build a robust attention-derived foreground mask (supersedes simple attn∩mask)
    # ------------------------------------------------------------------------
    if args.attn_fg:
        # 1) Take the selected attention layer map (H x W in [0,1])
        A_attn = attention_A.copy()
        B_attn = attention_B.copy()

        # 2) Threshold either by absolute or percentile
        if args.attn_fg_thresh_mode == 'absolute':
            thrA = float(args.attn_fg_thresh)
            thrB = float(args.attn_fg_thresh)
        else:
            # percentile per image (e.g., top 85%)
            pct = np.clip(args.attn_fg_thresh_pct, 0.0, 100.0)
            thrA = np.percentile(A_attn, pct)
            thrB = np.percentile(B_attn, pct)
    
        A_attn_bin = (A_attn >= thrA).astype(np.uint8)
        B_attn_bin = (B_attn >= thrB).astype(np.uint8)
    
        # 3) Optional morphology to clean the attention masks
        if args.attn_fg_morph != 'none' and args.attn_fg_kernel > 1:
            ksz = max(1, int(args.attn_fg_kernel))
            if ksz % 2 == 0:
                ksz += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            if args.attn_fg_morph in ('open', 'openclose'):
                A_attn_bin = cv2.morphologyEx(A_attn_bin, cv2.MORPH_OPEN, kernel)
                B_attn_bin = cv2.morphologyEx(B_attn_bin, cv2.MORPH_OPEN, kernel)
            if args.attn_fg_morph in ('close', 'openclose'):
                A_attn_bin = cv2.morphologyEx(A_attn_bin, cv2.MORPH_CLOSE, kernel)
                B_attn_bin = cv2.morphologyEx(B_attn_bin, cv2.MORPH_CLOSE, kernel)

        # 4) Combine with original masks
        if args.attn_fg_mode == 'intersection':
            maskA = (maskA.astype(bool) & (A_attn_bin > 0)).astype(np.float32)
            maskB = (maskB.astype(bool) & (B_attn_bin > 0)).astype(np.float32)
        elif args.attn_fg_mode == 'union':
            maskA = (maskA.astype(bool) | (A_attn_bin > 0)).astype(np.float32)
            maskB = (maskB.astype(bool) | (B_attn_bin > 0)).astype(np.float32)
        else:  # 'attn_only'
            maskA = (A_attn_bin > 0).astype(np.float32)
            maskB = (B_attn_bin > 0).astype(np.float32)
    
        # 5) Optional: erode/border suppression you already have later; keep here just debug saves
        if args.save_debug:
            # upscale attention masks to image size for visualization
            attnA_vis = (cv2.resize(A_attn, (img_size, img_size), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
            attnB_vis = (cv2.resize(B_attn, (img_size, img_size), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
            Image.fromarray(attnA_vis).save(os.path.join(maps_dir, 'attn_selected_layer_A_raw.png'))
            Image.fromarray(attnB_vis).save(os.path.join(maps_dir, 'attn_selected_layer_B_raw.png'))
            Image.fromarray((maskA * 255).astype(np.uint8)).save(os.path.join(maps_dir, 'maskA_after_attn_fg.png'))
            Image.fromarray((maskB * 255).astype(np.uint8)).save(os.path.join(maps_dir, 'maskB_after_attn_fg.png'))

    
    
    
    # ==========================
    # PATCH TOKENS (layer sweep)
    # ==========================
    use_multi = args.layers is not None and len(args.layers) > 0
    if use_multi:
        print(f"Extracting tokens for layers {args.layers} (agg={args.layer_agg})...")
        tokensA_list = extract_tokens_multi(imgA, model, processor, patch_size, img_size, args.layers, device)
        tokensB_list = extract_tokens_multi(imgB, model, processor, patch_size, img_size, args.layers, device)
        # Fused tokens (used by functions that expect tokens)
        patchesA = fuse_tokens(tokensA_list, method=args.layer_agg)
        patchesB = fuse_tokens(tokensB_list, method=args.layer_agg)
        # Also keep fused sims for histogram/matching
        sims_per_layer = layerwise_similarities(tokensA_list, tokensB_list)  # list of [N,N]
        sims_tensor = fuse_similarities(sims_per_layer, method=args.layer_agg)  # [N,N] torch
        if args.report_layer_std:
            # Simple diagnostic: std of best-in-B over layers
            with torch.no_grad():
                per_layer_best = torch.stack([S.max(dim=1).values for S in sims_per_layer], dim=0)  # [L,N]
                std_across_layers = per_layer_best.std(dim=0).mean().item()
            print(f"Across-layer std (avg over patches): {std_across_layers:.4f}")
    else:
        print(f"Extracting patch embeddings (single layer {args.layer})...")
        patchesA = extract_patch_embeddings_dinov3(imgA, model, processor, patch_size, img_size, args.layer, device)
        patchesB = extract_patch_embeddings_dinov3(imgB, model, processor, patch_size, img_size, args.layer, device)
        with torch.no_grad():
            sims_tensor = patchesA @ patchesB.T

    sims = sims_tensor.cpu().numpy()

    # ==================== FIX: Compute fillA and fillB BEFORE using them ====================
    fillA = make_patch_fill(maskA, grid_size, patch_size)
    fillB = make_patch_fill(maskB, grid_size, patch_size)
    # ========================================================================================

    # Foreground gating with border suppression
    validA = foreground_patch_mask(maskA, grid_size, patch_size, fg_thresh=args.fg_thresh, border=args.border_patches)
    validB = foreground_patch_mask(maskB, grid_size, patch_size, fg_thresh=args.fg_thresh, border=args.border_patches)
    joint_fg = validA & validB
    print(f"Foreground patches: A={validA.sum()}/{len(validA)}, B={validB.sum()}/{len(validB)}, Joint={joint_fg.sum()}/{len(validA)}")
    
    # ========================================================================
    # TPS HOMOLOGY MODE (NEW!)
    # ========================================================================
    # Resolve TPS grid parameters from CLI


    if args.matching_mode in ['tps_homology', 'both']:
        print("\n" + "="*60)
        print("TPS HOMOLOGY-BASED MATCHING (Grid Warping)")
        print("="*60)
        
        # Check if keypoints available
        if args.keypoints_json is None:
            print("ERROR: --keypoints-json required for TPS homology mode")
            print("Falling back to similarity matching only")
        else:
            # Load keypoints
            ### from DINOSAR_keypoint_alignment import KeypointAligner
            aligner = KeypointAligner(args.keypoints_json)
            
            kptsA = aligner.get_keypoints(args.img1)
            kptsB = aligner.get_keypoints(args.img2)
            
            # add keypoints to visualization
            if kptsA is not None and kptsB is not None:
                # Strip visibility column if present (COCO format: [x, y, visibility])
                if kptsA.shape[1] == 3:
                    kptsA = kptsA[:, :2]  # Keep only [x, y]
                if kptsB.shape[1] == 3:
                    kptsB = kptsB[:, :2]  # Keep only [x, y]
                print(f"Loaded keypoints (original space): A={len(kptsA)}, B={len(kptsB)}")
    
                # Transform keypoints to match processed images
                # Step 1: Scale from original image to resize_preserve_aspect output
                def scale_kpts_to_preserved_aspect(kpts, img_orig, img_size):
                    """Transform keypoints through resize_preserve_aspect."""
                    orig_w, orig_h = img_orig.size
                    scale = min(img_size / orig_w, img_size / orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
        
                    # Scale coordinates
                    kpts_scaled = kpts * scale
        
                    # Add centering offset
                    offset_x = (img_size - new_w) / 2
                    offset_y = (img_size - new_h) / 2
                    kpts_scaled[:, 0] += offset_x
                    kpts_scaled[:, 1] += offset_y
        
                    return kpts_scaled
    
                kptsA_scaled = scale_kpts_to_preserved_aspect(kptsA, imgA_raw, img_size)
                kptsB_scaled = scale_kpts_to_preserved_aspect(kptsB, imgB_raw, img_size)
    
                # Step 2: Apply crop transformation if cropping was done
                if not args.no_crop and crop_bbox_A is not None:
                    # Crop translates coordinates, then the result is resize_preserve_aspect'd again
                    x0, y0, x1, y1 = crop_bbox_A
                    crop_w, crop_h = x1 - x0, y1 - y0
        
                    # Translate to crop space
                    kptsA_crop = kptsA_scaled.copy()
                    kptsA_crop[:, 0] -= x0
                    kptsA_crop[:, 1] -= y0
        
                    # Scale from crop size to img_size (resize_preserve_aspect is called again)
                    scale = min(img_size / crop_w, img_size / crop_h)
                    new_w = int(crop_w * scale)
                    new_h = int(crop_h * scale)
                    kptsA_crop *= scale
        
                    # Add centering offset
                    kptsA_crop[:, 0] += (img_size - new_w) / 2
                    kptsA_crop[:, 1] += (img_size - new_h) / 2
        
                    kptsA_scaled = kptsA_crop
    
                if not args.no_crop and crop_bbox_B is not None:
                    x0, y0, x1, y1 = crop_bbox_B
                    crop_w, crop_h = x1 - x0, y1 - y0
        
                    kptsB_crop = kptsB_scaled.copy()
                    kptsB_crop[:, 0] -= x0
                    kptsB_crop[:, 1] -= y0
        
                    scale = min(img_size / crop_w, img_size / crop_h)
                    new_w = int(crop_w * scale)
                    new_h = int(crop_h * scale)
                    kptsB_crop *= scale
        
                    kptsB_crop[:, 0] += (img_size - new_w) / 2
                    kptsB_crop[:, 1] += (img_size - new_h) / 2
        
                    kptsB_scaled = kptsB_crop
    
                # Step 3: Apply affine transformation (only for A, which was aligned to B)
                if affine is not None:
                    ones = np.ones((len(kptsA_scaled), 1))
                    kptsA_hom = np.hstack([kptsA_scaled, ones])
                    kptsA_scaled = (affine @ kptsA_hom.T).T
    
                # Final keypoints
                kptsA = kptsA_scaled
                kptsB = kptsB_scaled
    
                print(f"Transformed keypoints:")
                print(f"  A range: x=[{kptsA[:,0].min():.1f}, {kptsA[:,0].max():.1f}], y=[{kptsA[:,1].min():.1f}, {kptsA[:,1].max():.1f}]")
                print(f"  B range: x=[{kptsB[:,0].min():.1f}, {kptsB[:,0].max():.1f}], y=[{kptsB[:,1].min():.1f}, {kptsB[:,1].max():.1f}]")
    
                # Validate keypoints are in valid range
                if not (0 <= kptsA[:,0].min() and kptsA[:,0].max() < img_size and 
                        0 <= kptsA[:,1].min() and kptsA[:,1].max() < img_size):
                    print("WARNING: Image A keypoints outside valid range!")
                if not (0 <= kptsB[:,0].min() and kptsB[:,0].max() < img_size and 
                        0 <= kptsB[:,1].min() and kptsB[:,1].max() < img_size):
                    print("WARNING: Image B keypoints outside valid range!")

                # Load or compute consensus shape
                consensus_shape = None
                if args.consensus_shape_db:
                    try:
                        consensus_data = np.load(args.consensus_shape_db)
                        # Try to load the right species if multiple in file
                        if 'consensus_shape' in consensus_data:
                            consensus_shape = consensus_data['consensus_shape']
                        else:
                            # File has multiple species, use first
                            key = list(consensus_data.keys())[0]
                            consensus_shape = consensus_data[key]
                        print(f"✓ Loaded consensus shape from {args.consensus_shape_db}")
                    except Exception as e:
                        print(f"Warning: Could not load consensus shape: {e}")
                
                if consensus_shape is None:
                    print("Computing consensus shape from current pair...")
                    consensus_shape = compute_consensus_shape([kptsA, kptsB])
           
                # ------------------------------------------------------------
                # Build TPS grid (now controllable by rows/cols or cell_size)
                # and optionally snap to ViT patch lattice
                # ------------------------------------------------------------
                tps_rows = getattr(args, 'tps_grid_rows', None)
                tps_cols = getattr(args, 'tps_grid_cols', None)
                grid_snap = bool(getattr(args, 'tps_grid_snap', False))
                tps_cell = int(getattr(args, 'tps_grid_cell_size', 16))

                # Snap cell size to a multiple of patch_size if requested
                chosen_cell_size = int(tps_cell)
                if grid_snap and patch_size is not None and patch_size > 0:
                    if chosen_cell_size % patch_size != 0:
                        k = max(1, round(chosen_cell_size / patch_size))
                        chosen_cell_size = int(k * patch_size)
                        print(f"Snapped TPS cell size to {chosen_cell_size} (k={k} × patch_size={patch_size})")

                # ─────────────────────────────────────────────────────────
                # PAD THE CONSENSUS SHAPE BBOX BY N CELLS ON EACH SIDE
                # → gives a border around landmarks
                # ─────────────────────────────────────────────────────────
                pad_cells = max(0, int(getattr(args, 'tps_pad_cells', 1)))
                pad_px = pad_cells * float(chosen_cell_size)
                if grid_snap and patch_size is not None and patch_size > 0:
                    pad_px = np.ceil(pad_px / patch_size) * patch_size

                # Compute bbox in consensus space
                cx0, cy0 = np.min(consensus_shape, axis=0)
                cx1, cy1 = np.max(consensus_shape, axis=0)

                # Expand by pad
                cx0_p = cx0 - pad_px
                cy0_p = cy0 - pad_px
                cx1_p = cx1 + pad_px
                cy1_p = cy1 + pad_px

                # Clamp to image bounds
                cx0_p = max(0.0, min(cx0_p, float(img_size - 2)))
                cy0_p = max(0.0, min(cy0_p, float(img_size - 2)))
                cx1_p = max(2.0, min(cx1_p, float(img_size)))
                cy1_p = max(2.0, min(cy1_p, float(img_size)))

                # Add four padded corners so the grid covers the padded bbox
                pad_corners = np.array([
                    [cx0_p, cy0_p],
                    [cx1_p, cy0_p],
                    [cx0_p, cy1_p],
                    [cx1_p, cy1_p],
                ], dtype=np.float32)

                # Augment consensus points
                consensus_shape_aug = np.vstack([consensus_shape.astype(np.float32), pad_corners])
                print(f"Using pad={pad_cells} cell(s) per side (≈{int(pad_px)} px) for TPS grid border.")

                # Now create the grid (rows/cols override cell_size)
                if tps_rows is not None:
                    if tps_cols is None:
                        tps_cols = tps_rows
                    print(f"Creating consensus grid with rows={tps_rows}, cols={tps_cols}.")
                    grid_info = create_consensus_grid(
                        consensus_keypoints=consensus_shape_aug,
                        rows=int(tps_rows),
                        cols=int(tps_cols),
                        margin=0.10
                    )
                else:
                    print(f"Creating consensus grid with cell_size={chosen_cell_size}px.")
                    grid_info = create_consensus_grid(
                        consensus_keypoints=consensus_shape_aug,
                        cell_size=int(chosen_cell_size),
                        margin=0.10
                    )

                # Only print AFTER grid_info exists
                print(f"  TPS grid: {grid_info['n_rows']}×{grid_info['n_cols']} cells "
                      f"(~{grid_info['cell_size']:.1f}px cell)")

                # create the grid in consensus space (rows/cols override cell_size if provided)
                if tps_rows is not None:
                    if tps_cols is None:
                        tps_cols = tps_rows
                    print(f"Creating consensus grid with rows={tps_rows}, cols={tps_cols}...")
                    grid_info = create_consensus_grid(
                        consensus_keypoints=consensus_shape_aug,   # <── changed
                        rows=int(tps_rows),
                        cols=int(tps_cols),
                        margin=0.10
                    )
                else:
                    print(f"Creating consensus grid with cell_size={chosen_cell_size}px...")
                    grid_info = create_consensus_grid(
                        consensus_keypoints=consensus_shape_aug,   # <── changed
                        cell_size=int(chosen_cell_size),
                        margin=0.10
                    )
                
                # Create TPS warpers (consensus → specimen)
                print("Fitting TPS transformations...")
                warperA = TPSGridWarper(consensus_shape, kptsA)
                warperB = TPSGridWarper(consensus_shape, kptsB)
                
                # Warp grid to each specimen
                print("Warping grid to specimens...")
                warped_grid_A = warp_grid_to_specimen(grid_info, warperA)
                warped_grid_B = warp_grid_to_specimen(grid_info, warperB)
                
                # Compute DINOv3 patch centers
                patch_centers = []
                for i in range(grid_size):  # grid_size from your existing code
                    for j in range(grid_size):
                        x = j * patch_size + patch_size // 2
                        y = i * patch_size + patch_size // 2
                        patch_centers.append([x, y])
                patch_centers = np.array(patch_centers)
                
                # Map patches to warped grid cells
                print("Mapping DINOv3 patches to grid cells...")
                patch_to_cell_A = map_patches_to_warped_grid(
                    patch_centers, warped_grid_A, img_size
                )
                patch_to_cell_B = map_patches_to_warped_grid(
                    patch_centers, warped_grid_B, img_size
                )
                
                print(f"  Specimen A: {len(patch_to_cell_A)}/{len(validA)} patches mapped")
                print(f"  Specimen B: {len(patch_to_cell_B)}/{len(validB)} patches mapped")
                
                # Compute homologous dissimilarity
                print("Computing homologous dissimilarities...")
                tps_dissim_stats = compute_homologous_dissimilarity(
                    patchesA, patchesB,
                    patch_to_cell_A, patch_to_cell_B,
                    maskA=validA, maskB=validB
                )
                
                if 'error' not in tps_dissim_stats:
                    print(f"\n✓ TPS Homology Results:")
                    print(f"  Mean dissimilarity:   {tps_dissim_stats['mean']:.4f}")
                    print(f"  Median dissimilarity: {tps_dissim_stats['median']:.4f}")
                    print(f"  Std dissimilarity: {tps_dissim_stats['std']:.4f}")
                    print(f"  P95 dissimilarity:    {tps_dissim_stats['p95']:.4f} <-- Key metric")
                    print(f"  P99 dissimilarity:    {tps_dissim_stats['p99']:.4f}")
                    print(f"  Common grid cells:    {tps_dissim_stats['num_cells']}")
                    
                    # Save TPS statistics
                    # ========================================================================
                    # CREATE TPS SUBDIRECTORY for organized outputs
                    # ========================================================================
                    tps_subdir = os.path.join(args.output_dir, 'dissimilarity_tps_homology')
                    os.makedirs(tps_subdir, exist_ok=True)
                    
                    # ========================================================================
                    # Extract per-cell dissimilarities for histogram analysis
                    # ========================================================================
                    tps_dissim_values = np.array([
                        info['dissimilarity'] 
                        for info in tps_dissim_stats['per_cell'].values()
                    ])
                    
                    # ========================================================================
                    # Compute comprehensive TPS histogram statistics (parallel to ranking mode)
                    # ========================================================================
                    tps_hist_stats = compute_histogram_statistics(tps_dissim_values)
                    tps_hist_stats['num_cells_total'] = grid_info['n_rows'] * grid_info['n_cols']
                    tps_hist_stats['num_cells_common'] = tps_dissim_stats['num_cells']
                    tps_hist_stats['matching_mode'] = 'tps_homology'
                    tps_hist_stats['grid_cell_size'] = grid_info['cell_size']
                    tps_hist_stats['grid_dimensions'] = {
                        'rows': grid_info['n_rows'],
                        'cols': grid_info['n_cols']
                    }
                    
                    # Save TPS histogram statistics
                    with open(os.path.join(tps_subdir, 'tps_histogram_statistics.json'), 'w') as f:
                        json.dump(tps_hist_stats, f, indent=2)
                    print(f"  Saved: dissimilarity_tps_homology/tps_histogram_statistics.json")
                    
                    # ========================================================================
                    # Plot TPS histogram (parallel to ranking mode)
                    # ========================================================================
                    # ========================================================================
                    # Plot TPS histogram - 3-PANEL LAYOUT (parallel to ranking mode)
                    # ========================================================================
                    # Prepare data for 3-panel visualization
                    # Panel 1: Common cells (cells present in both specimens)
                    # Panel 2: High-coverage vs low-coverage cells
                    # Panel 3: Distribution by patch density

                    # Collect patch counts and dissimilarities
                    patch_counts_A = np.array([info['n_patches_A'] for info in tps_dissim_stats['per_cell'].values()])
                    patch_counts_B = np.array([info['n_patches_B'] for info in tps_dissim_stats['per_cell'].values()])
                    avg_patch_counts = (patch_counts_A + patch_counts_B) / 2

                    # Split into high/low coverage based on median
                    median_coverage = np.median(avg_patch_counts)
                    high_coverage_mask = avg_patch_counts >= median_coverage
                    low_coverage_mask = ~high_coverage_mask

                    dissim_high_coverage = tps_dissim_values[high_coverage_mask]
                    dissim_low_coverage = tps_dissim_values[low_coverage_mask]

                    # Create 3-panel figure
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

                    # ========== PANEL 1: Common Cells (main distribution) ==========
                    ax1.hist(tps_dissim_values, bins=40, alpha=0.7, edgecolor='black', color='seagreen')
                    ax1.axvline(tps_hist_stats['mean'], color='red', linestyle='--', linewidth=2, 
                               label=f"Mean: {tps_hist_stats['mean']:.3f}")
                    ax1.axvline(tps_hist_stats['p95'], color='orange', linestyle='--', linewidth=2, 
                               label=f"P95: {tps_hist_stats['p95']:.3f}")
                    ax1.set_xlabel('Dissimilarity', fontsize=11)
                    ax1.set_ylabel('Frequency', fontsize=11)
                    ax1.set_title(f'Common Cells (n={len(tps_dissim_values)})\nTPS Homologous Regions', 
                                 fontsize=11, fontweight='bold')
                    ax1.legend(fontsize=10)
                    ax1.grid(alpha=0.3)

                    # ========== PANEL 2: High vs Low Coverage Cells ==========
                    ax2.hist(dissim_high_coverage, bins=30, alpha=0.6, edgecolor='black', 
                            color='darkgreen', label=f'High coverage (≥{median_coverage:.0f} patches)')
                    ax2.hist(dissim_low_coverage, bins=30, alpha=0.6, edgecolor='black', 
                            color='lightgreen', label=f'Low coverage (<{median_coverage:.0f} patches)')

                    # Add means for each group
                    if len(dissim_high_coverage) > 0:
                        mean_high = dissim_high_coverage.mean()
                        ax2.axvline(mean_high, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
                    if len(dissim_low_coverage) > 0:
                        mean_low = dissim_low_coverage.mean()
                        ax2.axvline(mean_low, color='lightgreen', linestyle='--', linewidth=2, alpha=0.7)

                    ax2.set_xlabel('Dissimilarity', fontsize=11)
                    ax2.set_ylabel('Frequency', fontsize=11)
                    ax2.set_title('Coverage Comparison\n(High vs Low Patch Density)', fontsize=11, fontweight='bold')
                    ax2.legend(fontsize=9)
                    ax2.grid(alpha=0.3)

                    # ========== PANEL 3: Per-Specimen Patch Balance ==========
                    # Show cells where patch counts are imbalanced between specimens
                    patch_imbalance = np.abs(patch_counts_A - patch_counts_B)
                    balanced_mask = patch_imbalance <= 2  # Within 2 patches
                    imbalanced_mask = ~balanced_mask

                    dissim_balanced = tps_dissim_values[balanced_mask]
                    dissim_imbalanced = tps_dissim_values[imbalanced_mask]

                    if len(dissim_imbalanced) > 0:
                        ax3.hist(dissim_imbalanced, bins=30, alpha=0.7, edgecolor='black', 
                                color='coral', label=f'Imbalanced (n={len(dissim_imbalanced)})')
                        mean_imbal = dissim_imbalanced.mean()
                        ax3.axvline(mean_imbal, color='darkred', linestyle='--', linewidth=2,
                                   label=f'Mean: {mean_imbal:.3f}')
                        
                    if len(dissim_balanced) > 0:
                        ax3.hist(dissim_balanced, bins=30, alpha=0.5, edgecolor='black', 
                                color='lightblue', label=f'Balanced (n={len(dissim_balanced)})')

                    ax3.set_xlabel('Dissimilarity', fontsize=11)
                    ax3.set_ylabel('Frequency', fontsize=11)
                    ax3.set_title('Patch Balance\n(A vs B patch counts)', fontsize=11, fontweight='bold')
                    ax3.legend(fontsize=9)
                    ax3.grid(alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(tps_subdir, 'tps_dissimilarity_histogram_cov_balanced.png'), dpi=150)
                    plt.close()
                    print(f"  Saved: dissimilarity_tps_homology/tps_dissimilarity_histogram_cov_balanced.png")
                    
                    # ========================================================================
                    # Plot TPS histogram - 3-PANEL LAYOUT with background (parallel to ranking)
                    # ========================================================================
                    # ========================================================================
                    # Plot TPS histogram - 3-PANEL with TRUE BACKGROUND (patches outside TPS grid)
                    # ========================================================================
                    
                    # Step 1: Identify which patches belong to COMMON TPS cells (foreground)
                    common_cells = set(tps_dissim_stats['per_cell'].keys())
                    
                    patches_in_common_cells_A = set()
                    patches_in_common_cells_B = set()
                    
                    for patch_idx, cell in patch_to_cell_A.items():
                        if cell in common_cells and validA[patch_idx]:
                            patches_in_common_cells_A.add(patch_idx)
                    
                    for patch_idx, cell in patch_to_cell_B.items():
                        if cell in common_cells and validB[patch_idx]:
                            patches_in_common_cells_B.add(patch_idx)
                    
                    # Step 2: Identify background patches (exist but NOT in common TPS cells)
                    # These are patches that:
                    # - Pass foreground mask (validA/validB)
                    # - But are NOT part of any common TPS cell
                    background_patches_A = []
                    for i in range(len(validA)):
                        if validA[i] and i not in patches_in_common_cells_A:
                            background_patches_A.append(i)
                    
                    background_patches_B = []
                    for i in range(len(validB)):
                        if validB[i] and i not in patches_in_common_cells_B:
                            background_patches_B.append(i)
                    
                    # Step 3: Compute dissimilarity for background patches
                    # Use the existing similarity matrix (sims is already computed from patchesA @ patchesB.T)
                    dissim_background_values = []
                    
                    if len(background_patches_A) > 0:
                        background_indices = np.array(background_patches_A)
                        best_match_sims = sims[background_indices].max(axis=1)
                        dissim_background_A = 1.0 - best_match_sims
                        dissim_background_values.extend(dissim_background_A[np.isfinite(dissim_background_A)].tolist())
                    
                    if len(background_patches_B) > 0:
                        background_indices_B = np.array(background_patches_B)
                        best_match_sims_B = sims[:, background_indices_B].max(axis=0)
                        dissim_background_B = 1.0 - best_match_sims_B
                        dissim_background_values.extend(dissim_background_B[np.isfinite(dissim_background_B)].tolist())
                    
                    dissim_background_array = np.array(dissim_background_values) if len(dissim_background_values) > 0 else np.array([])
                    
                    # Step 4: Get foreground (common TPS cells) dissimilarities
                    dissim_foreground = tps_dissim_values  # These are already the common cells
                    
                    # Step 5: All dissimilarities (combine TPS + background)
                    dissim_all = np.concatenate([dissim_foreground, dissim_background_array]) if len(dissim_background_array) > 0 else dissim_foreground
                    
                    # ========================================================================
                    # Create 3-panel figure (matching ranking mode exactly)
                    # ========================================================================
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

                    # ========== PANEL 1: TPS Common Cells (foreground) ==========
                    ax1.hist(dissim_foreground, bins=50, alpha=0.7, edgecolor='black', color='seagreen')
                    mean_fg = float(np.mean(dissim_foreground))
                    p95_fg = float(np.percentile(dissim_foreground, 95))
                    ax1.axvline(mean_fg, color='red', linestyle='--', linewidth=2, 
                               label=f"Mean: {mean_fg:.3f}")
                    ax1.axvline(p95_fg, color='orange', linestyle='--', linewidth=2, 
                               label=f"P95: {p95_fg:.3f}")
                    ax1.set_xlabel('Dissimilarity')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title(f'Used cells (n={len(dissim_foreground)})')
                    ax1.legend()
                    ax1.grid(alpha=0.3)

                    # ========== PANEL 2: All vs Used ==========
                    ax2.hist(dissim_all, bins=50, alpha=0.5, edgecolor='black', 
                            color='gray', label=f'All regions ({len(dissim_all)})', linewidth=1.5)
                    ax2.hist(dissim_foreground, bins=50, alpha=0.7, edgecolor='black', 
                            color='seagreen', label=f'Used (TPS cells)', linewidth=1.5)
                    ax2.set_xlabel('Dissimilarity')
                    ax2.set_title('All vs Used')
                    ax2.legend()
                    ax2.grid(alpha=0.3)

                    # ========== PANEL 3: Background (non-TPS patches) ==========
                    if len(dissim_background_array) > 0:
                        ax3.hist(dissim_background_array, bins=50, alpha=0.7, edgecolor='black', 
                                color='lightcoral', linewidth=1.5)
                        mean_bg = float(np.mean(dissim_background_array))
                        ax3.axvline(mean_bg, color='darkred', linestyle='--', linewidth=2,
                                   label=f'Mean: {mean_bg:.3f}')
                        ax3.set_xlabel('Dissimilarity')
                        ax3.set_ylabel('Frequency')
                        ax3.set_title(f'Background (n={len(dissim_background_array)})')
                        ax3.legend()
                        ax3.grid(alpha=0.3)
                    else:
                        ax3.text(0.5, 0.5, 'No background patches\n(all foreground patches\ncovered by TPS grid)', 
                                ha='center', va='center', transform=ax3.transAxes, 
                                fontsize=11, style='italic')
                        ax3.set_xlabel('Dissimilarity')
                        ax3.set_title('Background (n=0)')
                        ax3.grid(alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(tps_subdir, 'tps_dissimilarity_histogram.png'), dpi=150)
                    plt.close()
                    print(f"  Saved: dissimilarity_tps_homology/tps_dissimilarity_histogram.png")
                    print(f"    Foreground: {len(dissim_foreground)} TPS cells")
                    print(f"    Background: {len(dissim_background_array)} patches outside TPS grid")
                    print(f"    Total: {len(dissim_all)} regions")


                    # ========================================================================
                    # Export COMPREHENSIVE TPS dissimilarity data (grid + background)
                    # ========================================================================
                    
                    # Collect all dissimilarities with metadata
                    tps_comprehensive = {
                        'tps_grid_cells': [],
                        'background_patches': []
                    }
                    
                    # TPS grid cells (foreground)
                    for cell, info in tps_dissim_stats['per_cell'].items():
                        row, col = cell
                        tps_comprehensive['tps_grid_cells'].append({
                            'cell_row': int(row),
                            'cell_col': int(col),
                            'dissimilarity': float(info['dissimilarity']),
                            'n_patches_A': int(info['n_patches_A']),
                            'n_patches_B': int(info['n_patches_B']),
                            'avg_patches': float((info['n_patches_A'] + info['n_patches_B']) / 2),
                            'patch_imbalance': int(abs(info['n_patches_A'] - info['n_patches_B'])),
                            'is_tps_cell': True
                        })
                    
                    # Background patches (outside TPS grid)
                    if len(background_patches_A) > 0:
                        for idx in background_patches_A:
                            best_sim = float(sims[idx].max())
                            tps_comprehensive['background_patches'].append({
                                'patch_idx': int(idx),
                                'specimen': 'A',
                                'dissimilarity': float(1.0 - best_sim),
                                'cosine_sim': best_sim,
                                'row': int(idx // grid_size),
                                'col': int(idx % grid_size),
                                'is_tps_cell': False
                            })
                    
                    if len(background_patches_B) > 0:
                        for idx in background_patches_B:
                            best_sim = float(sims[:, idx].max())
                            tps_comprehensive['background_patches'].append({
                                'patch_idx': int(idx),
                                'specimen': 'B',
                                'dissimilarity': float(1.0 - best_sim),
                                'cosine_sim': best_sim,
                                'row': int(idx // grid_size),
                                'col': int(idx % grid_size),
                                'is_tps_cell': False
                            })
                    
                    # Save as JSON
                    with open(os.path.join(tps_subdir, 'tps_comprehensive_dissimilarities.json'), 'w') as f:
                        json.dump(tps_comprehensive, f, indent=2)
                    
                    # Save as NPZ for efficient analysis
                    np.savez_compressed(
                        os.path.join(tps_subdir, 'tps_comprehensive_dissimilarities.npz'),
                        tps_cell_dissim=dissim_foreground,
                        tps_cell_rows=np.array([cell[0] for cell in tps_dissim_stats['per_cell'].keys()]),
                        tps_cell_cols=np.array([cell[1] for cell in tps_dissim_stats['per_cell'].keys()]),
                        background_dissim=dissim_background_array,
                        background_patches_A=np.array(background_patches_A) if len(background_patches_A) > 0 else np.array([]),
                        background_patches_B=np.array(background_patches_B) if len(background_patches_B) > 0 else np.array([]),
                        all_dissim=dissim_all,
                        grid_dimensions=np.array([grid_info['n_rows'], grid_info['n_cols']]),
                        consensus_shape=consensus_shape
                    )
                    
                    # Save summary CSV for meta-analysis
                    tps_summary_csv = os.path.join(tps_subdir, 'tps_summary_stats.csv')
                    with open(tps_summary_csv, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['category', 'n_regions', 'mean', 'median', 'std', 'min', 'max',
                                        'p25', 'p75', 'p95', 'p99', 'iqr'])
                        
                        # TPS grid cells
                        writer.writerow([
                            'tps_grid',
                            len(dissim_foreground),
                            np.mean(dissim_foreground),
                            np.median(dissim_foreground),
                            np.std(dissim_foreground),
                            np.min(dissim_foreground),
                            np.max(dissim_foreground),
                            np.percentile(dissim_foreground, 25),
                            np.percentile(dissim_foreground, 75),
                            np.percentile(dissim_foreground, 95),
                            np.percentile(dissim_foreground, 99),
                            np.percentile(dissim_foreground, 75) - np.percentile(dissim_foreground, 25)
                        ])
                        
                        # Background patches
                        if len(dissim_background_array) > 0:
                            writer.writerow([
                                'background',
                                len(dissim_background_array),
                                np.mean(dissim_background_array),
                                np.median(dissim_background_array),
                                np.std(dissim_background_array),
                                np.min(dissim_background_array),
                                np.max(dissim_background_array),
                                np.percentile(dissim_background_array, 25),
                                np.percentile(dissim_background_array, 75),
                                np.percentile(dissim_background_array, 95),
                                np.percentile(dissim_background_array, 99),
                                np.percentile(dissim_background_array, 75) - np.percentile(dissim_background_array, 25)
                            ])
                    
                    print(f"  Saved: dissimilarity_tps_homology/tps_comprehensive_dissimilarities.json")
                    print(f"  Saved: dissimilarity_tps_homology/tps_comprehensive_dissimilarities.npz")
                    print(f"  Saved: dissimilarity_tps_homology/tps_summary_stats.csv")

                    # ========================================================================
                    # Create TPS sparse match visualization (parallel to attention matches)
                    # ========================================================================
                    print("Creating TPS sparse match visualization...")
                    
                    # Sort cells by dissimilarity (lowest = most similar)
                    sorted_cells_all = sorted(
                        tps_dissim_stats['per_cell'].items(),
                        key=lambda x: x[1]['dissimilarity']
                    )
                    # Top 50 for visualization only
                    sorted_cells_viz = sorted_cells_all[:50]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
                    ax1.imshow(imgA); ax1.axis('off')
                    ax2.imshow(imgB); ax2.axis('off')
                    ax1.set_title(f'Specimen A: {len(sorted_cells_viz)} homologous cells\n(TPS grid-based)', 
                                fontsize=12, fontweight='bold')
                    ax2.set_title(f'Specimen B: homologous correspondences\n(TPS grid-based)', 
                                fontsize=12, fontweight='bold')
                    
                    cmap = plt.colormaps.get_cmap('viridis')
                    colors = [cmap(i / max(1, len(sorted_cells_viz))) for i in range(len(sorted_cells_viz))]
                    
                    for rank, (cell, info) in enumerate(sorted_cells_viz):
                        row, col = cell
                        color = colors[rank]
                        
                        # Get warped cell centers in specimen space
                        centerA = get_warped_cell_center(warped_grid_A, row, col)
                        centerB = get_warped_cell_center(warped_grid_B, row, col)
                        
                        xA, yA = centerA
                        xB, yB = centerB
                        
                        # Draw cell markers
                        ax1.scatter(xA, yA, color=color, s=120, alpha=0.8, 
                                   edgecolors='white', linewidths=2, marker='s')
                        ax1.text(xA, yA, str(rank + 1), color='white', fontsize=7, 
                                ha='center', va='center', fontweight='bold')
                        
                        ax2.scatter(xB, yB, color=color, s=120, alpha=0.8,
                                   edgecolors='white', linewidths=2, marker='s')
                        ax2.text(xB, yB, str(rank + 1), color='white', fontsize=7,
                                ha='center', va='center', fontweight='bold')
                        
                        # Connection line
                        fig.add_artist(ConnectionPatch(
                            xyA=(xB, yB), xyB=(xA, yA),
                            coordsA='data', coordsB='data',
                            axesA=ax2, axesB=ax1,
                            color=color, lw=1.5, alpha=0.6
                        ))
                        
                        # Annotate first 10 with dissimilarity values
                        if rank < 10:
                            ax1.text(xA, yA-15, f'{info["dissimilarity"]:.3f}',
                                    color='yellow', fontsize=6, ha='center',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(tps_subdir, 'tps_homology_matches.png'), dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved: dissimilarity_tps_homology/tps_homology_matches.png")
                    
                    # ========================================================================
                    # Save TPS match CSV (parallel to ranking mode)
                    # ========================================================================
                    # ========================================================================
                    # Save COMPLETE TPS match CSV (all common cells, not just top 50)
                    # ========================================================================
                    tps_match_csv_full = os.path.join(tps_subdir, 'tps_all_cells_dissimilarities.csv')
                    with open(tps_match_csv_full, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['rank', 'cell_row', 'cell_col', 'dissimilarity', 
                                        'n_patches_A', 'n_patches_B', 'avg_patches',
                                        'patch_imbalance'])
                        
                        for rank, (cell, info) in enumerate(sorted_cells_all):
                            row, col = cell
                            writer.writerow([
                                rank + 1, row, col,
                                info['dissimilarity'],
                                info['n_patches_A'],
                                info['n_patches_B'],
                                (info['n_patches_A'] + info['n_patches_B']) / 2,
                                abs(info['n_patches_A'] - info['n_patches_B'])
                            ])
                    print(f"  Saved: dissimilarity_tps_homology/tps_all_cells_dissimilarities.csv ({len(sorted_cells_all)} cells)")
                    
                    # Save top 50 for quick reference (parallel to ranking mode's top matches)
                    tps_match_csv_top = os.path.join(tps_subdir, 'tps_top50_matches.csv')
                    with open(tps_match_csv_top, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['rank', 'cell_row', 'cell_col', 'dissimilarity', 
                                        'n_patches_A', 'n_patches_B'])
                        
                        for rank, (cell, info) in enumerate(sorted_cells_viz):
                            row, col = cell
                            writer.writerow([
                                rank + 1, row, col,
                                info['dissimilarity'],
                                info['n_patches_A'],
                                info['n_patches_B']
                            ])
                    print(f"  Saved: dissimilarity_tps_homology/tps_top50_matches.csv (visualization subset)")
                    
                    # ========================================================================
                    # Save specimen-specific cell coverage CSVs (optional detailed view)
                    # ========================================================================
                    # This shows which specimen has better coverage in each cell
                    
                    # Specimen A perspective (sorted by A's patch count)
                    tps_csv_A_view = os.path.join(tps_subdir, 'tps_cells_by_specimen_A_coverage.csv')
                    with open(tps_csv_A_view, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['cell_row', 'cell_col', 'dissimilarity',
                                        'patches_A', 'patches_B', 'A_minus_B',
                                        'A_has_more_coverage'])
                        
                        cells_by_A = sorted(tps_dissim_stats['per_cell'].items(),
                                          key=lambda x: x[1]['n_patches_A'], reverse=True)
                        
                        for cell, info in cells_by_A:
                            row, col = cell
                            writer.writerow([
                                row, col,
                                info['dissimilarity'],
                                info['n_patches_A'],
                                info['n_patches_B'],
                                info['n_patches_A'] - info['n_patches_B'],
                                info['n_patches_A'] > info['n_patches_B']
                            ])
                    
                    # Specimen B perspective (sorted by B's patch count)
                    tps_csv_B_view = os.path.join(tps_subdir, 'tps_cells_by_specimen_B_coverage.csv')
                    with open(tps_csv_B_view, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['cell_row', 'cell_col', 'dissimilarity',
                                        'patches_B', 'patches_A', 'B_minus_A',
                                        'B_has_more_coverage'])
                        
                        cells_by_B = sorted(tps_dissim_stats['per_cell'].items(),
                                          key=lambda x: x[1]['n_patches_B'], reverse=True)
                        
                        for cell, info in cells_by_B:
                            row, col = cell
                            writer.writerow([
                                row, col,
                                info['dissimilarity'],
                                info['n_patches_B'],
                                info['n_patches_A'],
                                info['n_patches_B'] - info['n_patches_A'],
                                info['n_patches_B'] > info['n_patches_A']
                            ])
                    
                    print(f"  Saved: dissimilarity_tps_homology/tps_cells_by_specimen_A_coverage.csv")
                    print(f"  Saved: dissimilarity_tps_homology/tps_cells_by_specimen_B_coverage.csv")
                    
                    # ========================================================================
                    # Save original TPS stats files to subdirectory
                    # ========================================================================
                    tps_stats_json = tps_dissim_stats.copy()
                    if 'per_cell' in tps_stats_json:
                        per_cell_json = {
                            f"{row},{col}": info 
                            for (row, col), info in tps_stats_json['per_cell'].items()
                        }
                        tps_stats_json['per_cell'] = per_cell_json

                    tps_output = {
                        'tps_homology_stats': tps_stats_json,
                        'grid_cell_size': grid_info['cell_size'],
                        'grid_dimensions': {
                            'rows': grid_info['n_rows'],
                            'cols': grid_info['n_cols']
                        },
                        'num_keypoints': len(consensus_shape),
                        'consensus_shape': consensus_shape.tolist()
                    }

                    with open(os.path.join(tps_subdir, 'tps_homology_stats.json'), 'w') as f:
                        json.dump(tps_output, f, indent=2)
                    
                    # Save per-cell CSV
                    tps_csv = os.path.join(tps_subdir, 'tps_per_cell_dissimilarities.csv')
                    with open(tps_csv, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['cell_row', 'cell_col', 'dissimilarity', 
                                       'n_patches_A', 'n_patches_B'])
                        
                        for cell, info in tps_dissim_stats['per_cell'].items():
                            row, col = cell
                            writer.writerow([
                                row, col, 
                                info['dissimilarity'],
                                info['n_patches_A'],
                                info['n_patches_B']
                            ])
                    print(f"  Saved: dissimilarity_tps_homology/tps_per_cell_dissimilarities.csv")
                    
                    # ========================================================================
                    # Visualize TPS grid overlay (comprehensive 3-panel figure)
                    # ========================================================================
                    print("Creating TPS grid overlay visualization...")
                    imgA_np = cv2.cvtColor(np.array(imgA), cv2.COLOR_RGB2BGR)
                    imgB_np = cv2.cvtColor(np.array(imgB), cv2.COLOR_RGB2BGR)
                    
                    visualize_homologous_comparison(
                        imgA_np, imgB_np,
                        warped_grid_A, warped_grid_B,
                        patch_centers, patch_centers,
                        patch_to_cell_A, patch_to_cell_B,
                        tps_dissim_stats['per_cell'],
                        os.path.join(tps_subdir, 'tps_grid_overlay.png'),
                        keypointsA=kptsA,
                        keypointsB=kptsB
                    )
                else:
                    print("Warning: Could not compute homologous dissimilarity")
                    print(f"  Error: {tps_dissim_stats.get('error', 'unknown')}")
            
            else:
                print("ERROR: Could not load keypoints for both images")
    
    # Continue with your existing similarity-based matching code...
    # (This stays exactly as it is now)
    # ==================== NEW: Export comprehensive dissimilarity data ====================
    #=========================================================================
    # CONDITIONAL: Only run RANKING mode if requested
    # ========================================================================
    if args.matching_mode in ['similarity', 'both']:
        print("\n" + "="*60)
        print("DISSIMILARITY RANKING MODE (Attention-based)")
        print("="*60)
        
        # Create ranking subdirectory
        ranking_subdir = os.path.join(args.output_dir, 'dissimilarity_ranking')
        os.makedirs(ranking_subdir, exist_ok=True)
        
        # Export comprehensive dissimilarity data
        print("\nExporting RANKING-based dissimilarity data...")
        dissim_data = export_comprehensive_dissimilarities(
            sims=sims,
            validA=validA,
            validB=validB,
            grid_size=grid_size,
            output_dir=ranking_subdir,  # Changed to subdirectory
            fillA=fillA,
            fillB=fillB
        )
        
        # SPARSE MATCHING (uses fused tokens)
        if args.match_flavor == 'demo':
            print("\nMatching patches using DINOv3 DEMO flavor (mutual/ratio)...")
            patches_A_idx, patches_B_idx, match_scores = demo_sparse_matches(
                patchesA, patchesB, validA, validB, grid_size, patch_size,
                top_n=args.demo_topn, ratio=args.demo_ratio, mutual=args.demo_mutual,
                ransac=args.demo_verify, ransac_thresh=args.demo_ransac_reproj, border=args.border_patches
            )
        else:
            print("\nMatching patches by ATTENTION importance...")
            patches_A_idx, patches_B_idx, match_scores = match_patches_by_attention(
                attention_A, attention_B, patchesA, patchesB, maskA, maskB,
                grid_size, patch_size, top_k=args.top_matches,
                fg_thresh=args.fg_thresh, border=args.border_patches
            )
        print(f"Matched {len(patches_A_idx)} patches")

        # SCORE MODES / HISTOGRAMS
        jointA = validA; jointB = validB
        sims[:, ~jointB] = -np.inf
        sA = sims.max(axis=1)
        dA = 1.0 - sA
        dissim_values_fg = None; weights_fg = None; used_mode = args.score_mode

        if args.score_mode == 'foreground':
            keep = jointA & np.isfinite(dA)
            dissim_values_fg = dA[keep]

        elif args.score_mode == 'demo_inliers' and args.match_flavor == 'demo':
            demo_inlier_mask = np.zeros_like(jointA, dtype=bool)
            demo_inlier_mask[patches_A_idx] = True
            keep = demo_inlier_mask & np.isfinite(dA)
            if keep.sum() == 0:
                print("Warning: No demo inliers for histogram; falling back to foreground.")
                keep = jointA & np.isfinite(dA); used_mode = 'foreground'
            dissim_values_fg = dA[keep]

        elif args.score_mode == 'bidirectional':
            sB = sims_tensor.T.cpu().numpy().max(axis=1)
            dB = 1.0 - sB
            keepA = jointA & np.isfinite(dA); keepB = jointB & np.isfinite(dB)
            dissim_values_fg = np.concatenate([dA[keepA], dB[keepB]], axis=0)
            keep = keepA

        elif args.score_mode == 'topk_mean':
            K = max(1, int(args.topk))
            neg = -sims
            part = np.partition(neg, K - 1, axis=1)
            topk = -np.sort(part[:, :K], axis=1)
            sA_k = np.mean(topk, axis=1)
            dA_k = 1.0 - sA_k
            keep = jointA & np.isfinite(dA_k)
            dissim_values_fg = dA_k[keep]

        elif args.score_mode == 'attention_weighted':
            S_aw, D_aw, aw_info = attention_weighted_similarity(
                patchesA, patchesB, attention_A, attention_B, maskA, maskB,
                grid_size, patch_size, fg_thresh=args.fg_thresh, border=args.border_patches
            )
            print(f"Attention-weighted: S={S_aw:.4f}, D={D_aw:.4f}, wP95={aw_info['wP95']:.4f}")
            def patch_attn(attn_map):
                return attn_map.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1, 3)).flatten().astype(np.float32)
            A_attn = patch_attn(attention_A)
            bestB = np.argmax(sims, axis=1)
            B_attn = patch_attn(attention_B)
            w = np.sqrt(np.clip(A_attn, 0, 1) * np.clip(B_attn[bestB], 0, 1))
            keep = jointA & np.isfinite(dA) & (w > 0)
            dissim_values_fg = dA[keep]
            weights_fg = w[keep]

        else:
            print(f"Warning: Unsupported score-mode '{args.score_mode}', using 'foreground'.")
            keep = jointA & np.isfinite(dA)
            dissim_values_fg = dA[keep]
            used_mode = 'foreground'

        if dissim_values_fg is None or len(dissim_values_fg) < args.hist_min_joint:
            print(f"Warning: insufficient patches for histogram in mode '{used_mode}', falling back to 'foreground'.")
            keep = jointA & np.isfinite(dA)
            dissim_values_fg = dA[keep]
            weights_fg = None
            used_mode = 'foreground'

        # Background (for plotting only)
        dissim_values_all = 1.0 - (sims).max(axis=1)
        dissim_values_bg = dissim_values_all[~(validA & validB)]
        
        # Save raw foreground dissimilarity values
        fg_csv = os.path.join(ranking_subdir, 'ranking_foreground_dissimilarities.csv')
        with open(fg_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['patch_idx_A', 'dissimilarity', 'weight'] if weights_fg is not None else ['patch_idx_A', 'dissimilarity'])
            fg_indices = np.where(keep)[0]
            for idx, dissim in zip(fg_indices, dissim_values_fg):
                if weights_fg is not None:
                    weight_idx = np.where(fg_indices == idx)[0][0]
                    writer.writerow([int(idx), float(dissim), float(weights_fg[weight_idx])])
                else:
                    writer.writerow([int(idx), float(dissim)])
        print(f"  Saved: dissimilarity_ranking/ranking_foreground_dissimilarities.csv ({len(fg_indices)} patches)")
        
        # Stats (weighted if attention_weighted)
        if weights_fg is None:
            hist_stats = compute_histogram_statistics(dissim_values_fg)
        else:
            w = weights_fg.astype(np.float64)
            w = w / (w.sum() + 1e-9)
            vals = dissim_values_fg.astype(np.float64)
            mean_fg = float(np.sum(vals * w))
            std_fg = float(np.sqrt(np.sum(w * (vals - mean_fg) ** 2)))
            p95_fg = weighted_percentile(vals, w, 0.95)
            p99_fg = weighted_percentile(vals, w, 0.99)
            median_fg = weighted_percentile(vals, w, 0.5)
            tail_ratio = float(np.sum(w[vals > 0.3]))
            m3 = np.sum(w * (vals - mean_fg) ** 3)
            skew = float(m3 / ((std_fg ** 3) + 1e-12)) if std_fg > 0 else 0.0
            hist_stats = {
                'mean': mean_fg, 'median': median_fg, 'std': std_fg,
                'min': float(np.min(vals)), 'max': float(np.max(vals)),
                'p25': weighted_percentile(vals, w, 0.25),
                'p75': weighted_percentile(vals, w, 0.75),
                'p90': weighted_percentile(vals, w, 0.90),
                'p95': p95_fg, 'p99': p99_fg,
                'iqr': float(weighted_percentile(vals, w, 0.75) - weighted_percentile(vals, w, 0.25)),
                'tail_ratio': tail_ratio,
                'heavy_tail_ratio': float(np.sum(w[vals > 0.5])),
                'skewness': skew,
                'kurtosis': float('nan'),
                'right_tail_mean': float(np.mean(vals[vals > median_fg])) if np.any(vals > median_fg) else float('nan'),
            }

        hist_stats['num_patches_total'] = int(patchesA.shape[0])
        hist_stats['num_patches_foreground'] = int(joint_fg.sum())
        hist_stats['foreground_ratio'] = float(joint_fg.sum() / patchesA.shape[0])
        hist_stats['score_mode'] = used_mode

        with open(os.path.join(ranking_subdir, 'ranking_histogram_statistics.json'), 'w') as f:
            json.dump(hist_stats, f, indent=2)

        # ========================================================================
        # Export COMPREHENSIVE ranking dissimilarity data (foreground + background)
        # ========================================================================
        
        # Collect all dissimilarities with metadata
        ranking_comprehensive = {
            'foreground_patches': [],
            'background_patches': []
        }
        
        # Foreground (patches used in histogram)
        fg_indices = np.where(keep)[0]  # 'keep' mask from your score mode
        for idx in fg_indices:
            ranking_comprehensive['foreground_patches'].append({
                'patch_idx': int(idx),
                'dissimilarity': float(dA[idx]),
                'cosine_sim': float(sA[idx]),
                'best_match_idx_B': int(np.argmax(sims[idx])),
                'is_foreground': True,
                'row': int(idx // grid_size),
                'col': int(idx % grid_size)
            })
        
        # Background (patches not used - either not foreground or outside borders)
        bg_indices = np.where(~keep & np.isfinite(dA))[0]
        for idx in bg_indices:
            ranking_comprehensive['background_patches'].append({
                'patch_idx': int(idx),
                'dissimilarity': float(dA[idx]),
                'cosine_sim': float(sA[idx]),
                'best_match_idx_B': int(np.argmax(sims[idx])),
                'is_foreground': False,
                'row': int(idx // grid_size),
                'col': int(idx % grid_size)
            })
        
        # Save as JSON for human readability
        with open(os.path.join(ranking_subdir, 'ranking_comprehensive_dissimilarities.json'), 'w') as f:
            json.dump(ranking_comprehensive, f, indent=2)
        
        # Save as NPZ for efficient Python analysis
        np.savez_compressed(
            os.path.join(ranking_subdir, 'ranking_comprehensive_dissimilarities.npz'),
            foreground_dissim=dA[keep],
            foreground_indices=fg_indices,
            foreground_coords=np.column_stack([fg_indices % grid_size, fg_indices // grid_size]),
            background_dissim=dA[bg_indices],
            background_indices=bg_indices,
            background_coords=np.column_stack([bg_indices % grid_size, bg_indices // grid_size]),
            all_dissim=dA,
            similarity_matrix=sims,
            valid_A=validA,
            valid_B=validB,
            grid_size=grid_size
        )
        
        # Save summary CSV for quick meta-analysis
        summary_csv = os.path.join(ranking_subdir, 'ranking_summary_stats.csv')
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'n_patches', 'mean', 'median', 'std', 'min', 'max', 
                            'p25', 'p75', 'p95', 'p99', 'iqr'])
            
            # Foreground stats
            fg_dissim = dA[keep]
            writer.writerow([
                'foreground',
                len(fg_dissim),
                np.mean(fg_dissim),
                np.median(fg_dissim),
                np.std(fg_dissim),
                np.min(fg_dissim),
                np.max(fg_dissim),
                np.percentile(fg_dissim, 25),
                np.percentile(fg_dissim, 75),
                np.percentile(fg_dissim, 95),
                np.percentile(fg_dissim, 99),
                np.percentile(fg_dissim, 75) - np.percentile(fg_dissim, 25)
            ])
            
            # Background stats
            if len(bg_indices) > 0:
                bg_dissim = dA[bg_indices]
                writer.writerow([
                    'background',
                    len(bg_dissim),
                    np.mean(bg_dissim),
                    np.median(bg_dissim),
                    np.std(bg_dissim),
                    np.min(bg_dissim),
                    np.max(bg_dissim),
                    np.percentile(bg_dissim, 25),
                    np.percentile(bg_dissim, 75),
                    np.percentile(bg_dissim, 95),
                    np.percentile(bg_dissim, 99),
                    np.percentile(bg_dissim, 75) - np.percentile(bg_dissim, 25)
                ])
        
        print(f"  Saved: dissimilarity_ranking/ranking_comprehensive_dissimilarities.json")
        print(f"  Saved: dissimilarity_ranking/ranking_comprehensive_dissimilarities.npz")
        print(f"  Saved: dissimilarity_ranking/ranking_summary_stats.csv")


        print("\n" + "=" * 60)
        print(f"RANKING HISTOGRAM STATISTICS (mode={used_mode})")
        print("=" * 60)
        print(f"Patches used: {len(dissim_values_fg)}  (FG total: {hist_stats['num_patches_foreground']}/{hist_stats['num_patches_total']})")
        print(f"Mean:   {hist_stats['mean']:.4f}")
        print(f"Median: {hist_stats['median']:.4f}")
        print(f"Std:    {hist_stats['std']:.4f}")
        print(f"P95:    {hist_stats['p95']:.4f}  <- Key metric")
        print(f"P99:    {hist_stats['p99']:.4f}")
        print(f"Tail ratio (>0.3): {hist_stats['tail_ratio']:.4f}")
        print(f"Skewness: {hist_stats['skewness']:.4f}")
        print("=" * 60)

        is_conspec = is_conspecific_threshold(hist_stats)
        print(f"\nPrediction: {'CONSPECIFIC' if is_conspec else 'DIFFERENT SPECIES'}")
        print("(Thresholds need calibration on your data!)")

        # Plot histograms (bars unweighted)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        ax1.hist(dissim_values_fg, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax1.axvline(hist_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {hist_stats['mean']:.3f}")
        ax1.axvline(hist_stats['p95'], color='orange', linestyle='--', linewidth=2, label=f"P95: {hist_stats['p95']:.3f}")
        ax1.set_xlabel('Dissimilarity'); ax1.set_ylabel('Frequency')
        ax1.set_title(f'Used patches (n={len(dissim_values_fg)})'); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.hist(dissim_values_all, bins=50, alpha=0.5, edgecolor='black', color='gray', label='All A patches')
        ax2.hist(dissim_values_fg, bins=50, alpha=0.7, edgecolor='black', color='steelblue', label=f'Used ({used_mode})')
        ax2.set_xlabel('Dissimilarity'); ax2.set_title('All vs Used'); ax2.legend(); ax2.grid(alpha=0.3)

        if len(dissim_values_bg) > 0:
            ax3.hist(dissim_values_bg, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
            bg_mean = float(np.mean(dissim_values_bg))
            ax3.axvline(bg_mean, color='darkred', linestyle='--', linewidth=2, label=f"Mean: {bg_mean:.3f}")
            ax3.set_xlabel('Dissimilarity'); ax3.set_title(f'Background (n={len(dissim_values_bg)})'); ax3.legend(); ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(ranking_subdir, 'ranking_dissimilarity_histogram.png'), dpi=150)
        plt.close()

        # Visualize matches and save CSV
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.imshow(imgA); ax1.axis('off')
        ax2.imshow(imgB); ax2.axis('off')
        ax1.set_title(f'Image A: {len(patches_A_idx)} matched patches')
        ax2.set_title(f'Image B: matched regions')
        cmap = plt.colormaps.get_cmap('viridis')
        colors = [cmap(i / max(1, len(patches_A_idx))) for i in range(len(patches_A_idx))]
        for rank, (idx_A, idx_B, score) in enumerate(zip(patches_A_idx, patches_B_idx, match_scores)):
            xA = (idx_A % grid_size) * patch_size + patch_size // 2
            yA = (idx_A // grid_size) * patch_size + patch_size // 2
            xB = (idx_B % grid_size) * patch_size + patch_size // 2
            yB = (idx_B // grid_size) * patch_size + patch_size // 2
            color = colors[rank]
            ax1.scatter(xA, yA, color=color, s=100, alpha=0.8, edgecolors='white', linewidths=1.5)
            ax1.text(xA, yA, str(rank + 1), color='white', fontsize=8, ha='center', va='center', fontweight='bold')
            ax2.scatter(xB, yB, color=color, s=100, alpha=0.8, edgecolors='white', linewidths=1.5)
            ax2.text(xB, yB, str(rank + 1), color='white', fontsize=8, ha='center', va='center', fontweight='bold')
            fig.add_artist(ConnectionPatch(xyA=(xB, yB), xyB=(xA, yA), coordsA='data', coordsB='data',
                                           axesA=ax2, axesB=ax1, color=color, lw=1.5, alpha=0.7))
        plt.tight_layout()
        plt.savefig(os.path.join(ranking_subdir, 'ranking_attention_matches.png'), dpi=150, bbox_inches='tight')
        plt.close()

        match_csv = os.path.join(ranking_subdir, 'ranking_attention_matches.csv')
        with open(match_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'patch_idx_A', 'patch_idx_B', 'cosine_sim', 'dissimilarity'])
            for rank, (idx_A, idx_B, score) in enumerate(zip(patches_A_idx, patches_B_idx, match_scores)):
                writer.writerow([rank + 1, int(idx_A), int(idx_B), float(score), float(1.0 - score)])
    
    print(f"\n{'=' * 60}\nOUTPUT SUMMARY\n{'=' * 60}")
    print(f"Results directory: {args.output_dir}")
    print(f"Run metadata: run_metadata.yaml")
    print(f"\nShared outputs:")
    print(f"  - Attention overlays in activation_maps/")

    if args.matching_mode in ['similarity', 'both']:
        print(f"\n{'='*60}")
        print(f"DISSIMILARITY RANKING (Attention-based):")
        print(f"{'='*60}")
        print(f"  Directory: dissimilarity_ranking/")
        print(f"  Histogram stats: ranking_histogram_statistics.json")
        print(f"  Histogram plot: ranking_dissimilarity_histogram.png")
        print(f"  Sparse matches: ranking_attention_matches.png")
        print(f"  Match CSV: ranking_attention_matches.csv")
        print(f"  Bidirectional CSVs:")
        print(f"    - foreground_dissimilarities_A_to_B.csv")
        print(f"    - foreground_dissimilarities_B_to_A.csv")
        print(f"    - bidirectional_mutual_matches.csv")
        print(f"    - dissimilarity_matrix_full.npy")

    if args.matching_mode in ['tps_homology', 'both']:
        print(f"\n{'='*60}")
        print(f"DISSIMILARITY TPS HOMOLOGY (Grid-based):")
        print(f"{'='*60}")
        print(f"  Directory: dissimilarity_tps_homology/")
        print(f"  Histogram stats: tps_histogram_statistics.json")
        print(f"  Histogram plot: tps_dissimilarity_histogram.png")
        print(f"  Grid overlay: tps_grid_overlay.png")
        print(f"  Sparse matches: tps_homology_matches.png")
        print(f"  Match CSV: tps_homology_matches.csv")
        print(f"  Per-cell CSV: tps_per_cell_dissimilarities.csv")
        print(f"  Full stats: tps_homology_stats.json")

    print(f"{'=' * 60}\nDone!")

if __name__ == '__main__':
    main()
