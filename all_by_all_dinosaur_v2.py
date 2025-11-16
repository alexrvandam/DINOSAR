#!/usr/bin/env python3
"""
Enhanced all_by_all_dinosaur.py with improved completion checking
"""
import argparse, os, sys, json, shlex, subprocess, itertools, math
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def is_pair_complete(pair_dir: Path, stat_source: str = "auto") -> bool:
    """
    Check if a pair run is complete by verifying the existence of expected output subdirectories.
    
    Args:
        pair_dir: Directory containing the pair run output
        stat_source: Which statistics to check for ('auto', 'ranking', 'foreground', 'tps')
    
    Returns:
        True if the run appears complete, False otherwise
    """
    if not pair_dir.exists():
        return False
    
    # Check for run metadata (should always exist)
    if not (pair_dir / "run_metadata.yaml").exists():
        return False
    
    # Check for at least one valid statistics file based on mode
    stat_json = find_stat_json(pair_dir, stat_source)
    if stat_json is None:
        return False
    
    # Additional checks: verify key output directories exist
    expected_subdirs = {
        'auto': ['dissimilarity_ranking', 'dissimilarity_tps_homology'],  # Either or both
        'ranking': ['dissimilarity_ranking'],
        'tps': ['dissimilarity_tps_homology'],
        'foreground': []  # histogram_statistics.json is in root
    }
    
    # For auto mode, we need at least ONE of the subdirs
    if stat_source == 'auto':
        has_ranking = (pair_dir / 'dissimilarity_ranking').exists()
        has_tps = (pair_dir / 'dissimilarity_tps_homology').exists()
        if not (has_ranking or has_tps):
            return False
    else:
        # For specific modes, check required subdirs
        for subdir in expected_subdirs.get(stat_source, []):
            if not (pair_dir / subdir).exists():
                return False
    
    return True


def find_stat_json(run_dir: Path, stat_source: str = "auto") -> Optional[Path]:
    """
    Locate a statistics JSON file produced by DINOSAR for a single pair run.
    Priority order:
      auto: ranking_histogram_statistics.json -> histogram_statistics.json -> tps_histogram_statistics.json
      ranking: ranking_histogram_statistics.json
      foreground: histogram_statistics.json
      tps: tps_histogram_statistics.json
    """
    candidates_by_mode = {
        "ranking": [run_dir / "dissimilarity_ranking" / "ranking_histogram_statistics.json"],
        "foreground": [run_dir / "histogram_statistics.json"],
        "tps": [run_dir / "dissimilarity_tps_homology" / "tps_histogram_statistics.json"],
        "auto": [
            run_dir / "dissimilarity_ranking" / "ranking_histogram_statistics.json",
            run_dir / "histogram_statistics.json",
            run_dir / "dissimilarity_tps_homology" / "tps_histogram_statistics.json",
        ],
    }
    for p in candidates_by_mode.get(stat_source, candidates_by_mode["auto"]):
        if p.exists():
            return p
    return None

def read_stats(stat_json: Path) -> Dict:
    try:
        with open(stat_json, "r") as f:
            data = json.load(f)
        # normalize keys across modes
        result = {}
        for k in ("mean","median","std","p95","p99","num_cells","num_patches_foreground","P95","P99"):
            if k in data:
                result[k.lower()] = data[k] if isinstance(data[k], (int,float)) else data[k]
        # sometimes numbers are nested; just ensure we have 'mean' and 'p95'
        # (TPS file already uses these names.)
        return result
    except Exception as e:
        print(f"[WARN] Failed to read stats from {stat_json}: {e}")
        return {}

def safe_stem(p: Path) -> str:
    # friendlier axis labels
    stem = p.stem
    return stem[:40] + "…" if len(stem) > 40 else stem

def run_pair(
    i: int, j: int, imgA: Path, imgB: Path,
    script: Path, common_args: List[str],
    run_root: Path, force: bool, stat_source: str,
) -> Tuple[int,int,Optional[float],Optional[float],Path]:
    """
    Run DINOSAR for a single pair and return (i, j, mean, p95, run_dir)
    """
    # pair-specific folder name (order-sensitive so we can do [i,j] and fill [j,i] if symmetric)
    pair_dir = run_root / f"{i:03d}_{imgA.stem}__VS__{j:03d}_{imgB.stem}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Enhanced completion check
    if not force and is_pair_complete(pair_dir, stat_source):
        print(f"[SKIP] ({i},{j}) {imgA.name} vs {imgB.name} - already complete")
        stat_json = find_stat_json(pair_dir, stat_source=stat_source)
        st = read_stats(stat_json)
        return i, j, st.get("mean"), st.get("p95"), pair_dir

    # Build command (explicit --outdir so script won't generate a dated name)
    cmd = [
        sys.executable, str(script),
        str(imgA), str(imgB),
        "--output-dir", str(pair_dir),
    ]
    # Add user's DINOSAR options
    cmd.extend(common_args)

    print(f"[RUN] ({i},{j}) {imgA.name} vs {imgB.name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Pair ({i},{j}) failed: {e}")
        return i, j, None, None, pair_dir

    stat_json = find_stat_json(pair_dir, stat_source=stat_source)
    if stat_json is None:
        print(f"[WARN] No stats found for pair ({i},{j}) in {pair_dir}")
        return i, j, None, None, pair_dir
    st = read_stats(stat_json)
    return i, j, st.get("mean"), st.get("p95"), pair_dir

def heatmap(matrix: np.ndarray, labels: List[str], title: str, outpng: Path, vmin: float = 0.0, vmax: float = 0.2):
    plt.figure(figsize=(max(6, len(labels)*0.25), max(5, len(labels)*0.25)))
    im = plt.imshow(matrix, vmin=vmin, vmax=vmax, cmap="magma_r", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run DINOSAR on all image pairs in a directory.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--glob", type=str, default="*.jpg", help="Glob pattern for images")
    parser.add_argument("--outdir", type=Path, required=True, help="Root output directory")
    parser.add_argument("--dinosar-script", type=Path, required=True, help="Path to DINOSAR script")
    parser.add_argument("--dinosar-opts", type=str, default="", help="Additional DINOSAR arguments (as a single quoted string)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (currently only 1 supported)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if output exists")
    parser.add_argument("--stat-source", type=str, choices=["auto","ranking","foreground","tps"], default="auto", help="Which statistics to use")
    parser.add_argument("--heat-vmin", type=float, default=0.0, help="Heatmap color scale minimum")
    parser.add_argument("--heat-vmax", type=float, default=0.2, help="Heatmap color scale maximum")
    
    args = parser.parse_args()
    
    # Find all images
    images = sorted(args.images_dir.glob(args.glob))
    if len(images) == 0:
        print(f"No images found in {args.images_dir} with pattern {args.glob}")
        sys.exit(1)
    
    print(f"Found {len(images)} images")
    print(f"Output directory: {args.outdir}")
    print(f"Resume mode: {'DISABLED (--force)' if args.force else 'ENABLED'}")
    
    # Parse DINOSAR options
    common_args = shlex.split(args.dinosar_opts) if args.dinosar_opts else []
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate all pairs (upper triangle including diagonal)
    n = len(images)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    
    print(f"Total pairs to process: {len(pairs)}")
    
    # Count already complete pairs
    completed = sum(1 for i, j in pairs 
                   if is_pair_complete(args.outdir / f"{i:03d}_{images[i].stem}__VS__{j:03d}_{images[j].stem}", 
                                      args.stat_source))
    
    if completed > 0 and not args.force:
        print(f"Already completed: {completed}/{len(pairs)} pairs")
        print(f"Remaining: {len(pairs) - completed} pairs")
    
    # Run all pairs
    results = []
    for idx, (i, j) in enumerate(pairs, 1):
        print(f"\n[{idx}/{len(pairs)}] Processing pair ({i},{j})")
        res = run_pair(i, j, images[i], images[j], args.dinosar_script, 
                      common_args, args.outdir, args.force, args.stat_source)
        results.append(res)
    
    # Build distance matrices
    mean_matrix = np.full((n, n), np.nan)
    p95_matrix = np.full((n, n), np.nan)
    
    for i, j, mean_val, p95_val, _ in results:
        if mean_val is not None:
            mean_matrix[i, j] = mean_val
            mean_matrix[j, i] = mean_val
        if p95_val is not None:
            p95_matrix[i, j] = p95_val
            p95_matrix[j, i] = p95_val
    
    # Save matrices
    np.save(args.outdir / "distance_matrix_mean.npy", mean_matrix)
    np.save(args.outdir / "distance_matrix_p95.npy", p95_matrix)
    
    # Create heatmaps
    labels = [safe_stem(img) for img in images]
    heatmap(mean_matrix, labels, "Mean Dissimilarity", 
           args.outdir / "heatmap_mean.png", vmin=args.heat_vmin, vmax=args.heat_vmax)
    heatmap(p95_matrix, labels, "P95 Dissimilarity",
           args.outdir / "heatmap_p95.png", vmin=args.heat_vmin, vmax=args.heat_vmax)
    
    # Create summary CSV
    summary_data = []
    for i, j, mean_val, p95_val, pair_dir in results:
        summary_data.append({
            'image_A_index': i,
            'image_B_index': j,
            'image_A': images[i].name,
            'image_B': images[j].name,
            'mean_dissimilarity': mean_val,
            'p95_dissimilarity': p95_val,
            'output_dir': pair_dir.name
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(args.outdir / "all_pairs_summary.csv", index=False)
    
    print(f"\n✓ All pairs completed!")
    print(f"  Distance matrices: distance_matrix_mean.npy, distance_matrix_p95.npy")
    print(f"  Heatmaps: heatmap_mean.png, heatmap_p95.png")
    print(f"  Summary: all_pairs_summary.csv")
    
if __name__ == "__main__":
    main()
