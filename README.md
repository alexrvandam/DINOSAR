# DINOSAR
## <ins>DINO</ins>v3 <ins>S</ins>pecies <ins>A</ins>uto-<ins>R</ins>ecovery (Zero-Shot AI Enabled Morphological Species Delimitaiton) 

[DINOSAR uses self-supervised pre-training for zero-shot learning **DINOv3 ViTs** to compare two specimen photos without any training. It builds foreground-aware, attention-weighted patch similarities and (optionally) a **thin-plate spline (TPS) homologous grid** from COCO-format keypoints to compare like-with-like regions. Outputs include robust dissimilarity histograms, attention overlays, sparse correspondences, and TPS grid overlays.](tps_grid_overlay-align-stage-none.png)

> Built around `DINOV3_patch_match_v_72_edge_suppression.py`. CLI options cited below come straight from the script. :contentReference[oaicite:0]{index=0}

---

## Features

- **Layer sweep & fusion**: extract tokens from multiple ViT layers (e.g. `-3 -2 -1`) and fuse by mean/median for stability. :contentReference[oaicite:1]{index=1}
- **Foreground & edge suppression**: threshold by foreground *and* drop N border patch rings; optional erosion; optional attention∩foreground gating. :contentReference[oaicite:2]{index=2}
- **Two sparse-match flavors**  
  - **Attention**: rank & match by attention importance.  
  - **Demo**: DINOv3-notebook style mutual/ratio matching with optional RANSAC verification. :contentReference[oaicite:3]{index=3}
- **Multiple histogram/score modes**: `foreground`, `attention_weighted`, `demo_inliers`, `bidirectional`, `topk_mean`. :contentReference[oaicite:4]{index=4}
- **TPS homology mode** *(optional)*: COCO keypoints → consensus shape → TPS warps → homologous grid cells → per-cell dissimilarities + overlays. :contentReference[oaicite:5]{index=5}
- **Run metadata** saved as YAML for reproducibility. :contentReference[oaicite:6]{index=6}

---

## Installation

### 1) Create a conda env (recommended)

```bash
conda env create -f environment.yml
conda activate dinosar
