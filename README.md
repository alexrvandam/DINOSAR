# DINOSAR
## <ins>DINO</ins>v3 <ins>S</ins>pecies <ins>A</ins>uto-<ins>R</ins>ecovery (Zero-Shot AI Enabled Morphological Species Delimitaiton) 

DINOSAR uses self-supervised pre-training for zero-shot learning **DINOv3 ViTs** to compare two specimen photos without any training. It builds foreground-aware, attention-weighted patch similarities and (optionally) a **thin-plate spline (TPS) homologous grid** from COCO-format keypoints to compare like-with-like regions. Outputs include robust dissimilarity histograms, attention overlays, sparse correspondences, and TPS grid overlays (below).

Eg. Colpoptera fusca x Colpoptera maculata
![TPS grid overlay](output/tps_grid_overlay-align-stage-none.png)

Built around `DINOV3_patch_match_v_76_edge_suppression.py`. CLI options cited below come straight from the script.

---

## Features

- **Layer sweep & fusion**: extract tokens from multiple ViT layers (e.g. `-3 -2 -1`) and fuse by mean/median for stability. 
- **Foreground & edge suppression**: threshold by foreground *and* drop N border patch rings; optional erosion; optional attention∩foreground gating. 
- **Two sparse-match flavors**  
  - **Attention**: rank & match by attention importance.  
  - **Demo**: DINOv3-notebook style mutual/ratio matching with optional RANSAC verification. 
- **Multiple histogram/score modes**: `foreground`, `attention_weighted`, `demo_inliers`, `bidirectional`, `topk_mean`.
- **TPS homology mode** *(optional)*: COCO keypoints → consensus shape → TPS warps → homologous grid cells → per-cell dissimilarities + overlays. 
- **Run metadata** saved as YAML for reproducibility. 
---
## Informative sample <--to--> sample, attention dissimilarity mapping
Colpoptera fusca x Colpoptera maculata
![TPS dissim hist](output/tps_dissimilarity_histogram.png)
![TPS dissim hist balanced ](output/tps_dissimilarity_histogram_cov_balanced.png)
![TPS fusca maculata patch match](output/tps_homology_matches--align-stage-none.png)

Colpoptera fusca x Colpoptera fusca
![TPS grid overlay fuscax2](output/tps_grid_overlay_fuscax2.png)
![TPS dissim hist fuscax2](output/tps_dissimilarity_histogram_fuscax2.png)
![TPS dissim hist balanced fuscax2](output/tps_dissimilarity_histogram_cov_balanced_fuscax2.png)
![TPS fuscax2 patch match](output/tps_homology_matches_fuscax2.png)

Colpoptera DINOv3 Attention maps
![imgA_attention](logo/imgA_attention_layer11.png)
![imgB_attention](logo/imgB_attention_layer11.png)


---

## Installation

### 1) Create a conda env (recommended)

```bash
conda env create -f DINOSARv76_environment.yml
conda activate dinosar-v76
```

## Quickstart

```bash
git clone https://github.com/alexrvandam/DINOSAR.git
cd DINOSAR

# (Option A) conda
conda env create -f environment.yml
conda activate dinosar-v76

# (Option B) plain pip
pip install -r requirements.txt
```

Download DINOv3 and put it some place safe and copy its path for later
```bash
~/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

A) Zero-shot species comparison (no keypoints, REMBG foreground)
```bash
python DINOSAR/DINOV3_patch_match_v_76_edge_suppression.py \
  DINOSAR/data/A.jpg DINOSAR/data/B.jpg \
  --dinov3-local-ckpt ~/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --dinov3-arch vits16 \
  --layers -3 -2 -1 --layer-agg median \
  --fg-mode rembg --align-stage none \
  --score-mode foreground \
  --attn-fg-thresh 0.15 \
  --attn-layers 9 10 11 \
  --mask-erode-px 2 \
  --border-patches 2 \
  --matching-mode similarity \
  --output-dir outputs/run_A_vs_B
```

B) With keypoints + TPS homology grid + similarity ranking (“both” mode)
```bash
python DINOSAR/DINOV3_patch_match_v_76_edge_suppression.py \
  DINOSAR/data/A.jpg DINOSAR/data/B.jpg \
  --dinov3-local-ckpt ~/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --dinov3-arch vits16 \
  --layers -3 -2 -1 --layer-agg median \
  --fg-mode rembg --align-stage none \
  --score-mode foreground \
  --attn-fg-thresh 0.15 \
  --attn-layers 9 10 11 \
  --mask-erode-px 2 \
  --border-patches 2 \
  --matching-mode similarity \
  --match-flavor demo \
  --demo-mutual \
  --demo-ratio 0.92 \
  --demo-topn 200 \
  --demo-verify --demo-ransac-reproj 3.0 \
  --output-dir outputs/run_A_vs_B_demo
```

C) With keypoints + TPS homology grid no mask alignment ('none' mode)
```bash
python DINOSAR/DINOV3_patch_match_v_76_edge_suppression.py 'toy_data/col-fusca_2_(L).jpg' 'toy_data/col-fusca_(L).jpg' \
--dinov3-local-ckpt ~/your_path/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
--dinov3-arch vits16   --layers -3 -2 -1   --layer-agg median   --fg-mode rembg \
--align-stage none --score-mode foreground   --attn-fg-thresh 0.15   --attn-layers 9 10 11 \
--mask-erode-px 2   --border-patches 2   --keypoints-json 'toy_data/working_copy_maculata_maculata_kpts_V2.json' \
--matching-mode both   --tps-grid-snap   --tps-grid-cell-size 16 --tps-pad-cells 0
```

## Short Guide
What the key flags do (short guide)

--layers -3 -2 -1 --layer-agg median
Use the last 3 transformer blocks’ [CLS]-replaced patch tokens; aggregate them with median for robustness. Good default.

--fg-mode rembg
Make a foreground mask with rembg (zero-shot background removal). You can swap for thresh, attn, or attn_intersect if you’ve tuned those.

--align-stage none | image | mask | both
Where to apply global alignment if available (from keypoints or zeroshot ECC on masks).
none → skip global warp (often better for TPS, since TPS will handle shape alignment).
image/both → warp image (and mask) to roughly align A→B before token extraction.
mask → only warp masks; images stay as-is.

--attn-layers 9 10 11
Which attention heads to visualize; doesn’t change the tokens used unless you pick --attn-fg for masks.

--mask-erode-px 2 and --border-patches 2
Clean edges by eroding the mask a bit and dropping patches within N patch-widths of the image border.

--matching-mode similarity | tps_homology | both
similarity → build cosine-similarity matrix between patch tokens; produce histograms/overlays/sparse matches.

tps_homology → build a TPS grid from COCO keypoints; compare homologous regions cell-by-cell.
both → do both pipelines.

--tps-grid-cell-size 16 and --tps-grid-snap
Cell size for the TPS consensus grid; snap aligns grid to ViT patch lattice (nice for visual alignment & patch→cell mapping).

--tps-pad-cells N (v76+)
Adds N full cells of border on each side of the TPS grid around the landmark bbox.
0 = tight bbox; 1 = your “+1 cell” border; 2 = wider frame.

--score-mode foreground | attention_weighted | demo_inliers | bidirectional | topk_mean
Which per-patch dissimilarities power the histogram/statistics.
foreground → simple and robust; joint-foreground best-match dissimilarity.
attention_weighted → foreground dissimilarities weighted by attention saliency.
demo_inliers → only the RANSAC-verified “demo” matches (needs enough inliers).
bidirectional → symmetric A→B and B→A best matches.
topk_mean → average of top-k similarities per patch (set --topk).

## Citations
If you find this code useful please you cite this github page! 
You must also cite:


## Liscense
Distributed under Apache2 licsence.



![Allosaurus](allosaurus_ubahn_naturkund.jpg)
![DINOSAR logo](DINOSAR_logo.png)

