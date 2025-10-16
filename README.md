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


![Allosaurus](logo/allosaurus_ubahn_naturkund.jpg)
![DINOSAR logo](logo/DINOSAR_logo.png)


## Cite
If you find this code useful please you cite this github page! 

@misc{VanDam_DINOSAR_2025,
  author       = {Alex R. Van Dam},
  title        = {DINOSAR: DINOv3 Species Auto-Recovery (Zero-Shot AI Enabled Morphological Species Delimitation)},
  year         = {2025},
  month        = {oct},
  version      = {v76},
  howpublished = {\url{https://github.com/alexrvandam/DINOSAR}},
  note         = {GitHub repository. Commit: <commit-hash>},
}

This work relied on the use of oter open githubs that should also be cited:



## License
Distributed under Apache Version 2.0 license:

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

