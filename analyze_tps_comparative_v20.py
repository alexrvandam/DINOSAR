#!/usr/bin/env python3
"""
Comprehensive Comparative Statistical Analysis for Morphology-Aware Alignment
Compares BOTH methods:
  1. dissimilarity_ranking (zero-shot, no morphology)
  2. dissimilarity_tps_homology (TPS morphology-aware)
  
Analyzes:
  - Same-specimen pairs (different photos of same individual)
  - Intra-specific pairs (different individuals, same species)
  - Inter-specific pairs (different species)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp, wilcoxon
from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
sns.set_palette("Set2")


def _extract_intra_by_species(stats_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    From a long 'stats_df' (one row per image pair), keep only intra-specific rows
    (comparison_type == 'intra-specific' and species_A == species_B) and compute
    per-species distribution + summary stats for the selected metric.
    Returns a tidy table with one row per species.
    """
    # Guard: common column names
    needed = {'species_A', 'species_B', 'comparison_type', metric_col}
    missing = [c for c in needed if c not in stats_df.columns]
    if missing:
        raise ValueError(f"stats_df missing columns: {missing} (has {list(stats_df.columns)[:12]}...)")

    df = stats_df.copy()
    df = df[(df['comparison_type'] == 'intra-specific') & (df['species_A'] == df['species_B'])]
    if df.empty:
        return pd.DataFrame(columns=[
            'species','n_pairs','mean','median','p95','std','mad','cv','values'
        ])

    # robust MAD
    def mad(x):
        m = np.median(x)
        return np.median(np.abs(x - m))

    grp = df.groupby('species_A')[metric_col].agg(
        n_pairs='count',
        mean='mean',
        median='median',
        p95=lambda x: np.percentile(x, 95),
        std='std',
        mad=mad
    ).reset_index().rename(columns={'species_A':'species'})

    # Coefficient of variation (guard 0-division)
    grp['cv'] = grp['std'] / grp['mean'].replace(0, np.nan)

    # Keep raw values list for plotting violins
    vals = (df.groupby('species_A')[metric_col]
              .apply(lambda x: list(x.astype(float)))
              .rename('values').reset_index())
    grp = grp.merge(vals, left_on='species', right_on='species_A', how='left').drop(columns=['species_A'])

    # Sort for nicer plotting
    grp = grp.sort_values('median').reset_index(drop=True)
    return grp


def _flag_super_variable(spec_table: pd.DataFrame,
                         inter_all: np.ndarray,
                         metric_name: str,
                         method_label: str,
                         iqr_k: float = 1.5,
                         inter_q: float = 25.0) -> pd.DataFrame:
    """
    Add boolean flags for “super variable” species using two rules:
      R1 (within-species rule): CV above (median + 1.5*IQR) of species CVs.
      R2 (between-species proximity rule): species P95 >= inter-specific Q25.
    Returns a copy with columns:
      ['flag_cv_outlier','flag_p95_vs_inter','flag_super_variable']
    """
    tab = spec_table.copy()
    # Rule 1: CV outliers by IQR fence across species
    cv_med = np.nanmedian(tab['cv'])
    cv_q1  = np.nanpercentile(tab['cv'], 25)
    cv_q3  = np.nanpercentile(tab['cv'], 75)
    cv_iqr = cv_q3 - cv_q1
    cv_upper_fence = cv_med + iqr_k * cv_iqr
    tab['flag_cv_outlier'] = tab['cv'] > cv_upper_fence

    # Rule 2: species p95 overlapping low end of inter-specific
    inter_qcut = np.percentile(inter_all, inter_q) if len(inter_all) else np.nan
    tab['flag_p95_vs_inter'] = tab['p95'] >= inter_qcut

    tab['flag_super_variable'] = tab[['flag_cv_outlier','flag_p95_vs_inter']].any(axis=1)

    print(f"[{method_label}] CV fence (median + {iqr_k}*IQR): {cv_upper_fence:.4f} | "
          f"Inter {inter_q:.0f}th percentile: {inter_qcut:.4f}")
    n_flag = int(tab['flag_super_variable'].sum())
    print(f"[{method_label}] Super-variable species flagged: {n_flag}/{len(tab)}")
    return tab


def compute_intra_per_species_tables(ranking_stats: pd.DataFrame,
                                     tps_stats: pd.DataFrame,
                                     metric: str = 'p95') -> dict:
    """
    Build per-species tables for RANKING and TPS, plus return inter-specific arrays
    for each method to support Rule 2 flagging.
    """
    metric_col_rank = f"{metric}" if metric in ranking_stats.columns else f"{metric}_ranking"
    metric_col_tps  = f"{metric}" if metric in tps_stats.columns else f"{metric}_tps"

    # Extract inter-specific arrays for Rule 2
    inter_rank = ranking_stats[(ranking_stats['comparison_type']=='inter-specific')][metric_col_rank].values
    inter_tps  = tps_stats[(tps_stats['comparison_type']=='inter-specific')][metric_col_tps].values

    rank_tab = _extract_intra_by_species(ranking_stats, metric_col_rank)
    tps_tab  = _extract_intra_by_species(tps_stats, metric_col_tps)

    # Flagging
    rank_tab = _flag_super_variable(rank_tab, inter_rank, metric.upper(), "RANKING")
    tps_tab  = _flag_super_variable(tps_tab, inter_tps, metric.upper(), "TPS")

    return {
        'rank_table': rank_tab,
        'tps_table' : tps_tab,
        'inter_rank': inter_rank,
        'inter_tps' : inter_tps,
        'metric'    : metric
    }


def plot_intra_species_violins(rank_tab: pd.DataFrame,
                               tps_tab: pd.DataFrame,
                               outdir: Path,
                               metric: str = 'p95',
                               top_n: int = None):
    """
    Side-by-side violins per species (RANKING on left panel, TPS on right).
    Species sorted by within-species median; optional top_n to keep only the first N species.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    title_metric = metric.upper()

    # Harmonize species ordering by RANKING medians (fallback TPS if empty)
    base = rank_tab if not rank_tab.empty else tps_tab
    order = list(base.sort_values('median')['species'])
    if top_n:
        order = order[:top_n]

    def _panel(ax, table, label):
        sub = table[table['species'].isin(order)]
        data = [sub.loc[sub['species']==sp, 'values'].values[0] for sp in order]
        parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.8)
        ax.set_xticks(range(1, len(order)+1))
        ax.set_xticklabels(order, rotation=90, fontsize=8)
        ax.set_ylabel(f"{title_metric} dissimilarity")
        ax.set_title(f"{label} — Intra-specific variation ({len(sub)} spp.)")

        # Overplot sample sizes
        for i, sp in enumerate(order, start=1):
            n = int(sub.loc[sub['species']==sp, 'n_pairs'])
            ax.text(i, ax.get_ylim()[0], f"n={n}", ha='center', va='bottom', fontsize=7, rotation=90)

    fig, axs = plt.subplots(1, 2, figsize=(max(12, len(order)*0.4), 6), constrained_layout=True)
    _panel(axs[0], rank_tab, "RANKING")
    _panel(axs[1], tps_tab,  "TPS")
    fig.suptitle(f"Intra-specific per-species variation — {title_metric}", fontsize=14, fontweight='bold')
    fig.savefig(outdir / f"intra_species_violins_{metric}.png", dpi=150)
    plt.close(fig)


def plot_intra_species_cv_bars(rank_tab: pd.DataFrame,
                               tps_tab: pd.DataFrame,
                               outdir: Path,
                               metric: str = 'p95',
                               top_n: int = None):
    """
    Bar chart of CV per species for both methods, aligned by species.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Align by species union
    species = sorted(set(rank_tab['species']).union(set(tps_tab['species'])))
    if top_n:
        species = species[:top_n]

    r_map = dict(zip(rank_tab['species'], rank_tab['cv']))
    t_map = dict(zip(tps_tab['species'],  tps_tab['cv']))

    r_vals = [r_map.get(s, np.nan) for s in species]
    t_vals = [t_map.get(s, np.nan) for s in species]

    x = np.arange(len(species))
    w = 0.42

    fig, ax = plt.subplots(figsize=(max(12, len(species)*0.35), 5))
    ax.bar(x - w/2, r_vals, width=w, alpha=0.8, label='RANKING CV')
    ax.bar(x + w/2, t_vals, width=w, alpha=0.8, label='TPS CV')
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=90, fontsize=8)
    ax.set_ylabel('Coefficient of Variation (std/mean)')
    ax.set_title(f"Intra-specific CV per species — {metric.upper()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"intra_species_cv_{metric}.png", dpi=150)
    plt.close(fig)


def save_intra_species_tables(rank_tab: pd.DataFrame,
                              tps_tab: pd.DataFrame,
                              outdir: Path,
                              metric: str = 'p95'):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rank_csv = outdir / f"intra_species_stats_RANKING_{metric}.csv"
    tps_csv  = outdir / f"intra_species_stats_TPS_{metric}.csv"
    rank_tab.to_csv(rank_csv, index=False)
    tps_tab.to_csv(tps_csv, index=False)
    print(f"[SAVE] {rank_csv}")
    print(f"[SAVE] {tps_csv}")

    flagged_txt = outdir / f"intra_species_super_variable_{metric}.txt"
    with open(flagged_txt, "w") as f:
        f.write("=== Super-variable species (any rule) ===\n\n")
        def _dump(label, tab):
            flagged = tab[tab['flag_super_variable']]
            f.write(f"[{label}] {len(flagged)}/{len(tab)} flagged\n")
            for _, r in flagged.iterrows():
                f.write(f" - {r['species']} | n={int(r['n_pairs'])} | p95={r['p95']:.3f} | CV={r['cv']:.3f} "
                        f"| cv_outlier={bool(r['flag_cv_outlier'])} | p95_vs_inter={bool(r['flag_p95_vs_inter'])}\n")
            f.write("\n")
        _dump("RANKING", rank_tab)
        _dump("TPS",     tps_tab)
    print(f"[SAVE] {flagged_txt}")





# --- THRESHOLD + ROC UTILITIES -----------------------------------------------
import numpy as np
import pandas as pd

def _safe_roc_curve(y_true, scores):
    """Return fpr, tpr, thresholds without requiring sklearn."""
    # Binary labels 1=inter, 0=intra; scores = dissimilarity (higher = more inter)
    # Sort by score desc
    order = np.argsort(scores)[::-1]
    y = np.asarray(y_true)[order]
    s = np.asarray(scores)[order]

    # Unique thresholds
    thresholds, idx = np.unique(s, return_index=True)
    thresholds = thresholds[::-1]  # from high to low
    # Append +inf and -inf to mimic sklearn behavior (optional)
    thresholds = np.r_[np.inf, thresholds, -np.inf]

    P = (y == 1).sum()
    N = (y == 0).sum()
    tp = fp = 0
    tpr = [0.0]
    fpr = [0.0]
    j = 0
    for thr in thresholds[1:-1]:
        # advance pointer while s >= thr
        while j < len(s) and s[j] >= thr:
            if y[j] == 1: tp += 1
            else:         fp += 1
            j += 1
        tpr.append(tp / P if P else 0.0)
        fpr.append(fp / N if N else 0.0)
    tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr), np.array(tpr), thresholds

def auroc_from_roc(fpr, tpr):
    # trapezoid rule, assume fpr increasing
    order = np.argsort(fpr)
    return np.trapz(tpr[order], fpr[order])

def intrap95_threshold(intra_vals):
    return np.nanpercentile(intra_vals, 95.0)

def youdens_j_threshold(y_true, scores):
    fpr, tpr, thr = _safe_roc_curve(y_true, scores)
    j = tpr - fpr
    k = np.argmax(j)
    return float(thr[k]), float(tpr[k]), float(fpr[k]), float(j[k]), fpr, tpr, thr

def evaluate_at_threshold(y_true, scores, thr):
    """Return TPR, FPR, precision, recall, F1 at thr (>= thr => predict inter)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    P = tp + fn
    N = tn + fp
    tpr = tp / P if P else 0.0
    fpr = fp / N if N else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tpr
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, TPR=tpr, FPR=fpr, precision=prec, recall=rec, F1=f1)

def classify_pairs_dataframe(df_pairs, scores_col, thr, out_csv_path, label_col='comparison_type'):
    """
    df_pairs must have: image_A, image_B, comparison_type ('intra-specific' or 'inter-specific'),
    and the metric column (scores_col).
    Writes per-pair decisions:
      decision = 'split' (score>=thr) or 'lump-risk' (score<thr)
      correct = whether decision matches ground truth
    """
    df = df_pairs.copy()
    df = df[df[label_col].isin(['intra-specific','inter-specific'])].copy()
    df['score'] = df[scores_col].astype(float)
    df['decision'] = np.where(df['score'] >= thr, 'split', 'lump-risk')
    df['truth_is_inter'] = (df[label_col] == 'inter-specific').astype(int)
    df['correct'] = np.where(((df['decision'] == 'split') & (df['truth_is_inter'] == 1)) |
                             ((df['decision'] == 'lump-risk') & (df['truth_is_inter'] == 0)), 1, 0)
    cols = ['image_A','image_B','species_A','species_B','score','decision','comparison_type','correct']
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_csv_path, index=False)
    return df

class ComparativeDissimilarityAnalyzer:
    """Analyzes and compares ranking vs TPS methods from DINOSAR all-by-all runs."""
    
    def __init__(self, observations_csv: Path, results_dir: Path, output_dir: Path):
        """
        Args:
            observations_csv: Path to observations_photos.csv with species labels
            results_dir: Path to all_by_all output directory with results
            output_dir: Path where analysis outputs will be saved
        """
        self.observations_csv = observations_csv
        self.results_dir = results_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.obs_df = pd.read_csv(observations_csv)
        self.pairs_df = None
        self.species_map = {}
        self.specimen_map = {}  # Maps image to observation_id (specimen ID)
        self.specimen_labels = {}  # Maps specimen to nice label
        
        # Will hold dataframes for each method
        self.ranking_stats = None
        self.tps_stats = None
        
    def map_images_to_taxonomy(self):
        """Create mapping from image filenames to species and specimen IDs."""
        print("\n" + "="*80)
        print("STEP 1: Mapping Images to Taxonomy and Specimens")
        print("="*80)
        
        # Track specimens per species
        species_specimens = defaultdict(list)
        
        # Extract filename from saved_as path and map to species + specimen
        for _, row in self.obs_df.iterrows():
            base_filename = Path(row['saved_as']).name
            species = row['taxon_name']
            specimen_id = row['observation_id']
            
            # Map the base filename
            self.species_map[base_filename] = species
            self.specimen_map[base_filename] = specimen_id
            
            # Track this specimen for this species
            if specimen_id not in species_specimens[species]:
                species_specimens[species].append(specimen_id)
            
            # Create variations with common suffixes and extensions
            stem = Path(base_filename).stem
            for suffix in ['_lat', '_dors', '_vent', '_lateral', '_dorsal', '_ventral']:
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    variant = stem + suffix + ext
                    self.species_map[variant] = species
                    self.specimen_map[variant] = specimen_id
        
        # Create nice specimen labels (e.g., "Colpoptera fusca-1", "Colpoptera fusca-2")
        for species, specimens in species_specimens.items():
            if len(specimens) == 1:
                # Only one specimen, no need for numbering
                self.specimen_labels[specimens[0]] = species
            else:
                # Multiple specimens, add numbering
                for i, spec_id in enumerate(sorted(specimens), 1):
                    self.specimen_labels[spec_id] = f"{species}-{i}"
        
        print(f"Ã¢Å“â€œ Mapped {len(self.species_map)} image variations to species")
        print(f"Ã¢Å“â€œ Identified {len(set(self.specimen_map.values()))} unique specimens")
        
        # Analyze taxonomic levels
        unique_taxa = set(self.species_map.values())
        
        def get_taxonomic_level(taxon_name):
            if pd.isna(taxon_name):
                return 'unknown'
            parts = str(taxon_name).strip().split()
            if str(taxon_name).endswith('ini'):
                return 'tribe'
            elif len(parts) == 1:
                return 'genus'
            elif len(parts) >= 2:
                return 'species'
            return 'unknown'
        
        taxa_by_level = defaultdict(list)
        for taxon in unique_taxa:
            level = get_taxonomic_level(taxon)
            taxa_by_level[level].append(taxon)
        
        print(f"\nÃ°Å¸â€Â¬ Taxonomic Level Summary:")
        print(f"  Total unique taxa: {len(unique_taxa)}")
        for level in ['species', 'genus', 'tribe', 'unknown']:
            if taxa_by_level[level]:
                print(f"  {level.capitalize()}-level: {len(taxa_by_level[level])} taxa")
        
        # Count specimens per species
        species_counts = defaultdict(int)
        for specimen_id in set(self.specimen_map.values()):
            species = self.specimen_labels.get(specimen_id, "Unknown")
            if '-' in species:
                species = species.rsplit('-', 1)[0]
            species_counts[species] += 1
        
        print(f"\nÃ°Å¸â€œÅ  Specimens per species (top 20):")
        for species, count in sorted(species_counts.items(), key=lambda x: -x[1])[:20]:
            level = get_taxonomic_level(species)
            marker = "Ã°Å¸â€Â¬" if level == 'species' else "Ã¢Å¡Â Ã¯Â¸Â"
            print(f"  {marker} {species}: {count} specimens")
        
        non_species = taxa_by_level['genus'] + taxa_by_level['tribe']
        if non_species:
            print(f"\n  Ã¢Å¡Â Ã¯Â¸Â  WARNING: {len(non_species)} taxa are genus/tribe-level")
            print("  These will NOT be counted as intra-specific pairs")
            print(f"  Examples: {', '.join(non_species[:5])}")
        
        return self.species_map, self.specimen_map
        
    def load_pairwise_results(self):
        """Load the all_pairs_summary.csv file and classify comparisons."""
        print("\n" + "="*80)
        print("STEP 2: Loading and Classifying Pairwise Results")
        print("="*80)
        
        summary_path = self.results_dir / "all_pairs_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Could not find {summary_path}")
        
        self.pairs_df = pd.read_csv(summary_path)
        print(f"Ã¢Å“â€œ Loaded {len(self.pairs_df)} pairwise comparisons")
        
        # Add taxonomy and specimen information
        self.pairs_df['species_A'] = self.pairs_df['image_A'].map(self.species_map)
        self.pairs_df['species_B'] = self.pairs_df['image_B'].map(self.species_map)
        self.pairs_df['specimen_A'] = self.pairs_df['image_A'].map(self.specimen_map)
        self.pairs_df['specimen_B'] = self.pairs_df['image_B'].map(self.specimen_map)
        self.pairs_df['specimen_label_A'] = self.pairs_df['specimen_A'].map(self.specimen_labels)
        self.pairs_df['specimen_label_B'] = self.pairs_df['specimen_B'].map(self.specimen_labels)
        
        # Check taxonomic level
        def is_species_level(taxon_name):
            if pd.isna(taxon_name):
                return False
            parts = str(taxon_name).strip().split()
            if len(parts) < 2:
                return False
            if str(taxon_name).endswith('ini'):
                return False
            return True
        
        self.pairs_df['species_level_A'] = self.pairs_df['species_A'].apply(is_species_level)
        self.pairs_df['species_level_B'] = self.pairs_df['species_B'].apply(is_species_level)
        
        # Classify comparison type
        def classify_comparison(row):
            # Same specimen (different photos of same individual)
            if row['specimen_A'] == row['specimen_B']:
                return 'same-specimen'
            
            # Check if both are species-level
            if not (row['species_level_A'] and row['species_level_B']):
                return 'inter-specific'
            
            # Both species-level: check if same species
            if row['species_A'] == row['species_B']:
                return 'intra-specific'
            else:
                return 'inter-specific'
        
        self.pairs_df['comparison_type'] = self.pairs_df.apply(classify_comparison, axis=1)
        
        # Count comparison types BEFORE filtering
        type_counts_before = self.pairs_df['comparison_type'].value_counts()
        total_before = len(self.pairs_df)
        
        print(f"\nÃ°Å¸â€œÅ  Comparison type distribution (BEFORE filtering incomplete IDs):")
        print(f"  Same-specimen (different photos): {type_counts_before.get('same-specimen', 0)}")
        print(f"  Intra-specific (different individuals, same species): {type_counts_before.get('intra-specific', 0)}")
        print(f"  Inter-specific (different species): {type_counts_before.get('inter-specific', 0)}")
        print(f"  Total pairs: {total_before}")
        
        # FILTER OUT INCOMPLETE TAXONOMIC IDs
        print(f"\nÃ°Å¸â€Â Filtering out incomplete taxonomic identifications...")
        print(f"   (Excluding: tribe-level, genus-only, and other incomplete IDs)")
        
        def has_incomplete_id(taxon_name):
            """Check if taxonomic ID is incomplete (tribe, genus-only, etc.)."""
            if pd.isna(taxon_name):
                return True
            
            taxon_str = str(taxon_name).strip()
            
            # Empty or just whitespace
            if not taxon_str:
                return True
            
            # Tribe-level (ends in -ini, -idae, -inae, etc.)
            if taxon_str.endswith('ini') or taxon_str.endswith('idae') or taxon_str.endswith('inae'):
                return True
            
            # Genus-only (single word, no species epithet)
            parts = taxon_str.split()
            if len(parts) < 2:
                return True
            
            # Check for common incomplete indicators
            incomplete_indicators = ['sp.', 'spp.', 'sp', 'indet', 'undet', 'undetermined']
            if any(indicator in taxon_str.lower() for indicator in incomplete_indicators):
                return True
            
            return False
        
        # Mark pairs with incomplete IDs
        self.pairs_df['incomplete_A'] = self.pairs_df['species_A'].apply(has_incomplete_id)
        self.pairs_df['incomplete_B'] = self.pairs_df['species_B'].apply(has_incomplete_id)
        self.pairs_df['has_incomplete'] = self.pairs_df['incomplete_A'] | self.pairs_df['incomplete_B']
        
        # Report what will be excluded
        excluded_pairs = self.pairs_df[self.pairs_df['has_incomplete']]
        if len(excluded_pairs) > 0:
            print(f"\nÃ¢Å¡Â Ã¯Â¸Â  Found {len(excluded_pairs)} pairs with incomplete IDs:")
            
            # Count by incomplete ID type
            incomplete_ids = []
            for _, row in excluded_pairs.iterrows():
                if row['incomplete_A']:
                    incomplete_ids.append(row['species_A'])
                if row['incomplete_B']:
                    incomplete_ids.append(row['species_B'])
            
            incomplete_counts = pd.Series(incomplete_ids).value_counts()
            print(f"\n   Incomplete IDs found:")
            for incomplete_id, count in incomplete_counts.items():
                print(f"     Ã¢â‚¬Â¢ {incomplete_id}: {count} pairs excluded")
            
            # Show example pairs being excluded
            print(f"\n   Example excluded pairs:")
            for i, (_, row) in enumerate(excluded_pairs.head(5).iterrows()):
                print(f"     {i+1}. {row['species_A']} <-> {row['species_B']}")
        
        # FILTER: Keep only pairs with complete species-level IDs
        self.pairs_df = self.pairs_df[~self.pairs_df['has_incomplete']].copy()
        
        print(f"\nÃ¢Å“â€œ Filtered to {len(self.pairs_df)} pairs with complete species-level IDs")
        print(f"  Excluded: {total_before - len(self.pairs_df)} pairs with incomplete IDs")
        
        # Count comparison types AFTER filtering
        type_counts = self.pairs_df['comparison_type'].value_counts()
        print(f"\nÃ°Å¸â€œÅ  Comparison type distribution (AFTER filtering):")
        print(f"  Same-specimen (different photos): {type_counts.get('same-specimen', 0)}")
        print(f"  Intra-specific (different individuals, same species): {type_counts.get('intra-specific', 0)}")
        print(f"  Inter-specific (different species): {type_counts.get('inter-specific', 0)}")
        print(f"  Total pairs: {len(self.pairs_df)}")
        
        # Show which species are included
        included_species = set()
        for _, row in self.pairs_df.iterrows():
            if pd.notna(row['species_A']):
                included_species.add(row['species_A'])
            if pd.notna(row['species_B']):
                included_species.add(row['species_B'])
        
        if len(included_species) > 0:
            print(f"\nÃ¢Å“â€œ Analysis includes {len(included_species)} species with complete IDs:")
            for species in sorted(included_species)[:10]:  # Show first 10
                print(f"    Ã¢â‚¬Â¢ {species}")
            if len(included_species) > 10:
                print(f"    ... and {len(included_species) - 10} more")
        
        # Check matching success
        matched_a = self.pairs_df['species_A'].notna().sum()
        matched_b = self.pairs_df['species_B'].notna().sum()
        print(f"\nÃ°Å¸â€Â Species Matching Results:")
        print(f"  Image A: {matched_a}/{len(self.pairs_df)} matched ({100*matched_a/len(self.pairs_df):.1f}%)")
        print(f"  Image B: {matched_b}/{len(self.pairs_df)} matched ({100*matched_b/len(self.pairs_df):.1f}%)")
        
        return self.pairs_df
    
    def extract_statistics_by_method(self):
        """Extract statistics from BOTH methods (ranking and TPS)."""
        print("\n" + "="*80)
        print("STEP 3: Extracting Statistics for BOTH Methods")
        print("="*80)
        
        ranking_data = []
        tps_data = []
        
        missing_ranking = 0
        missing_tps = 0
        
        for _, row in self.pairs_df.iterrows():
            pair_dir = self.results_dir / row['output_dir']
            
            base_entry = {
                'image_A': row['image_A'],
                'image_B': row['image_B'],
                'species_A': row['species_A'],
                'species_B': row['species_B'],
                'specimen_label_A': row['specimen_label_A'],
                'specimen_label_B': row['specimen_label_B'],
                'comparison_type': row['comparison_type'],
            }
            
            # Extract RANKING statistics
            ranking_json = pair_dir / "dissimilarity_ranking" / "ranking_histogram_statistics.json"
            if ranking_json.exists():
                try:
                    with open(ranking_json, 'r') as f:
                        stats = json.load(f)
                    entry = base_entry.copy()
                    entry.update({
                        'mean': stats.get('mean', np.nan),
                        'median': stats.get('median', np.nan),
                        'std': stats.get('std', np.nan),
                        'p95': stats.get('p95', stats.get('P95', np.nan)),
                        'p99': stats.get('p99', stats.get('P99', np.nan)),
                        'num_patches': stats.get('num_patches_foreground', np.nan),
                    })
                    ranking_data.append(entry)
                except Exception as e:
                    print(f"Warning: Could not read ranking stats from {ranking_json}: {e}")
                    missing_ranking += 1
            else:
                missing_ranking += 1
            
            # Extract TPS statistics
            tps_json = pair_dir / "dissimilarity_tps_homology" / "tps_histogram_statistics.json"
            if tps_json.exists():
                try:
                    with open(tps_json, 'r') as f:
                        stats = json.load(f)
                    entry = base_entry.copy()
                    entry.update({
                        'mean': stats.get('mean', np.nan),
                        'median': stats.get('median', np.nan),
                        'std': stats.get('std', np.nan),
                        'p95': stats.get('p95', stats.get('P95', np.nan)),
                        'p99': stats.get('p99', stats.get('P99', np.nan)),
                        'num_cells': stats.get('num_cells', np.nan),
                    })
                    tps_data.append(entry)
                except Exception as e:
                    print(f"Warning: Could not read TPS stats from {tps_json}: {e}")
                    missing_tps += 1
            else:
                missing_tps += 1
        
        self.ranking_stats = pd.DataFrame(ranking_data)
        self.tps_stats = pd.DataFrame(tps_data)
        
        print(f"\nÃ¢Å“â€œ RANKING METHOD (Zero-shot):")
        print(f"  Successfully extracted: {len(self.ranking_stats)} pairs")
        if missing_ranking > 0:
            print(f"  Missing data: {missing_ranking} pairs")
        
        print(f"\nÃ¢Å“â€œ TPS METHOD (Morphology-aware):")
        print(f"  Successfully extracted: {len(self.tps_stats)} pairs")
        if missing_tps > 0:
            print(f"  Missing data: {missing_tps} pairs")
        
        # Save detailed statistics for both methods
        ranking_csv = self.output_dir / "detailed_statistics_RANKING.csv"
        tps_csv = self.output_dir / "detailed_statistics_TPS.csv"
        
        self.ranking_stats.to_csv(ranking_csv, index=False)
        self.tps_stats.to_csv(tps_csv, index=False)
        
        print(f"\nÃ¢Å“â€œ Saved detailed statistics:")
        print(f"  Ranking: {ranking_csv}")
        print(f"  TPS: {tps_csv}")
        
        return self.ranking_stats, self.tps_stats
    
    def compute_summary_statistics(self):
        """Compute summary statistics for both methods and all comparison types."""
        print("\n" + "="*80)
        print("STEP 4: Computing Summary Statistics & Distribution Shapes")
        print("="*80)
        
        summary_rows = []
        tail_metrics_rows = []
        
        for method_name, df in [('RANKING', self.ranking_stats), ('TPS', self.tps_stats)]:
            print(f"\n{'='*60}")
            print(f"{method_name} METHOD STATISTICS")
            print(f"{'='*60}")
            
            for comp_type in ['same-specimen', 'intra-specific', 'inter-specific']:
                subset = df[df['comparison_type'] == comp_type]
                
                if len(subset) == 0:
                    print(f"\n{comp_type}: NO DATA")
                    continue
                
                print(f"\n{comp_type.upper()} (n={len(subset)}):")
                print("-" * 60)
                
                # Calculate tail metrics (P95 - Mean and P95 - Median)
                mean_vals = subset['mean'].dropna()
                median_vals = subset['median'].dropna()
                p95_vals = subset['p95'].dropna()
                
                if len(mean_vals) > 0 and len(p95_vals) > 0:
                    # Match indices
                    common_idx = mean_vals.index.intersection(p95_vals.index)
                    if len(common_idx) > 0:
                        tail_from_mean = p95_vals[common_idx] - mean_vals[common_idx]
                        tail_from_median = p95_vals[common_idx] - median_vals[common_idx]
                        
                        # Calculate skewness
                        from scipy.stats import skew
                        mean_skewness = skew(mean_vals) if len(mean_vals) > 3 else np.nan
                        
                        print(f"  TAIL METRICS (Distribution Shape):")
                        print(f"    P95 - Mean: {tail_from_mean.mean():.4f} Ã‚Â± {tail_from_mean.std():.4f}")
                        print(f"    P95 - Median: {tail_from_median.mean():.4f} Ã‚Â± {tail_from_median.std():.4f}")
                        print(f"    Skewness (mean dist): {mean_skewness:.3f}")
                        
                        tail_metrics_rows.append({
                            'method': method_name,
                            'comparison_type': comp_type,
                            'n': len(tail_from_mean),
                            'tail_from_mean_avg': tail_from_mean.mean(),
                            'tail_from_mean_std': tail_from_mean.std(),
                            'tail_from_median_avg': tail_from_median.mean(),
                            'tail_from_median_std': tail_from_median.std(),
                            'skewness': mean_skewness
                        })
                
                for metric in ['mean', 'median', 'p95']:
                    values = subset[metric].dropna()
                    if len(values) == 0:
                        continue
                    
                    print(f"  {metric.upper()} Dissimilarity:")
                    print(f"    Mean Ã‚Â± SD: {values.mean():.4f} Ã‚Â± {values.std():.4f}")
                    print(f"    Median [Q1-Q3]: {values.median():.4f} [{values.quantile(0.25):.4f}-{values.quantile(0.75):.4f}]")
                    print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
                    
                    summary_rows.append({
                        'method': method_name,
                        'comparison_type': comp_type,
                        'metric': metric,
                        'n': len(values),
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'min': values.min(),
                        'max': values.max()
                    })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = self.output_dir / "summary_statistics_BOTH_METHODS.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nÃ¢Å“â€œ Saved summary statistics to {summary_csv}")
        
        # Save tail metrics
        if tail_metrics_rows:
            tail_df = pd.DataFrame(tail_metrics_rows)
            tail_csv = self.output_dir / "tail_metrics_distribution_shape.csv"
            tail_df.to_csv(tail_csv, index=False)
            print(f"Ã¢Å“â€œ Saved tail/shape metrics to {tail_csv}")
        
        return summary_df
    
    def compare_methods_statistically(self):
        """Statistical comparison between RANKING and TPS methods."""
        print("\n" + "="*80)
        print("STEP 5: Statistical Comparison Between Methods")
        print("="*80)
        
        # Merge dataframes on comparison pairs
        merged = pd.merge(
            self.ranking_stats,
            self.tps_stats,
            on=['image_A', 'image_B', 'comparison_type'],
            suffixes=('_ranking', '_tps')
        )
        
        print(f"\nÃ¢Å“â€œ Matched {len(merged)} pairs present in both methods")
        
        comparison_results = []
        
        for comp_type in ['same-specimen', 'intra-specific', 'inter-specific']:
            subset = merged[merged['comparison_type'] == comp_type]
            
            if len(subset) < 3:
                print(f"\n{comp_type}: Insufficient data (n={len(subset)})")
                continue
            
            print(f"\n{'='*60}")
            print(f"{comp_type.upper()} COMPARISON (n={len(subset)})")
            print(f"{'='*60}")
            
            for metric in ['mean', 'median', 'p95']:
                ranking_vals = subset[f'{metric}_ranking'].dropna()
                tps_vals = subset[f'{metric}_tps'].dropna()
                
                if len(ranking_vals) < 3 or len(tps_vals) < 3:
                    continue
                
                # Effect size
                diff = tps_vals.mean() - ranking_vals.mean()
                pooled_std = np.sqrt((ranking_vals.std()**2 + tps_vals.std()**2) / 2)
                cohens_d = diff / pooled_std if pooled_std > 0 else 0
                
                # Paired tests (same pairs)
                # Skip Wilcoxon if all differences are zero (same-specimen case)
                differences = tps_vals.values - ranking_vals.values
                if np.all(np.abs(differences) < 1e-10):
                    # All differences are essentially zero
                    w_stat = 0
                    p_wilcoxon = 1.0  # No difference
                    print(f"\n  {metric.upper()}:")
                    print(f"    RANKING: {ranking_vals.mean():.4f} Ã‚Â± {ranking_vals.std():.4f}")
                    print(f"    TPS:     {tps_vals.mean():.4f} Ã‚Â± {tps_vals.std():.4f}")
                    print(f"    Difference: {diff:+.4f}")
                    print(f"    Note: Both methods identical (all differences = 0)")
                    print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
                else:
                    # Normal case: run Wilcoxon test
                    w_stat, p_wilcoxon = wilcoxon(ranking_vals, tps_vals)
                    print(f"\n  {metric.upper()}:")
                    print(f"    RANKING: {ranking_vals.mean():.4f} Ã‚Â± {ranking_vals.std():.4f}")
                    print(f"    TPS:     {tps_vals.mean():.4f} Ã‚Â± {tps_vals.std():.4f}")
                    print(f"    Difference: {diff:+.4f}")
                    print(f"    Wilcoxon signed-rank test p-value: {p_wilcoxon:.4e}")
                    print(f"    Significant: {'YES' if p_wilcoxon < 0.05 else 'NO'}")
                    print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
                
                
                comparison_results.append({
                    'comparison_type': comp_type,
                    'metric': metric,
                    'n': len(ranking_vals),
                    'ranking_mean': ranking_vals.mean(),
                    'tps_mean': tps_vals.mean(),
                    'difference': diff,
                    'wilcoxon_p': p_wilcoxon,
                    'significant': p_wilcoxon < 0.05 if p_wilcoxon < 1.0 else False,
                    'cohens_d': cohens_d
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        comp_csv = self.output_dir / "method_comparison_statistics.csv"
        comparison_df.to_csv(comp_csv, index=False)
        print(f"\nÃ¢Å“â€œ Saved method comparison to {comp_csv}")
        
        return comparison_df
    
    def analyze_discrimination_power(self):
        """Analyze how well each method separates intra from inter-specific."""
        print("\n" + "="*80)
        print("STEP 5B: Analyzing Discrimination Power (Intra vs Inter Separation)")
        print("="*80)
        
        discrimination_results = []
        
        for method_name, df in [('RANKING', self.ranking_stats), ('TPS', self.tps_stats)]:
            print(f"\n{'='*60}")
            print(f"{method_name} METHOD - Discrimination Analysis")
            print(f"{'='*60}")
            
            intra = df[df['comparison_type'] == 'intra-specific']
            inter = df[df['comparison_type'] == 'inter-specific']
            
            if len(intra) < 3 or len(inter) < 3:
                print(f"  Insufficient data for discrimination analysis")
                continue
            
            for metric in ['mean', 'median', 'p95']:
                intra_vals = intra[metric].dropna()
                inter_vals = inter[metric].dropna()
                
                if len(intra_vals) < 3 or len(inter_vals) < 3:
                    continue
                
                # Calculate separation metrics
                intra_mean = intra_vals.mean()
                inter_mean = inter_vals.mean()
                separation = inter_mean - intra_mean
                
                # Effect size (Cohen's d) for discrimination
                pooled_std = np.sqrt((intra_vals.std()**2 + inter_vals.std()**2) / 2)
                cohens_d = separation / pooled_std if pooled_std > 0 else 0
                
                # Statistical test
                u_stat, p_val = mannwhitneyu(intra_vals, inter_vals, alternative='less')
                
                # Calculate AUROC (Area Under ROC Curve)
                # Combine labels: 0 = intra, 1 = inter
                labels = np.concatenate([np.zeros(len(intra_vals)), np.ones(len(inter_vals))])
                scores = np.concatenate([intra_vals.values, inter_vals.values])
                
                # For ROC: higher dissimilarity should predict inter-specific
                # So we want intra to have low scores, inter to have high scores
                try:
                    from sklearn.metrics import roc_auc_score, roc_curve, auc as auc_metric
                    auroc = roc_auc_score(labels, scores)
                except:
                    auroc = np.nan
                
                # Calculate Cliff's Delta (non-parametric effect size)
                # Count how many inter > intra comparisons
                n_greater = np.sum(inter_vals.values[:, np.newaxis] > intra_vals.values)
                n_less = np.sum(inter_vals.values[:, np.newaxis] < intra_vals.values)
                n_total = len(intra_vals) * len(inter_vals)
                cliffs_delta = (n_greater - n_less) / n_total if n_total > 0 else 0
                
                # Calculate tail differences
                intra_p95 = intra['p95'].dropna()
                inter_p95 = inter['p95'].dropna()
                
                if metric == 'mean' and len(intra_p95) > 0 and len(inter_p95) > 0:
                    # Tail length comparison
                    intra_idx = intra_vals.index.intersection(intra_p95.index)
                    inter_idx = inter_vals.index.intersection(inter_p95.index)
                    
                    intra_tail = (intra_p95[intra_idx] - intra_vals[intra_idx]).mean()
                    inter_tail = (inter_p95[inter_idx] - inter_vals[inter_idx]).mean()
                    tail_diff = inter_tail - intra_tail
                    
                    print(f"\n  {metric.upper()} Discrimination:")
                    print(f"    Intra-specific: {intra_mean:.4f} Ã‚Â± {intra_vals.std():.4f}")
                    print(f"    Inter-specific: {inter_mean:.4f} Ã‚Â± {inter_vals.std():.4f}")
                    print(f"    Separation: {separation:+.4f} ({100*separation/intra_mean:+.1f}%)")
                    print(f"    Cohen's d: {cohens_d:.3f} {'(good)' if cohens_d > 0.5 else '(poor)'}")
                    print(f"    AUROC: {auroc:.3f} {'(excellent)' if auroc > 0.9 else '(good)' if auroc > 0.8 else '(fair)' if auroc > 0.7 else '(poor)'}")
                    print(f"    Cliff's Delta: {cliffs_delta:.3f} {'(large)' if abs(cliffs_delta) > 0.474 else '(medium)' if abs(cliffs_delta) > 0.33 else '(small)'}")
                    print(f"    Mann-Whitney U p-value: {p_val:.4e}")
                    print(f"    Significant: {'YES' if p_val < 0.05 else 'NO'}")
                    print(f"    Tail Analysis:")
                    print(f"      Intra tail length (P95-mean): {intra_tail:.4f}")
                    print(f"      Inter tail length (P95-mean): {inter_tail:.4f}")
                    print(f"      Tail difference: {tail_diff:+.4f} {'(inter has longer tail Ã¢Å“â€œ)' if tail_diff > 0 else '(WARNING: intra has longer tail!)'}")
                    
                    discrimination_results.append({
                        'method': method_name,
                        'metric': metric,
                        'intra_mean': intra_mean,
                        'intra_std': intra_vals.std(),
                        'inter_mean': inter_mean,
                        'inter_std': inter_vals.std(),
                        'separation': separation,
                        'separation_pct': 100 * separation / intra_mean,
                        'cohens_d': cohens_d,
                        'auroc': auroc,
                        'cliffs_delta': cliffs_delta,
                        'mann_whitney_p': p_val,
                        'significant': p_val < 0.05,
                        'intra_tail_length': intra_tail,
                        'inter_tail_length': inter_tail,
                        'tail_difference': tail_diff
                    })
                else:
                    discrimination_results.append({
                        'method': method_name,
                        'metric': metric,
                        'intra_mean': intra_mean,
                        'intra_std': intra_vals.std(),
                        'inter_mean': inter_mean,
                        'inter_std': inter_vals.std(),
                        'separation': separation,
                        'separation_pct': 100 * separation / intra_mean,
                        'cohens_d': cohens_d,
                        'auroc': auroc,
                        'cliffs_delta': cliffs_delta,
                        'mann_whitney_p': p_val,
                        'significant': p_val < 0.05,
                        'intra_tail_length': np.nan,
                        'inter_tail_length': np.nan,
                        'tail_difference': np.nan
                    })
        
        # Compare discrimination between methods
        if len(discrimination_results) >= 2:
            print(f"\n{'='*60}")
            print("DISCRIMINATION COMPARISON: RANKING vs TPS")
            print(f"{'='*60}")
            
            for metric in ['mean', 'median', 'p95']:
                ranking_disc = [d for d in discrimination_results if d['method'] == 'RANKING' and d['metric'] == metric]
                tps_disc = [d for d in discrimination_results if d['method'] == 'TPS' and d['metric'] == metric]
                
                if ranking_disc and tps_disc:
                    r = ranking_disc[0]
                    t = tps_disc[0]
                    
                    sep_improvement = t['separation'] - r['separation']
                    cohens_improvement = t['cohens_d'] - r['cohens_d']
                    auroc_improvement = t['auroc'] - r['auroc']
                    cliffs_improvement = t['cliffs_delta'] - r['cliffs_delta']
                    
                    if metric == 'mean':
                        tail_improvement = t['tail_difference'] - r['tail_difference']
                        print(f"\n  {metric.upper()}:")
                        print(f"    RANKING separation: {r['separation']:.4f} (d={r['cohens_d']:.3f}, AUROC={r['auroc']:.3f}, Cliff={r['cliffs_delta']:.3f})")
                        print(f"    TPS separation:     {t['separation']:.4f} (d={t['cohens_d']:.3f}, AUROC={t['auroc']:.3f}, Cliff={t['cliffs_delta']:.3f})")
                        print(f"    Improvement: {sep_improvement:+.4f} {'Ã¢Å“â€œ TPS better!' if sep_improvement > 0 else 'Ã¢Å“â€” RANKING better'}")
                        print(f"    Cohen's d improvement: {cohens_improvement:+.3f}")
                        print(f"    AUROC improvement: {auroc_improvement:+.3f} {'Ã¢Å“â€œ' if auroc_improvement > 0 else 'Ã¢Å“â€”'}")
                        print(f"    Cliff's Delta improvement: {cliffs_improvement:+.3f} {'Ã¢Å“â€œ' if cliffs_improvement > 0 else 'Ã¢Å“â€”'}")
                        print(f"    RANKING tail difference: {r['tail_difference']:.4f}")
                        print(f"    TPS tail difference:     {t['tail_difference']:.4f}")
                        print(f"    Tail improvement: {tail_improvement:+.4f} {'Ã¢Å“â€œ Better tail separation!' if tail_improvement > 0 else 'Ã¢Å“â€” Worse tail separation'}")
        
        disc_df = pd.DataFrame(discrimination_results)
        disc_csv = self.output_dir / "discrimination_analysis.csv"
        disc_df.to_csv(disc_csv, index=False)
        print(f"\nÃ¢Å“â€œ Saved discrimination analysis to {disc_csv}")
        # ====================================================================
        # STEP 5C: Intra-specific variation per species (violins, CV bars, CSV)
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 5C: Intra-specific variation per species (RANKING vs TPS)")
        print("="*80)

        # Choose metric: 'mean' | 'median' | 'p95'
        metric_for_intra = 'p95'

        bundle = compute_intra_per_species_tables(
            self.ranking_stats,
            self.tps_stats,
            metric=metric_for_intra
        )
        rank_tab = bundle['rank_table']
        tps_tab  = bundle['tps_table']

        # Save detailed per-species tables + flags
        save_intra_species_tables(rank_tab, tps_tab, self.output_dir, metric=metric_for_intra)

        # Plots: violins (per-species distributions) and CV bar chart
        plot_intra_species_violins(rank_tab, tps_tab, self.output_dir, metric=metric_for_intra)
        plot_intra_species_cv_bars(rank_tab, tps_tab, self.output_dir, metric=metric_for_intra)
        
        return disc_df
    
    def plot_comparative_distributions(self):
        """Create distribution plots comparing both methods."""
        print("\n" + "="*80)
        print("STEP 6: Creating Comparative Distribution Plots")
        print("="*80)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Distribution Comparison: RANKING vs TPS Methods', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        metrics = ['mean', 'median', 'p95']
        comp_types = ['same-specimen', 'intra-specific', 'inter-specific']
        colors = {'RANKING': '#FF6B6B', 'TPS': '#4ECDC4'}
        
        for i, metric in enumerate(metrics):
            for j, comp_type in enumerate(comp_types):
                ax = axes[i, j]
                
                # Get data for both methods
                ranking_data = self.ranking_stats[self.ranking_stats['comparison_type'] == comp_type][metric].dropna()
                tps_data = self.tps_stats[self.tps_stats['comparison_type'] == comp_type][metric].dropna()
                
                if len(ranking_data) == 0 and len(tps_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{comp_type}\n{metric.upper()}")
                    continue
                
                # Plot histograms
                if len(ranking_data) > 0:
                    ax.hist(ranking_data, bins=30, alpha=0.6, label=f'RANKING (n={len(ranking_data)})',
                           color=colors['RANKING'], edgecolor='black', linewidth=0.5)
                if len(tps_data) > 0:
                    ax.hist(tps_data, bins=30, alpha=0.6, label=f'TPS (n={len(tps_data)})',
                           color=colors['TPS'], edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel('Dissimilarity')
                ax.set_ylabel('Frequency')
                ax.set_title(f"{comp_type}\n{metric.upper()}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "distributions_comparison_BOTH_METHODS.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Ã¢Å“â€œ Saved distribution comparison to {plot_path}")
    
    def plot_method_comparison_boxplots(self):
        """Create box plots comparing methods side-by-side."""
        print("\n" + "="*80)
        print("STEP 7: Creating Method Comparison Box Plots")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Method Comparison: RANKING vs TPS', fontsize=14, fontweight='bold')
        
        metrics = ['mean', 'median', 'p95']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for plotting
            plot_data = []
            labels = []
            
            for comp_type in ['same-specimen', 'intra-specific', 'inter-specific']:
                ranking_vals = self.ranking_stats[self.ranking_stats['comparison_type'] == comp_type][metric].dropna()
                tps_vals = self.tps_stats[self.tps_stats['comparison_type'] == comp_type][metric].dropna()
                
                if len(ranking_vals) > 0:
                    plot_data.append(ranking_vals)
                    labels.append(f"{comp_type[:8]}\nRANK")
                
                if len(tps_vals) > 0:
                    plot_data.append(tps_vals)
                    labels.append(f"{comp_type[:8]}\nTPS")
            
            if len(plot_data) > 0:
                bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                               showmeans=True, meanline=True)
                
                # Color boxes alternately
                for j, box in enumerate(bp['boxes']):
                    if 'RANK' in labels[j]:
                        box.set_facecolor('#FF6B6B')
                        box.set_alpha(0.6)
                    else:
                        box.set_facecolor('#4ECDC4')
                        box.set_alpha(0.6)
            
            ax.set_ylabel('Dissimilarity')
            ax.set_title(f'{metric.upper()} Dissimilarity')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = self.output_dir / "method_comparison_boxplots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Ã¢Å“â€œ Saved box plot comparison to {plot_path}")
    
    def plot_paired_comparison(self):
        """Create scatter plots showing paired differences between methods."""
        print("\n" + "="*80)
        print("STEP 8: Creating Paired Comparison Plots")
        print("="*80)
        
        # Merge datasets
        merged = pd.merge(
            self.ranking_stats,
            self.tps_stats,
            on=['image_A', 'image_B', 'comparison_type'],
            suffixes=('_ranking', '_tps')
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Paired Comparison: RANKING vs TPS (same image pairs)', 
                     fontsize=14, fontweight='bold')
        
        metrics = ['mean', 'median', 'p95']
        colors_map = {
            'same-specimen': '#FFA500',
            'intra-specific': '#4169E1',
            'inter-specific': '#32CD32'
        }
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for comp_type, color in colors_map.items():
                subset = merged[merged['comparison_type'] == comp_type]
                
                if len(subset) == 0:
                    continue
                
                ranking_vals = subset[f'{metric}_ranking']
                tps_vals = subset[f'{metric}_tps']
                
                ax.scatter(ranking_vals, tps_vals, alpha=0.5, s=30, 
                          label=f'{comp_type} (n={len(subset)})', color=color)
            
            # Add diagonal line (y=x)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=1, label='y=x')
            
            ax.set_xlabel('RANKING Dissimilarity')
            ax.set_ylabel('TPS Dissimilarity')
            ax.set_title(f'{metric.upper()}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plot_path = self.output_dir / "paired_comparison_scatter.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Ã¢Å“â€œ Saved paired comparison to {plot_path}")
    
    
    
    
    def plot_distribution_tails(self):
        """Create plots highlighting distribution tails and the 'morphological gap'
        used for split/lump decisions:
          • Blue dashed = intra-specific mean; blue dotted = intra-specific P95
          • Green dashed = inter-specific mean; green dotted = inter-specific P95
          • Shaded regions:
              - red (left):  LUMP-RISK  (x < inter_mean)
              - gray (middle): UNCERTAIN (inter_mean ≤ x < intra_P95)
              - yellow (right): SPLIT-SAFE (x ≥ intra_P95)
        """
        print("\n" + "="*80)
        print("STEP 9: Creating Distribution Tail Analysis Plots (with decision bands)")
        print("="*80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Distribution Tail Analysis: Intra vs Inter-Specific\n'
                     'Bands: red=LUMP-RISK  gray=UNCERTAIN  yellow=SPLIT-SAFE',
                     fontsize=14, fontweight='bold')

        methods = [('RANKING', self.ranking_stats), ('TPS', self.tps_stats)]

        for method_idx, (method_name, df) in enumerate(methods):
            for metric_idx, metric in enumerate(['mean', 'median', 'p95']):
                ax = axes[method_idx, metric_idx]

                intra = df[df['comparison_type'] == 'intra-specific'][metric].dropna()
                inter = df[df['comparison_type'] == 'inter-specific'][metric].dropna()

                ax.set_title(f"{method_name} - {metric.upper()}")

                if len(intra) == 0 or len(inter) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel('Dissimilarity')
                    ax.set_ylabel('Density')
                    ax.grid(True, alpha=0.3)
                    continue

                # Core summary points
                intra_mean = float(intra.mean())
                intra_p95  = float(intra.quantile(0.95))
                inter_mean = float(inter.mean())
                inter_p95  = float(inter.quantile(0.95))

                # Histograms (density) so heights are comparable even with different Ns
                ax.hist(intra, bins=20, alpha=0.55, density=True,
                        color='#4169E1', edgecolor='black', linewidth=0.5,
                        label=f'Intra (n={len(intra)})')
                ax.hist(inter, bins=30, alpha=0.55, density=True,
                        color='#32CD32', edgecolor='black', linewidth=0.5,
                        label=f'Inter (n={len(inter)})')

                # Vertical markers
                ax.axvline(intra_mean, color='#4169E1', linestyle='--', linewidth=2,
                           label=f'Intra mean: {intra_mean:.3f}')
                ax.axvline(intra_p95,  color='#4169E1', linestyle=':',  linewidth=2,
                           label=f'Intra P95: {intra_p95:.3f}')
                ax.axvline(inter_mean, color='#228B22', linestyle='--', linewidth=2,
                           label=f'Inter mean: {inter_mean:.3f}')
                ax.axvline(inter_p95,  color='#228B22', linestyle=':',  linewidth=2,
                           label=f'Inter P95: {inter_p95:.3f}')

                # Decision regions (left=red, middle=gray, right=yellow)
                xmin, xmax = ax.get_xlim()
                left_end   = max(xmin, min(inter_mean, xmax))
                mid_end    = max(left_end,  min(intra_p95, xmax))

                # LUMP-RISK: x < inter_mean
                if left_end > xmin:
                    ax.axvspan(xmin, left_end, color='red', alpha=0.12, label='Lump-risk')

                # UNCERTAIN: inter_mean ≤ x < intra_P95
                if mid_end > left_end:
                    ax.axvspan(left_end, mid_end, color='gray', alpha=0.15, label='Uncertain')

                # SPLIT-SAFE: x ≥ intra_P95
                if xmax > mid_end:
                    ax.axvspan(mid_end, xmax, color='yellow', alpha=0.22, label='Split-safe')

                # Quick counts to annotate how many pairs fall in each region (inter only)
                inter_lump_risk = (inter < inter_mean).mean()
                inter_uncertain = ((inter >= inter_mean) & (inter < intra_p95)).mean()
                inter_split_safe = (inter >= intra_p95).mean()

                # Put a compact note in the upper right
                txt = (f"Inter in regions (fractions):\n"
                       f"  Lump-risk: {inter_lump_risk:5.2%}\n"
                       f"  Uncertain: {inter_uncertain:5.2%}\n"
                       f"  Split-safe: {inter_split_safe:5.2%}")
                ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.8))

                ax.set_xlabel('Dissimilarity')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='upper left')

                # Console summary for reproducible logs
                print(f"\n[{method_name} • {metric.upper()}]")
                print(f"  inter_mean={inter_mean:.4f} | intra_P95={intra_p95:.4f}")
                print(f"  Region fractions (inter): "
                      f"lump={inter_lump_risk:.3f}, uncertain={inter_uncertain:.3f}, split={inter_split_safe:.3f}")

        plt.tight_layout()
        out_path = self.output_dir / "distribution_tail_analysis.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved tail analysis plots with decision bands to {out_path}")

    

    def plot_debugging_heatmap(self):
        """Create debugging heatmap showing RANKING vs TPS values with species labels AND filenames."""
        print("\n" + "="*80)
        print("STEP 10: Creating Debugging Heatmap (Image Files with Species Labels)")
        print("="*80)

        # FIRST: Build the authoritative list of intra-specific pairs from pairs_df
        print("\n  Building list of TRUE intra-specific pairs...")
        intra_specific_pairs = set()

        if self.pairs_df is not None:
            # Filter for intra-specific pairs only
            intra_df = self.pairs_df[self.pairs_df['comparison_type'] == 'intra-specific']

            print(f"  âœ“ Found {len(intra_df)} TRUE intra-specific pairs")
            print(f"    (These are the ONLY pairs that will get red borders)\n")

            # Print each intra-specific pair for debugging
            if len(intra_df) > 0:
                print("  ðŸ“‹ INTRA-SPECIFIC PAIRS LIST:")
                print("  " + "="*76)
                for idx, row in intra_df.iterrows():
                    img_a = row['image_A']
                    img_b = row['image_B']
                    species_a = row.get('species_A', 'Unknown')
                    species_b = row.get('species_B', 'Unknown')

                    # Extract filenames
                    filename_a = img_a.split('/')[-1] if '/' in img_a else img_a
                    filename_b = img_b.split('/')[-1] if '/' in img_b else img_b

                    print(f"  {idx+1:2d}. {filename_a}")
                    print(f"      {species_a}")
                    print(f"      <--->")
                    print(f"      {filename_b}")
                    print(f"      {species_b}")
                    print()

                    # Add both directions since matrix is symmetric
                    intra_specific_pairs.add((img_a, img_b))
                    intra_specific_pairs.add((img_b, img_a))

                print("  " + "="*76)
                print(f"  Total: {len(intra_specific_pairs)//2} pairs added to red box set\n")
            else:
                print("  â„¹ï¸  No intra-specific pairs found in this dataset\n")
        else:
            print("  âš  WARNING: pairs_df not available, cannot identify intra-specific pairs")

        # Merge ranking and TPS data
        merged = pd.merge(
            self.ranking_stats,
            self.tps_stats,
            on=['image_A', 'image_B'],
            suffixes=('_ranking', '_tps')
        )

        # Create a mapping of images to their labels (filename | species)
        image_to_label = {}
        image_to_species = {}  # Keep species for intra-specific detection

        for _, row in merged.iterrows():
            img_a = row['image_A']
            img_b = row['image_B']

            # Get species label (prefer ranking, fallback to tps)
            spec_a = row.get('specimen_label_A_ranking') or row.get('specimen_label_A_tps')
            spec_b = row.get('specimen_label_B_ranking') or row.get('specimen_label_B_tps')

            # Extract just filename from path
            filename_a = img_a.split('/')[-1] if '/' in img_a else img_a
            filename_b = img_b.split('/')[-1] if '/' in img_b else img_b

            # Create label: "filename | species"
            if pd.notna(spec_a):
                image_to_label[img_a] = f"{filename_a} | {spec_a}"
                image_to_species[img_a] = spec_a
            else:
                image_to_label[img_a] = filename_a
                image_to_species[img_a] = "Unknown"

            if pd.notna(spec_b):
                image_to_label[img_b] = f"{filename_b} | {spec_b}"
                image_to_species[img_b] = spec_b
            else:
                image_to_label[img_b] = filename_b
                image_to_species[img_b] = "Unknown"

        # Get unique images (sorted by SPECIES NAME first, then filename)
        def sort_key(img):
            """Sort by species name first, then by filename."""
            species = image_to_species.get(img, "zzz_Unknown")  # Put unknowns at end
            filename = img.split('/')[-1] if '/' in img else img
            return (species, filename)

        all_images = sorted(list(image_to_label.keys()), key=sort_key)
        image_list = all_images
        n_images = len(image_list)

        # DEBUG: Check if intra_specific_pairs images are in image_list
        if len(intra_specific_pairs) > 0:
            print(f"\n  ðŸ” DEBUGGING: Checking if intra-specific pairs are in image_list...")
            img_a_sample, img_b_sample = list(intra_specific_pairs)[0]
            print(f"     Sample pair from set: '{img_a_sample}' <-> '{img_b_sample}'")
            print(f"     Is img_a in image_list? {img_a_sample in image_list}")
            print(f"     Is img_b in image_list? {img_b_sample in image_list}\n")

        if n_images > 100:
            print(f"  Too many images ({n_images}) for readable heatmap")
            print(f"  Showing top 50 most common species only")
            image_counts = {}
            for img in image_list:
                count = 0
                for _, row in merged.iterrows():
                    if img == row['image_A'] or img == row['image_B']:
                        count += 1
                image_counts[img] = count
            top_images = sorted(image_counts.items(), key=lambda x: -x[1])[:50]
            image_list = [img for img, _ in top_images]
            n_images = len(image_list)

        # Create image index mapping
        img_to_idx = {img: i for i, img in enumerate(image_list)}

        # Initialize matrices
        ranking_matrix = np.full((n_images, n_images), np.nan)
        tps_matrix = np.full((n_images, n_images), np.nan)

        # Fill matrices
        for _, row in merged.iterrows():
            img_a = row['image_A']
            img_b = row['image_B']
            if img_a not in img_to_idx or img_b not in img_to_idx:
                continue
            idx_a = img_to_idx[img_a]
            idx_b = img_to_idx[img_b]
            ranking_p95 = row.get('p95_ranking', np.nan)
            tps_p95 = row.get('p95_tps', np.nan)
            ranking_matrix[idx_a, idx_b] = ranking_p95
            ranking_matrix[idx_b, idx_a] = ranking_p95
            tps_matrix[idx_a, idx_b] = tps_p95
            tps_matrix[idx_b, idx_a] = tps_p95

        # Create combined matrix: lower triangle = RANKING, upper triangle = TPS
        combined_matrix = np.full((n_images, n_images), np.nan)
        for i in range(n_images):
            for j in range(n_images):
                if i == j:
                    combined_matrix[i, j] = 0
                elif i > j:
                    combined_matrix[i, j] = ranking_matrix[i, j]
                else:
                    combined_matrix[i, j] = tps_matrix[i, j]

        # Figure
        figsize = max(15, n_images * 0.25)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        masked_matrix = np.ma.array(combined_matrix, mask=np.isnan(combined_matrix))
        im = ax.imshow(masked_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=0.25)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('P95 Dissimilarity', rotation=270, labelpad=20)

        labels = [image_to_label[img] for img in image_list]
        ax.set_xticks(range(n_images))
        ax.set_yticks(range(n_images))
        ax.set_xticklabels(labels, rotation=90, fontsize=7, ha='right')
        ax.set_yticklabels(labels, fontsize=7)

        ax.set_title(
            'Debugging Heatmap: Image-by-Image P95 Dissimilarity\n'
            'Lower Triangle = RANKING | Upper Triangle = TPS | Diagonal = 0\n'
            'Format: filename | species_name | SORTED BY SPECIES',
            fontsize=11, fontweight='bold', pad=20
        )

        ax.text(0.02, 0.98, 'RANKING\n(zero-shot)',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.98, 0.02, 'TPS\n(morphology-aware)',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # ---------- species boundaries + robust intra-specific outlines ----------
        def strip_spec_suffix(label: str) -> str:
            """'Colpoptera fusca-2' -> 'Colpoptera fusca' (leave other labels untouched)."""
            if not isinstance(label, str):
                return ""
            return re.sub(r"-\d+$", "", label.strip())

        image_to_specimen_label = {}
        image_to_base_species = {}
        for img in image_list:
            spec_label = image_to_species.get(img, "Unknown")
            image_to_specimen_label[img] = spec_label
            image_to_base_species[img] = strip_spec_suffix(spec_label)

        # separator lines (light gray)
        prev_species = None
        for i, img in enumerate(image_list):
            curr_species = image_to_base_species.get(img, "Unknown")
            if prev_species is not None and curr_species != prev_species:
                ax.axhline(y=i - 0.5, color='0.75', linewidth=1.0, alpha=0.8)
                ax.axvline(x=i - 0.5, color='0.75', linewidth=1.0, alpha=0.8)
            prev_species = curr_species

        red_boxes_drawn = 0
        for i in range(n_images):
            for j in range(n_images):
                if i == j:
                    ax.plot([j - 0.35, j + 0.35], [i - 0.35, i + 0.35], color='red', linewidth=1)
                    ax.plot([j - 0.35, j + 0.35], [i + 0.35, i - 0.35], color='red', linewidth=1)
                    continue
                if np.isnan(combined_matrix[i, j]):
                    continue

                img_i = image_list[i]
                img_j = image_list[j]
                base_i = image_to_base_species.get(img_i, "")
                base_j = image_to_base_species.get(img_j, "")
                spec_i = image_to_specimen_label.get(img_i, "")
                spec_j = image_to_specimen_label.get(img_j, "")

                # SAME base species but DIFFERENT specimens
                is_true_intra = (
                    base_i and base_j and base_i == base_j and
                    spec_i and spec_j and spec_i != spec_j
                )
                if is_true_intra:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor='red', linewidth=2.0)
                    ax.add_patch(rect)
                    red_boxes_drawn += 1
        # ------------------------------------------------------------------------

        print(f"\n  ðŸŽ¨ Drew {red_boxes_drawn} red boxes on heatmap")
        print(f"     (Note: boxes mark SAME base species but DIFFERENT specimens;")
        print(f"            lower triangle=Ranking, upper triangle=TPS)\n")

        plt.tight_layout()
        plot_path = self.output_dir / "debugging_heatmap_IMAGE_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"âœ“ Saved debugging heatmap to {plot_path}")
        print(f"  â—¦ SORTED BY SPECIES NAME for easy visual grouping")
        print(f"  Grey LINES = boundaries between different species")
        print(f"  Red BOXES  = intra-specific pairs (same species, different specimens)")
        print(f"  Lower triangle = RANKING P95 values")
        print(f"  Upper triangle = TPS P95 values")
        print(f"  Diagonal = self-comparisons (0)")
        print(f"  Labels format: filename | species_name")


        # Export intra and inter-specific pairs to separate CSVs for verification
        if self.pairs_df is not None:
            # Intra-specific pairs
            intra_csv_path = self.output_dir / "intra_specific_pairs_ONLY.csv"
            intra_export = self.pairs_df[self.pairs_df['comparison_type'] == 'intra-specific'].copy()
            if len(intra_export) > 0:
                # Add filename columns
                intra_export['filename_A'] = intra_export['image_A'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
                intra_export['filename_B'] = intra_export['image_B'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
                intra_export.to_csv(intra_csv_path, index=False)
                print(f"\nâœ“ Saved intra-specific pairs to {intra_csv_path}")
                print(f"   ({len(intra_export)} pairs - these should have RED BOXES)")
            
            # Inter-specific pairs (sample)
            inter_csv_path = self.output_dir / "inter_specific_pairs_SAMPLE.csv"
            inter_export = self.pairs_df[self.pairs_df['comparison_type'] == 'inter-specific'].copy()
            if len(inter_export) > 0:
                # Take a sample if too many
                if len(inter_export) > 100:
                    inter_export = inter_export.sample(100, random_state=42)
                # Add filename columns
                inter_export['filename_A'] = inter_export['image_A'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
                inter_export['filename_B'] = inter_export['image_B'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
                inter_export.to_csv(inter_csv_path, index=False)
                print(f"âœ“ Saved inter-specific pairs (sample) to {inter_csv_path}")
                print(f"   (Sample of {len(inter_export)} pairs - these should have NO RED BOXES)\n")
        
        # Also save a CSV with the pairs for easy debugging
        debug_pairs = []
        for i in range(n_images):
            for j in range(i+1, n_images):
                img_i = image_list[i]
                img_j = image_list[j]
                
                species_i = image_to_species.get(img_i, "")
                species_j = image_to_species.get(img_j, "")
                
                ranking_val = ranking_matrix[i, j]
                tps_val = tps_matrix[i, j]
                
                if not np.isnan(ranking_val) or not np.isnan(tps_val):
                    # Check if intra-specific
                    base_i = species_i.rsplit('-', 1)[0] if '-' in species_i and species_i.split('-')[-1].isdigit() else species_i
                    base_j = species_j.rsplit('-', 1)[0] if '-' in species_j and species_j.split('-')[-1].isdigit() else species_j
                    is_intra = (base_i == base_j and base_i != "" and base_i != "Unknown")
                    
                    # Extract filenames
                    filename_i = img_i.split('/')[-1] if '/' in img_i else img_i
                    filename_j = img_j.split('/')[-1] if '/' in img_j else img_j
                    
                    debug_pairs.append({
                        'image_A_filename': filename_i,
                        'image_B_filename': filename_j,
                        'image_A_full_path': img_i,
                        'image_B_full_path': img_j,
                        'species_A': species_i,
                        'species_B': species_j,
                        'species_A_base': base_i,
                        'species_B_base': base_j,
                        'is_intra_specific': is_intra,
                        'ranking_p95': ranking_val,
                        'tps_p95': tps_val,
                        'tps_minus_ranking': tps_val - ranking_val if not np.isnan(tps_val) and not np.isnan(ranking_val) else np.nan
                    })
        
        debug_df = pd.DataFrame(debug_pairs)
        debug_csv = self.output_dir / "debugging_IMAGE_pairs_with_filenames.csv"
        debug_df.to_csv(debug_csv, index=False)
        print(f"Ã¢Å“â€œ Saved debugging pairs CSV to {debug_csv}")
        
        # Print summary of intra-specific pairs found
        if len(debug_df) > 0:
            intra_pairs = debug_df[debug_df['is_intra_specific']]
            print(f"\n  INTRA-SPECIFIC PAIRS FOUND: {len(intra_pairs)}")
            if len(intra_pairs) > 0:
                print(f"  Pairs (showing filenames and species):")
                for _, row in intra_pairs.iterrows():
                    print(f"    {row['image_A_filename']} ({row['species_A']})")
                    print(f"      <-> {row['image_B_filename']} ({row['species_B']})")
                    print(f"      RANKING={row['ranking_p95']:.4f}, TPS={row['tps_p95']:.4f}")
                    print()
    

    def generate_comprehensive_report(self):
        """Generate a comprehensive text report."""
        print("\n" + "="*80)
        print("STEP 10: Generating Comprehensive Report")
        print("="*80)
        
        report_path = self.output_dir / "COMPREHENSIVE_ANALYSIS_REPORT.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MORPHOLOGY-AWARE ALIGNMENT ANALYSIS\n")
            f.write("Comparative Evaluation: RANKING vs TPS Methods\n")
            f.write("="*80 + "\n\n")
            
            f.write("ANALYSIS OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Observations CSV: {self.observations_csv}\n")
            f.write(f"Results directory: {self.results_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write("METHODS COMPARED\n")
            f.write("-"*80 + "\n")
            f.write("1. RANKING METHOD (dissimilarity_ranking/)\n")
            f.write("   - Zero-shot comparison without morphological alignment\n")
            f.write("   - Uses feature dissimilarity ranking\n\n")
            f.write("2. TPS METHOD (dissimilarity_tps_homology/)\n")
            f.write("   - Morphology-aware comparison with TPS alignment\n")
            f.write("   - Accounts for shape correspondence via thin-plate splines\n\n")
            
            f.write("DATA SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total unique specimens: {len(set(self.specimen_map.values()))}\n")
            f.write(f"Total unique species: {len(set(self.species_map.values()))}\n")
            f.write(f"Total image pairs analyzed: {len(self.pairs_df)}\n\n")
            
            f.write("FILTERING APPLIED:\n")
            f.write("  Ã¢â‚¬Â¢ Excluded pairs with incomplete taxonomic IDs\n")
            f.write("  Ã¢â‚¬Â¢ Incomplete IDs include:\n")
            f.write("    - Tribe-level only (e.g., 'Colpopterini')\n")
            f.write("    - Genus-level only (e.g., 'Colpoptera', 'Jamaha')\n")
            f.write("    - Indeterminate species (e.g., 'sp.', 'indet.')\n")
            f.write("  Ã¢â‚¬Â¢ Only pairs with full binomial species names are analyzed\n\n")
            
            type_counts = self.pairs_df['comparison_type'].value_counts()
            f.write("Comparison type distribution:\n")
            f.write(f"  Same-specimen (different photos): {type_counts.get('same-specimen', 0)}\n")
            f.write(f"  Intra-specific (different individuals, same species): {type_counts.get('intra-specific', 0)}\n")
            f.write(f"  Inter-specific (different species): {type_counts.get('inter-specific', 0)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*80 + "\n\n")
            
            # Load comparison results
            try:
                comp_df = pd.read_csv(self.output_dir / "method_comparison_statistics.csv")
                
                f.write("1. METHOD COMPARISON (RANKING vs TPS):\n")
                f.write("-"*80 + "\n\n")
                
                for comp_type in ['same-specimen', 'intra-specific', 'inter-specific']:
                    subset = comp_df[comp_df['comparison_type'] == comp_type]
                    if len(subset) == 0:
                        continue
                    
                    f.write(f"{comp_type.upper()}:\n")
                    for _, row in subset.iterrows():
                        f.write(f"  {row['metric'].upper()}:\n")
                        f.write(f"    RANKING: {row['ranking_mean']:.4f}\n")
                        f.write(f"    TPS:     {row['tps_mean']:.4f}\n")
                        f.write(f"    Difference: {row['difference']:+.4f}")
                        if row['difference'] > 0:
                            f.write(" (TPS higher Ã¢Å¡Â Ã¯Â¸Â)\n")
                        elif row['difference'] < 0:
                            f.write(" (TPS lower Ã¢Å“â€œ)\n")
                        else:
                            f.write(" (identical)\n")
                        f.write(f"    p-value: {row['wilcoxon_p']:.4e}\n")
                        f.write(f"    Significant: {'YES' if row['significant'] else 'NO'}\n")
                        f.write(f"    Effect size (Cohen's d): {row['cohens_d']:.3f}\n\n")
                
            except Exception as e:
                f.write(f"Could not load method comparison statistics: {e}\n\n")
            
            # Load discrimination analysis
            try:
                disc_df = pd.read_csv(self.output_dir / "discrimination_analysis.csv")
                
                f.write("\n2. DISCRIMINATION POWER (Intra vs Inter Separation):\n")
                f.write("-"*80 + "\n\n")
                
                for method in ['RANKING', 'TPS']:
                    subset = disc_df[disc_df['method'] == method]
                    if len(subset) == 0:
                        continue
                    
                    f.write(f"{method} METHOD:\n")
                    for _, row in subset.iterrows():
                        if row['metric'] == 'mean':  # Focus on mean for main report
                            f.write(f"  Separation: {row['separation']:.4f} ({row['separation_pct']:.1f}%)\n")
                            f.write(f"  Cohen's d: {row['cohens_d']:.3f}")
                            if row['cohens_d'] > 0.8:
                                f.write(" (LARGE - excellent discrimination)\n")
                            elif row['cohens_d'] > 0.5:
                                f.write(" (MEDIUM - good discrimination)\n")
                            elif row['cohens_d'] > 0.2:
                                f.write(" (SMALL - poor discrimination)\n")
                            else:
                                f.write(" (VERY SMALL - no discrimination)\n")
                            f.write(f"  Statistical significance: {'YES (p<0.05)' if row['significant'] else 'NO (pÃ¢â€°Â¥0.05)'}\n")
                            if not pd.isna(row['tail_difference']):
                                f.write(f"  Tail analysis:\n")
                                f.write(f"    Intra tail length (P95-mean): {row['intra_tail_length']:.4f}\n")
                                f.write(f"    Inter tail length (P95-mean): {row['inter_tail_length']:.4f}\n")
                                f.write(f"    Difference: {row['tail_difference']:+.4f}")
                                if row['tail_difference'] > 0:
                                    f.write(" (inter has longer tail Ã¢Å“â€œ)\n")
                                else:
                                    f.write(" (WARNING: intra has longer tail!)\n")
                            f.write("\n")
                
                # Compare discrimination between methods
                ranking_mean = disc_df[(disc_df['method'] == 'RANKING') & (disc_df['metric'] == 'mean')]
                tps_mean = disc_df[(disc_df['method'] == 'TPS') & (disc_df['metric'] == 'mean')]
                
                if len(ranking_mean) > 0 and len(tps_mean) > 0:
                    r = ranking_mean.iloc[0]
                    t = tps_mean.iloc[0]
                    
                    f.write("DISCRIMINATION COMPARISON:\n")
                    f.write(f"  RANKING: sep={r['separation']:.4f}, d={r['cohens_d']:.3f}, AUROC={r['auroc']:.3f}, Cliff={r['cliffs_delta']:.3f}\n")
                    f.write(f"  TPS:     sep={t['separation']:.4f}, d={t['cohens_d']:.3f}, AUROC={t['auroc']:.3f}, Cliff={t['cliffs_delta']:.3f}\n")
                    
                    sep_improvement = t['separation'] - r['separation']
                    d_improvement = t['cohens_d'] - r['cohens_d']
                    auroc_improvement = t['auroc'] - r['auroc']
                    cliffs_improvement = t['cliffs_delta'] - r['cliffs_delta']
                    
                    f.write(f"  Separation improvement: {sep_improvement:+.4f}")
                    if sep_improvement > 0:
                        f.write(" Ã¢Å“â€œ TPS improves separation!\n")
                    else:
                        f.write(" Ã¢Å“â€” RANKING has better separation\n")
                    
                    f.write(f"  Cohen's d improvement: {d_improvement:+.3f}")
                    if d_improvement > 0:
                        f.write(" Ã¢Å“â€œ TPS improves effect size!\n")
                    else:
                        f.write(" Ã¢Å“â€” RANKING has better effect size\n")
                    
                    f.write(f"  AUROC improvement: {auroc_improvement:+.3f}")
                    if auroc_improvement > 0:
                        f.write(" Ã¢Å“â€œ TPS improves discrimination!\n")
                    else:
                        f.write(" Ã¢Å“â€” RANKING has better discrimination\n")
                    
                    f.write(f"  Cliff's Delta improvement: {cliffs_improvement:+.3f}")
                    if cliffs_improvement > 0:
                        f.write(" Ã¢Å“â€œ TPS improves effect!\n")
                    else:
                        f.write(" Ã¢Å“â€” RANKING has better effect\n")
                    
                    if not pd.isna(r['tail_difference']) and not pd.isna(t['tail_difference']):
                        tail_improvement = t['tail_difference'] - r['tail_difference']
                        f.write(f"  Tail separation improvement: {tail_improvement:+.4f}")
                        if tail_improvement > 0:
                            f.write(" Ã¢Å“â€œ TPS improves tail separation!\n")
                        else:
                            f.write(" Ã¢Å“â€” RANKING has better tail separation\n")
                    f.write("\n")
                
            except Exception as e:
                f.write(f"Could not load discrimination analysis: {e}\n\n")
            
            # Load tail metrics
            try:
                tail_df = pd.read_csv(self.output_dir / "tail_metrics_distribution_shape.csv")
                
                f.write("\n3. DISTRIBUTION SHAPE ANALYSIS (Right-Hand Tail):\n")
                f.write("-"*80 + "\n\n")
                
                for method in ['RANKING', 'TPS']:
                    f.write(f"{method} METHOD:\n")
                    for comp_type in ['intra-specific', 'inter-specific']:
                        subset = tail_df[(tail_df['method'] == method) & (tail_df['comparison_type'] == comp_type)]
                        if len(subset) > 0:
                            row = subset.iloc[0]
                            f.write(f"  {comp_type}:\n")
                            f.write(f"    Tail length (P95-mean): {row['tail_from_mean_avg']:.4f} Ã‚Â± {row['tail_from_mean_std']:.4f}\n")
                            f.write(f"    Skewness: {row['skewness']:.3f}\n")
                    f.write("\n")
                
            except Exception as e:
                f.write(f"Could not load tail metrics: {e}\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXPECTED OUTCOMES FOR SUCCESSFUL MORPHOLOGY-AWARE METHOD:\n\n")
            f.write("1. SAME-SPECIMEN comparisons:\n")
            f.write("   - Should have LOWEST dissimilarity (near 0)\n")
            f.write("   - TPS may perform similarly or slightly better than RANKING\n\n")
            
            f.write("2. INTRA-SPECIFIC comparisons:\n")
            f.write("   - Should have LOW dissimilarity\n")
            f.write("   - TPS should ideally show LOWER values than RANKING\n")
            f.write("   - This demonstrates better recognition of morphological similarity\n\n")
            
            f.write("3. INTER-SPECIFIC comparisons:\n")
            f.write("   - Should have HIGHER dissimilarity than intra-specific\n")
            f.write("   - TPS may show similar or higher values than RANKING\n")
            f.write("   - Clear separation from intra-specific is key\n\n")
            
            f.write("METRICS:\n")
            f.write("  - MEAN: Average dissimilarity across all patches/cells\n")
            f.write("  - MEDIAN: Middle value (robust to outliers)\n")
            f.write("  - P95: 95th percentile (captures worst-case dissimilarity)\n\n")
            
            f.write("MEAN vs P95 INTERPRETATION:\n")
            f.write("  - Large gap (P95 >> Mean): High variability, some regions differ greatly\n")
            f.write("  - Small gap (P95 Ã¢â€°Ë† Mean): Uniform dissimilarity across specimen\n")
            f.write("  - TPS should ideally reduce the gap for true homologous regions\n\n")
            
            f.write("ADVANCED METRICS EXPLAINED:\n")
            f.write("-"*80 + "\n")
            f.write("Ã¢â‚¬Â¢ Separation: Inter-mean minus Intra-mean (HIGHER is better)\n")
            f.write("Ã¢â‚¬Â¢ Cohen's d: Standardized effect size\n")
            f.write("    < 0.2: trivial, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large\n")
            f.write("Ã¢â‚¬Â¢ AUROC: Area Under ROC Curve (discrimination ability)\n")
            f.write("    0.5: random, 0.7-0.8: fair, 0.8-0.9: good, > 0.9: excellent\n")
            f.write("Ã¢â‚¬Â¢ Cliff's Delta: Non-parametric effect size (robust to outliers)\n")
            f.write("    < 0.147: negligible, 0.147-0.33: small, 0.33-0.474: medium, > 0.474: large\n")
            f.write("Ã¢â‚¬Â¢ Tail length: P95 minus Mean (measures right skew)\n")
            f.write("Ã¢â‚¬Â¢ Tail difference: Inter_tail - Intra_tail (POSITIVE means inter has longer tail)\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FILES GENERATED\n")
            f.write("="*80 + "\n\n")
            f.write("Detailed Statistics:\n")
            f.write("  Ã¢â‚¬Â¢ detailed_statistics_RANKING.csv\n")
            f.write("  Ã¢â‚¬Â¢ detailed_statistics_TPS.csv\n\n")
            f.write("Summary Statistics:\n")
            f.write("  Ã¢â‚¬Â¢ summary_statistics_BOTH_METHODS.csv\n")
            f.write("  Ã¢â‚¬Â¢ tail_metrics_distribution_shape.csv\n")
            f.write("  Ã¢â‚¬Â¢ method_comparison_statistics.csv\n")
            f.write("  Ã¢â‚¬Â¢ discrimination_analysis.csv (includes AUROC and Cliff's Delta)\n\n")
            f.write("Visualizations:\n")
            f.write("  Ã¢â‚¬Â¢ distributions_comparison_BOTH_METHODS.png\n")
            f.write("  Ã¢â‚¬Â¢ method_comparison_boxplots.png\n")
            f.write("  Ã¢â‚¬Â¢ paired_comparison_scatter.png\n")
            f.write("  Ã¢â‚¬Â¢ distribution_tail_analysis.png\n\n")
            f.write("Debugging Files (NEW!):\n")
            f.write("  Ã¢â‚¬Â¢ debugging_heatmap_IMAGE_comparison.png - Image-by-image heatmap\n")
            f.write("    (Labels: filename | species_name)\n")
            f.write("    (Lower triangle = RANKING, Upper triangle = TPS, Red boxes = intra-specific)\n")
            f.write("  Ã¢â‚¬Â¢ debugging_IMAGE_pairs_with_filenames.csv - All pairs with filenames\n")
            f.write("    (Use to verify which images are truly same species)\n")
            f.write("    (Check for tribe-level IDs like 'Colpopterini' being wrongly grouped!)\n\n")
        
        print(f"Ã¢Å“â€œ Saved comprehensive report to {report_path}")
    
    def run_full_analysis(self):
        """Run the complete comparative analysis pipeline."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE COMPARATIVE ANALYSIS")
        print("="*80)
        print(f"Observations CSV: {self.observations_csv}")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Run all analysis steps
        self.map_images_to_taxonomy()
        self.load_pairwise_results()
        self.extract_statistics_by_method()
        self.compute_summary_statistics()
        self.compare_methods_statistically()
        self.analyze_discrimination_power()
        self.plot_comparative_distributions()
        self.plot_method_comparison_boxplots()
        self.plot_paired_comparison()
        self.plot_distribution_tails()
        self.plot_debugging_heatmap()
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("Ã¢Å“â€œ COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nKey findings in: COMPREHENSIVE_ANALYSIS_REPORT.txt")
        print("\nÃ¢Å¡Â Ã¯Â¸Â  CHECK DEBUGGING FILES:")
        print("  Ã¢â‚¬Â¢ debugging_heatmap_IMAGE_comparison.png - Visual image-by-image comparison")
        print("    (Labels show: filename | species_name)")
        print("  Ã¢â‚¬Â¢ debugging_IMAGE_pairs_with_filenames.csv - List of all pairs with image filenames")
        print("\nÃ°Å¸â€™Â¡ TIP: Use the filenames to verify which images are truly the same species!")
        print("         Red boxes in heatmap = pairs classified as intra-specific")
        print("         Check if 'Colpopterini' (tribe-only IDs) are incorrectly grouped!")


def main():
    parser = argparse.ArgumentParser(
        description="Comparative analysis of RANKING vs TPS morphology-aware methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_tps_comparative_v4.py \\
    --observations-csv observations_photos.csv \\
    --results-dir /path/to/pairwise_runs_nogodinidae \\
    --output-dir ./comparative_analysis_results
        """
    )
    
    parser.add_argument(
        '--observations-csv',
        type=Path,
        required=True,
        help='Path to observations_photos.csv with species labels'
    )
    
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help='Path to all_by_all output directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory where analysis outputs will be saved'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = ComparativeDissimilarityAnalyzer(
        observations_csv=args.observations_csv,
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
