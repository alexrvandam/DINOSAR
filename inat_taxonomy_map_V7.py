#!/usr/bin/env python3
# inat_taxonomy_map_V6b.py

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from itertools import islice

from pyinaturalist import get_observations

PAIR_RE = re.compile(r'(?P<photo>\d+)__+(?P<obs>\d+)')

def chunks(iterable: Iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def normalize_name(name: str, strip_tokens: List[str]) -> str:
    p = Path(name)
    stem = p.stem
    for tok in strip_tokens:
        if tok and stem.lower().endswith(tok.lower()):
            stem = stem[: -len(tok)]
    return f"{stem}{p.suffix}"

def parse_ids_from_name(base_name: str) -> Tuple[Optional[int], Optional[int]]:
    m = PAIR_RE.search(base_name)
    if m:
        return int(m.group('photo')), int(m.group('obs'))
    nums = re.findall(r'\d+', base_name)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])  # best-guess: photo, obs
    elif len(nums) == 1:
        return None, int(nums[0])          # single number → obs
    return None, None

def wanted_id_for_filename(path: Path, mode: str, strip_tokens: List[str]) -> Tuple[Optional[int], Dict[str, Optional[int]], str]:
    original = path.name
    normalized = normalize_name(original, strip_tokens)
    photo_id, obs_id = parse_ids_from_name(normalized)
    info = {'photo_id_in_name': photo_id, 'observation_id_in_name': obs_id}
    if mode == 'obs':
        return obs_id, info, normalized
    if mode == 'photo':
        return photo_id, info, normalized
    # auto: prefer obs when available
    return (obs_id if obs_id is not None else photo_id), info, normalized

def fetch_observations_bulk(user: str, taxon_id: Optional[int], quality: str):
    """Bulk fetch observations for a user (and optional taxon/quality)."""
    obs = []
    page = 1
    allowed = {"casual", "needs_id", "research"}
    while True:
        params = {'user_id': user, 'taxon_id': taxon_id, 'per_page': 200, 'page': page}
        if quality in allowed:
            params['quality_grade'] = quality
        r = get_observations(**params)
        results = r.get('results', [])
        if not results:
            break
        obs.extend(results)
        if len(results) < 200:
            break
        page += 1
    return obs

def fetch_by_observation_ids(obs_ids: List[int]) -> List[Dict]:
    """Exact lookup by observation IDs (batches)."""
    found = []
    for batch in chunks(obs_ids, 200):
        r = get_observations(id=batch, per_page=200)
        found.extend(r.get('results', []))
    return found

def fetch_by_photo_ids(photo_ids: List[int]) -> List[Dict]:
    """Exact lookup by photo IDs (batches)."""
    found = []
    for batch in chunks(photo_ids, 200):
        r = get_observations(photo_id=batch, per_page=200)
        found.extend(r.get('results', []))
    return found

def lineage_from_taxon(taxon: Dict) -> Dict[str, str]:
    anc = taxon.get('ancestors') or []
    d = {a.get('rank', ''): a.get('name', '') for a in anc}
    return {
        'kingdom': d.get('kingdom', ''), 'phylum': d.get('phylum', ''), 'class': d.get('class', ''),
        'order': d.get('order', ''), 'family': d.get('family', ''), 'subfamily': d.get('subfamily', ''),
        'tribe': d.get('tribe', ''), 'genus': d.get('genus', ''), 'species': d.get('species', ''),
        'subspecies': d.get('infraspecies', '') or d.get('subspecies', ''),
    }

def flat_row_from_observation(o: Dict) -> Dict:
    tax = (o.get('taxon') or {})
    coords = (o.get('geojson') or {}).get('coordinates') or [None, None]
    return {
        'observation_id': o.get('id'),
        'quality_grade': o.get('quality_grade') or '',
        'user_login': (o.get('user') or {}).get('login') or '',
        'observed_on': o.get('observed_on') or '',
        'place_guess': o.get('place_guess') or '',
        'latitude': coords[1],
        'longitude': coords[0],
        'taxon_id': tax.get('id'),
        'rank': tax.get('rank') or '',
        'scientific_name': tax.get('name') or '',
        'preferred_common_name': tax.get('preferred_common_name') or '',
        **lineage_from_taxon(tax),
    }

def build_indexes(observations: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Return (obs_id → flat_row, photo_id → obs_id)."""
    flat_rows_by_obs: Dict[str, Dict] = {}
    photo_to_obs: Dict[str, str] = {}
    for o in observations:
        oid = str(o.get('id'))
        if not oid:
            continue
        flat_rows_by_obs[oid] = flat_row_from_observation(o)
        for p in o.get('photos') or []:
            pid = p.get('id')
            if pid is not None:
                photo_to_obs[str(pid)] = oid
    return flat_rows_by_obs, photo_to_obs

def main():
    ap = argparse.ArgumentParser("Map iNat filenames to taxonomy")
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--glob', default='*.jpg')
    ap.add_argument('--outfile', required=True)
    ap.add_argument('--user', required=True, help='iNat username (e.g., hopper-museum)')
    ap.add_argument('--taxon-id', type=int, default=None)
    ap.add_argument('--quality', default='any', choices=['any', 'casual', 'needs_id', 'research'])
    ap.add_argument('--id-field', default='auto', choices=['auto', 'obs', 'photo'])
    ap.add_argument('--filename-strip', nargs='*', default=['_lat'],
                    help="Suffix tokens to strip from stem before parsing IDs (e.g., _lat _dor)")
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    images = sorted(Path(args.images_dir).glob(args.glob))
    if args.debug:
        print(f"[DEBUG] Found {len(images)} image files with pattern {args.glob}")
        print(f"[DEBUG] Will strip tokens: {args.filename_strip if args.filename_strip else '[]'}")

    file_records = []
    for img in images:
        id_for_match, info, normalized = wanted_id_for_filename(img, args.id_field, args.filename_strip)
        rec = {
            'path': str(img),
            'filename_original': img.name,
            'filename_normalized': normalized,
            'id_value': id_for_match,
            **info,
        }
        file_records.append(rec)

    # 1) Bulk fetch by user (fast path)
    print("Fetching observations from iNaturalist …")
    obs_all = fetch_observations_bulk(args.user, args.taxon_id, args.quality)
    print(f"Fetched {len(obs_all)} observations")

    flat_rows_by_obs, photo_to_obs = build_indexes(obs_all)

    if args.debug:
        print(f"[DEBUG] Index sizes: obs={len(flat_rows_by_obs)}  photo→obs={len(photo_to_obs)}")
        print("[DEBUG] Sample parsed filenames:")
        for rec in file_records[:8]:
            print(f"  - {rec['filename_original']}  → norm={rec['filename_normalized']}  "
                  f"photo={rec['photo_id_in_name']}  obs={rec['observation_id_in_name']}  "
                  f"id_for_match={rec['id_value']}")

    def try_match(rec) -> Optional[Dict]:
        photo_in_name = rec['photo_id_in_name']
        obs_in_name = rec['observation_id_in_name']
        matched_obs_id = None

        if args.id_field == 'obs':
            if obs_in_name is not None and str(obs_in_name) in flat_rows_by_obs:
                matched_obs_id = str(obs_in_name)
            elif photo_in_name is not None and str(photo_in_name) in photo_to_obs:
                matched_obs_id = photo_to_obs[str(photo_in_name)]
        elif args.id_field == 'photo':
            if photo_in_name is not None and str(photo_in_name) in photo_to_obs:
                matched_obs_id = photo_to_obs[str(photo_in_name)]
            elif obs_in_name is not None and str(obs_in_name) in flat_rows_by_obs:
                matched_obs_id = str(obs_in_name)
        else:  # auto
            if obs_in_name is not None and str(obs_in_name) in flat_rows_by_obs:
                matched_obs_id = str(obs_in_name)
            elif photo_in_name is not None and str(photo_in_name) in photo_to_obs:
                matched_obs_id = photo_to_obs[str(photo_in_name)]

        if matched_obs_id:
            base = flat_rows_by_obs[matched_obs_id]
            return {
                'filename_original': rec['filename_original'],
                'filename_normalized': rec['filename_normalized'],
                'observation_id': int(matched_obs_id),
                'photo_id': int(photo_in_name) if isinstance(photo_in_name, int) else (photo_in_name or ''),
                **base,
            }
        return None

    # First pass match
    out_rows = []
    still_unmatched = []
    for rec in file_records:
        row = try_match(rec)
        if row is not None:
            out_rows.append(row)
        else:
            still_unmatched.append(rec)

    # 2) Fallback: actively look up exact IDs for the ones we missed
    if still_unmatched:
        if args.debug:
            print(f"[DEBUG] Attempting active lookup for {len(still_unmatched)} unmatched files…")

        # Collect what we need to look up
        need_obs_ids = sorted(set(
            rec['observation_id_in_name'] for rec in still_unmatched if rec['observation_id_in_name'] is not None
        ))
        need_photo_ids = sorted(set(
            rec['photo_id_in_name'] for rec in still_unmatched if rec['photo_id_in_name'] is not None
        ))

        # Query by observation IDs
        if need_obs_ids:
            if args.debug:
                print(f"[DEBUG] Looking up {len(need_obs_ids)} observation IDs directly")
            found_obs = fetch_by_observation_ids(need_obs_ids)
            extra_flat, extra_p2o = build_indexes(found_obs)
            flat_rows_by_obs.update(extra_flat)
            photo_to_obs.update(extra_p2o)

        # Query by photo IDs
        if need_photo_ids:
            if args.debug:
                print(f"[DEBUG] Looking up {len(need_photo_ids)} photo IDs directly")
            found_by_photo = fetch_by_photo_ids(need_photo_ids)
            extra_flat, extra_p2o = build_indexes(found_by_photo)
            flat_rows_by_obs.update(extra_flat)
            photo_to_obs.update(extra_p2o)

        # Second pass match
        out_rows2 = []
        still_unmatched2 = []
        for rec in still_unmatched:
            row = try_match(rec)
            if row is not None:
                out_rows2.append(row)
            else:
                still_unmatched2.append(rec)

        out_rows.extend(out_rows2)
        still_unmatched = still_unmatched2

    # Write results
    cols = [
        'filename_original','filename_normalized',
        'observation_id','photo_id',
        'taxon_id','rank','scientific_name','preferred_common_name',
        'kingdom','phylum','class','order','family','subfamily','tribe','genus','species','subspecies',
        'quality_grade','user_login','observed_on','place_guess','latitude','longitude',
    ]
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, '') for k in cols})

    print(f"Wrote {len(out_rows)} rows to {out_path}")
    if len(out_rows) == 0:
        print("Note: None of your filenames matched. This usually means your filenames use photo IDs "
              "but you matched on observation IDs (or vice-versa). Try --id-field photo or --id-field obs.")

    # Debug artifacts
    if args.debug:
        unf_files_csv = out_path.parent / 'unmatched_files.csv'
        unf_ids_csv = out_path.parent / 'unmatched_ids.csv'
        # use a permissive writer: include whatever keys are present
        if still_unmatched:
            fieldnames = sorted(set().union(*[r.keys() for r in still_unmatched]))
            with open(unf_files_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader(); w.writerows(still_unmatched)
        else:
            with open(unf_files_csv, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['no_unmatched'])

        # IDs file: store both forms we saw in names to help debug
        ids_rows = []
        for r in still_unmatched:
            ids_rows.append([r.get('photo_id_in_name') or '', r.get('observation_id_in_name') or ''])
        with open(unf_ids_csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['photo_id_in_name','observation_id_in_name']); w.writerows(ids_rows)

        print(f"[DEBUG] Unmatched files: {len(still_unmatched)} -> {unf_files_csv}")
        print(f"[DEBUG] Unmatched ids:   {len(ids_rows)} -> {unf_ids_csv}")

        # Show a few concrete misses:
        for r in still_unmatched[:6]:
            p, o = r.get('photo_id_in_name'), r.get('observation_id_in_name')
            print(f"[MISS] {r['filename_original']}  photo={p} ({'hit' if p and str(p) in photo_to_obs else 'miss'})  "
                  f"obs={o} ({'hit' if o and str(o) in flat_rows_by_obs else 'miss'})")

if __name__ == '__main__':
    main()
