#!/usr/bin/env python3
"""
Count flips per explainer from existing experiment outputs.

Assumptions (matching your pipeline):
- For each project/model/explainer[/search_strategy], results are in:
    {EXPERIMENTS}/{project}/{model}/{explainer}_all.csv
    or
    {EXPERIMENTS}/{project}/{model}/{explainer}_{search}_all.csv
- A flipped instance is a non-NaN row; non-flips are all-NaN rows.
- We DO NOT recompute models or flips. We only summarize already-saved CSVs.

Examples:
    python count_flips_per_explainer.py --projects all --model RandomForest
    python count_flips_per_explainer.py --projects apachejit openstack --model XGBoost
    python count_flips_per_explainer.py --model LightGBM --by-search --save ./flip_summary.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
from tabulate import tabulate

from hyparams import EXPERIMENTS  # uses your existing path constant
from data_utils import read_dataset  # used only to list projects when --projects all


def _parse_explainer_and_search(stem: str) -> Tuple[str, Optional[str]]:
    """
    From a file stem like:
        'LIME_all'                     -> ('LIME', None)
        'kfeature_random_all'          -> ('kfeature', 'random')
        'TimeLIME_all'                 -> ('TimeLIME', None)
        'SQAPlanner_greedy_all'        -> ('SQAPlanner', 'greedy')
    we first strip the trailing '_all', then split the remainder once on '_'.
    """
    base = stem[:-4] if stem.endswith("_all") else stem
    if "_" in base:
        explainer, search = base.split("_", 1)
        return explainer, search
    return base, None


def _summarize_folder(folder: Path, by_search: bool):
    """
    Scan one {EXPERIMENTS}/{project}/{model} folder and aggregate flips.
    Returns:
        per_key: dict[(explainer, search_or_None)] -> {'flips': int, 'total': int, 'projects': set()}
        files_seen: int
    """
    per_key: Dict[Tuple[str, Optional[str]], Dict[str, object]] = defaultdict(
        lambda: {"flips": 0, "total": 0, "projects": set()}
    )
    files_seen = 0

    if not folder.exists():
        return per_key, files_seen

    for csv_path in folder.glob("*_all.csv"):
        files_seen += 1
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue  # skip unreadable files

        flips = len(df.dropna())
        total = len(df)

        explainer, search = _parse_explainer_and_search(csv_path.stem)
        key = (explainer, search if by_search else None)

        per_key[key]["flips"] += flips
        per_key[key]["total"] += total
        per_key[key]["projects"].add(folder.parts[-2])  # project name in path
    return per_key, files_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", type=str, default="all",
                    help="Space/comma-separated projects or 'all' to scan all from read_dataset()")
    ap.add_argument("--model", type=str, default="RandomForest",
                    help="Model folder to scan under each project (e.g., RandomForest, SVM, XGBoost, ...)")
    ap.add_argument("--by-search", action="store_true",
                    help="If set, split results by search strategy; otherwise group by explainer only.")
    ap.add_argument("--save", type=str, default=None,
                    help="Optional CSV path to save the summary table.")
    args = ap.parse_args()

    # Resolve projects
    if args.projects.strip().lower() == "all":
        projects = list(read_dataset().keys())
    else:
        projects = [p for chunk in args.projects.replace(",", " ").split() for p in [chunk.strip()] if p]

    base = Path(EXPERIMENTS)
    grand: Dict[Tuple[str, Optional[str]], Dict[str, object]] = defaultdict(
        lambda: {"flips": 0, "total": 0, "projects": set()}
    )

    files_total = 0
    for project in sorted(projects):
        model_dir = base / project / args.model
        per_key, seen = _summarize_folder(model_dir, args.by_search)
        files_total += seen
        for k, v in per_key.items():
            grand[k]["flips"] += v["flips"]
            grand[k]["total"] += v["total"]
            grand[k]["projects"] |= v["projects"]

    if not grand:
        print("No matching *_all.csv files were found. Check EXPERIMENTS path, model, or projects.")
        return

    # Build summary dataframe
    rows = []
    for (explainer, search), agg in sorted(grand.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        flips = agg["flips"]
        total = agg["total"]
        rate = (flips / total) if total > 0 else 0.0
        rows.append({
            "Explainer": explainer,
            **({"Search": search or "-"} if args.by_search else {}),
            "Projects": len(agg["projects"]),
            "Flips": flips,
            "Computed": total,
            "Flip%": f"{rate*100:.2f}%"
        })

    df = pd.DataFrame(rows)

    # Pretty print
    headers = ["Explainer"] + (["Search"] if args.by_search else []) + ["Projects", "Flips", "Computed", "Flip%"]
    print(tabulate(df[headers].values.tolist(), headers=headers, tablefmt="github"))
    print(f"\nScanned CSV files: {files_total}")

    # Optional save
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved summary to: {out.resolve()}")

if __name__ == "__main__":
    main()
