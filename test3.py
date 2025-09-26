#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare RQ3 data-point counts between:
  A) DiCE viz (per-method/strategy/hist_seed):  ./evaluations/feasibility/mahalanobis/{ABBR}_DiCE_{method}_{strategy}_hist_seed.csv
  B) Legacy viz (per-explainer files):          ./evaluations/feasibility/mahalanobis/{ABBR}_{Explainer}.csv

For each, this script reports:
  - raw row count
  - how many 'min' entries are non-null
  - how many 'min' are numeric (coercible)
  - (optional) how many would be kept under a cutoff (e.g., <= 10.0)

It also aggregates Legacy counts across explainers to compare against DiCE counts per model.
"""

from pathlib import Path
import argparse
import pandas as pd

# ------------ Config ------------
MODEL_ABBR = {
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "SVM": "SVM",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

LEGACY_MODELS = ["RandomForest", "XGBoost", "SVM"]
ALL_MODELS     = ["RandomForest", "XGBoost", "SVM", "LightGBM", "CatBoost"]
LEGACY_EXPLAINERS = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]


# ------------ Helpers ------------

def _read_csv_lenient(path: Path) -> pd.DataFrame | None:
    """Read CSV; return None on any failure."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _count_min_columns(df: pd.DataFrame, cutoff: float | None = None) -> dict:
    """Count raw rows, non-null mins, numeric mins, and <= cutoff (if provided)."""
    if df is None or df.empty:
        return dict(rows_raw=0, min_non_na=0, min_numeric=0, kept_cutoff=None)

    rows_raw = len(df)
    if "min" not in df.columns:
        return dict(rows_raw=rows_raw, min_non_na=0, min_numeric=0, kept_cutoff=None)

    mins = pd.to_numeric(df["min"], errors="coerce")
    min_non_na = int(df["min"].notna().sum())
    min_numeric = int(mins.notna().sum())
    kept_cutoff = None
    if cutoff is not None:
        kept_cutoff = int((mins.dropna() <= cutoff).sum())

    return dict(rows_raw=rows_raw, min_non_na=min_non_na, min_numeric=min_numeric, kept_cutoff=kept_cutoff)

def count_points_rq3_dice(method="random", strategy="best", hist_seed=True, cutoff: float | None = 10.0) -> pd.DataFrame:
    """
    Count per-model points for DiCE viz files. Returns a DataFrame with columns:
    [Model, File, rows_raw, min_non_na, min_numeric, kept_cutoff, status]
    """
    rows = []
    suffix = "_hist_seed" if hist_seed else ""
    for model in ALL_MODELS:
        abbr = MODEL_ABBR[model]
        path = Path(f"./evaluations/feasibility/mahalanobis/{abbr}_DiCE_{method}_{strategy}{suffix}.csv")
        df = _read_csv_lenient(path)
        counts = _count_min_columns(df, cutoff=cutoff)
        status = "OK" if (df is not None and not df.empty and "min" in df.columns) else (
                 "NO_MIN" if (df is not None and "min" not in (df.columns if hasattr(df, "columns") else [])) else
                 "MISSING/EMPTY")
        rows.append([model, str(path), counts["rows_raw"], counts["min_non_na"], counts["min_numeric"], counts["kept_cutoff"], status])

    out = pd.DataFrame(rows, columns=["Model","File","rows_raw","min_non_na","min_numeric","kept_cutoff","status"])
    return out

def count_points_rq3_legacy(cutoff: float | None = None) -> pd.DataFrame:
    """
    Count per-(Model, Explainer) points for legacy viz files. Returns a DataFrame with columns:
    [Model, Explainer, File, rows_raw, min_non_na, min_numeric, kept_cutoff, status]
    """
    rows = []
    for model in LEGACY_MODELS:
        abbr = MODEL_ABBR[model]
        for exp in LEGACY_EXPLAINERS:
            path = Path(f"./evaluations/feasibility/mahalanobis/{abbr}_{exp}.csv")
            df = _read_csv_lenient(path)
            counts = _count_min_columns(df, cutoff=cutoff)
            status = "OK" if (df is not None and not df.empty and "min" in df.columns) else (
                     "NO_MIN" if (df is not None and "min" not in (df.columns if hasattr(df, "columns") else [])) else
                     "MISSING/EMPTY")
            rows.append([model, exp, str(path), counts["rows_raw"], counts["min_non_na"], counts["min_numeric"], counts["kept_cutoff"], status])

    out = pd.DataFrame(rows, columns=["Model","Explainer","File","rows_raw","min_non_na","min_numeric","kept_cutoff","status"])
    return out

def compare_dice_vs_legacy(dice_df: pd.DataFrame, legacy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Legacy counts across explainers per model, and compare to DiCE per-model counts.
    Produces columns:
      Model, dice_min_numeric, legacy_min_numeric_total, diff (legacy - dice), pct_diff
    """
    # Aggregate legacy per model (sum of numeric mins across explainers)
    legacy_agg = legacy_df.groupby("Model", as_index=False)["min_numeric"].sum().rename(columns={"min_numeric":"legacy_min_numeric_total"})

    # Keep only overlapping models (RandomForest, XGBoost, SVM) for a fair comparison
    dice_subset = dice_df[dice_df["Model"].isin(LEGACY_MODELS)][["Model","min_numeric"]].copy()
    dice_agg = dice_subset.rename(columns={"min_numeric":"dice_min_numeric"})

    merged = pd.merge(legacy_agg, dice_agg, on="Model", how="outer").fillna(0)
    merged["diff"] = merged["legacy_min_numeric_total"] - merged["dice_min_numeric"]
    merged["pct_diff"] = merged.apply(lambda r: (r["diff"] / r["legacy_min_numeric_total"] * 100.0) if r["legacy_min_numeric_total"] > 0 else 0.0, axis=1)
    return merged[["Model","dice_min_numeric","legacy_min_numeric_total","diff","pct_diff"]]

def pretty_print(title: str, df: pd.DataFrame):
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("(no rows)")
        return
    try:
        from tabulate import tabulate
        print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))
    except Exception:
        print(df.to_string(index=False))


# ------------ CLI ------------

def main():
    ap = argparse.ArgumentParser(description="Compare RQ3 data-point counts between DiCE and Legacy visualizations.")
    ap.add_argument("--method", type=str, default="random", help="DiCE method (random, kdtree, genetic)")
    ap.add_argument("--strategy", type=str, default="best", help="Selection strategy (best, first)")
    ap.add_argument("--hist_seed", action="store_true", help="Use _hist_seed file suffix for DiCE files")
    ap.add_argument("--no_hist_seed", dest="hist_seed", action="store_false", help="Do NOT use _hist_seed suffix")
    ap.add_argument("--cutoff", type=float, default=10.0, help="Optional cutoff for 'min' (set to -1 to disable)")
    ap.set_defaults(hist_seed=True)
    args = ap.parse_args()

    cutoff = None if (args.cutoff is not None and args.cutoff < 0) else args.cutoff

    # Count
    dice_counts   = count_points_rq3_dice(method=args.method, strategy=args.strategy, hist_seed=args.hist_seed, cutoff=cutoff)
    legacy_counts = count_points_rq3_legacy(cutoff=cutoff)

    # Print per-source details
    pretty_print(f"DiCE counts (method={args.method}, strategy={args.strategy}, hist_seed={args.hist_seed}, cutoff={cutoff})", dice_counts)
    pretty_print("Legacy counts (per model × explainer)", legacy_counts)

    # Summaries
    dice_summary = dice_counts.groupby("Model", as_index=False)[["rows_raw","min_non_na","min_numeric"]].sum()
    legacy_summary = legacy_counts.groupby("Model", as_index=False)[["rows_raw","min_non_na","min_numeric"]].sum()

    pretty_print("DiCE per-model summary", dice_summary)
    pretty_print("Legacy per-model summary (explainer-aggregated)", legacy_summary)

    # Comparison (only overlapping models)
    comp = compare_dice_vs_legacy(dice_counts, legacy_counts)
    pretty_print("Comparison (Legacy total vs DiCE per model) — using 'min_numeric' counts", comp)

    # Totals
    total_dice_numeric   = int(dice_counts["min_numeric"].sum())
    total_legacy_numeric = int(legacy_counts["min_numeric"].sum())
    print(f"\nTOTAL numeric 'min' entries — DiCE: {total_dice_numeric} | Legacy (sum over explainers): {total_legacy_numeric}")

    # Optional: save CSVs
    outdir = Path("./evaluations/debug")
    outdir.mkdir(parents=True, exist_ok=True)
    dice_counts.to_csv(outdir / f"dice_counts_{args.method}_{args.strategy}{'_hist' if args.hist_seed else ''}.csv", index=False)
    legacy_counts.to_csv(outdir / "legacy_counts.csv", index=False)
    comp.to_csv(outdir / f"compare_legacy_vs_dice_{args.method}_{args.strategy}{'_hist' if args.hist_seed else ''}.csv", index=False)
    print(f"\nSaved CSVs to {outdir}")

if __name__ == "__main__":
    main()
