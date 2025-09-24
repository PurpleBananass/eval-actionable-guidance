# audit_first_vs_any.py
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from data_utils import read_dataset, get_model
from hyparams import EXPERIMENTS

def _safe_read(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def _feat_cols(df): return [c for c in df.columns if c not in {"test_idx","candidate_id","proba0","proba1","target"}]
def _features_only(df): return df.loc[:, df.columns != "target"]

def first_vs_any(model_type):
    ds = read_dataset()
    rows = []
    for project,(train,test) in ds.items():
        multi = _safe_read(Path(EXPERIMENTS)/project/model_type/"DiCE_all.csv")
        single = _safe_read(Path(EXPERIMENTS)/project/model_type/"DiCE_all_single_cf.csv")
        if multi is None or multi.empty or "test_idx" not in multi.columns: 
            continue
        feat_cols = list(_features_only(test).columns)
        fcols = [c for c in multi.columns if c in feat_cols]
        model = get_model(project, model_type)
        scaler = StandardScaler().fit(_features_only(train).values)

        def flips_for_group(g, t):
            orig = test.loc[t, feat_cols].astype(float)
            X = g[fcols].astype(float).values
            full = []
            for r in X:
                v = orig.copy()
                v[fcols] = r
                full.append(v.values)
            preds = model.predict(scaler.transform(np.asarray(full)))
            return preds == 0  # boolean array

        first_flips, any_flips = 0, 0
        for t, g in multi.groupby("test_idx"):
            t = int(t)
            ok = flips_for_group(g, t)
            if ok.any(): any_flips += 1
            if len(ok) > 0 and ok[0]: first_flips += 1

        # how many singles exist & flip?
        single_present, single_flip = 0, 0
        if single is not None and not single.empty and "test_idx" in single.columns:
            for t, g in single.groupby("test_idx"):
                t = int(t)
                ok = flips_for_group(g, t)
                single_present += 1
                if ok.any(): single_flip += 1

        rows.append([model_type, project, any_flips, first_flips, single_present, single_flip])
    return pd.DataFrame(rows, columns=["Model","Project","multi_any","multi_first","single_present","single_flip"])

if __name__ == "__main__":
    out = []
    for m in ["CatBoost","LightGBM","XGBoost","RandomForest","SVM"]:
        out.append(first_vs_any(m))
    df = pd.concat(out, ignore_index=True)
    print(df.to_string(index=False))
    print("\nSummary by model (does first==any and single match?):")
    print(df.groupby("Model")[["multi_any","multi_first","single_present","single_flip"]].sum())
