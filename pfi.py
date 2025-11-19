# importance_compare.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List
from argparse import ArgumentParser
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---- your helpers (must be available in PYTHONPATH) ----
from data_utils import read_dataset, get_model, get_true_positives

# ---------- small utilities ----------
def _mad(v: np.ndarray, c: float = 1.4826) -> float:
    m = np.median(v)
    return float(c * np.median(np.abs(v - m)) + 1e-12)

def _l1_normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x.astype(float), 0.0)
    s = x.sum()
    return x / s if s > 0 else x

class _RobustUnits:
    def __init__(self, X: np.ndarray):
        self.med = np.median(X, axis=0)
        self.mad = np.array([_mad(X[:, j]) for j in range(X.shape[1])], dtype=float)
    def to_units(self, X: np.ndarray) -> np.ndarray:
        return (X - self.med) / self.mad
    def from_units(self, U: np.ndarray) -> np.ndarray:
        return U * self.mad + self.med

class _ProbaModel:
    def __init__(self, model): self.m = model
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        m = self.m
        if hasattr(m, "predict_proba"):
            P = m.predict_proba(X)
            if P.ndim == 1: p1 = P; return np.stack([1-p1, p1], 1)
            if P.shape[1] == 1: p1 = P[:,0]; return np.stack([1-p1, p1], 1)
            return P[:, :2]
        if hasattr(m, "decision_function"):
            f = m.decision_function(X)
            f = f if f.ndim == 1 else (f[:,0] if f.shape[1] == 1 else f[:,1])
            p1 = 1/(1+np.exp(-f)); return np.stack([1-p1, p1], 1)
        y = m.predict(X); p0=(y==0).astype(float); return np.stack([p0, 1-p0], 1)

# ---------- Flip-aware PFI (local, conditional) ----------
def flip_pfi_local(
    X_train: pd.DataFrame,
    x0: pd.Series,
    model,
    *,
    k_neighbors: int = 100,
    samples_per_feature: int = 400,
    topk_aggregate: int = 25,
    density_lambda: float = 0.02,     # softer
    clip_to_train_range: bool = True,
    random_state: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    feats = list(X_train.columns)
    X = X_train.values.astype(float)
    x = x0[feats].astype(float).values

    # robust units
    med = np.median(X, axis=0)
    mad = np.maximum(1e-12, np.array([np.median(np.abs(X[:,j]-np.median(X[:,j])))*1.4826 for j in range(X.shape[1])]))
    def to_u(v): return (v - med) / mad
    def from_u(u): return u * mad + med

    Xu = to_u(X); x_u = to_u(x)
    lo, hi = X.min(axis=0), X.max(axis=0)

    # neighbors (bigger pool)
    from sklearn.neighbors import NearestNeighbors
    k = min(max(30, k_neighbors), len(X))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Xu)
    _, idx = nn.kneighbors(x_u.reshape(1,-1), return_distance=True)
    neigh_u = Xu[idx[0]]; neigh = from_u(neigh_u)

    # base prob & target
    def prob1(Z): 
        P = model.predict_proba(Z)
        return P[:,1]
    p = prob1(x.reshape(1,-1))[0]; target = 1 - int(p >= 0.5)
    p0 = prob1(x.reshape(1,-1))[0] if target==1 else 1 - prob1(x.reshape(1,-1))[0]

    # helper for target prob
    def p_target(Z):
        p1 = prob1(Z)
        return p1 if target==1 else (1.0 - p1)

    mean_neigh_dist = float(np.mean(np.linalg.norm(neigh_u - x_u, axis=1))) + 1e-12

    rows = []
    for j, name in enumerate(feats):
        # sample half from neighbors, half from global quantiles
        idxs = rng.integers(0, len(neigh), size=samples_per_feature//2)
        v_neigh = neigh[idxs, j].astype(float)
        qs = np.linspace(0, 1, max(7, samples_per_feature//2))
        v_quant = np.unique(np.quantile(X[:,j], qs, method="linear"))
        v_vals = np.unique(np.concatenate([v_neigh, v_quant, [x[j]]]))
        if clip_to_train_range:
            v_vals = np.clip(v_vals, lo[j], hi[j])

        Xc = np.repeat(x.reshape(1,-1), repeats=len(v_vals), axis=0)
        Xc[:, j] = v_vals
        P = p_target(Xc)

        # unit step
        step_u = np.abs((v_vals - x[j]) / mad[j]) + 1e-12
        gains = (P - p0) / step_u

        # simple density penalty in unit space
        x_move_u = x_u.reshape(1,-1).repeat(len(v_vals), axis=0)
        x_move_u[:, j] = (v_vals - med[j]) / mad[j]
        dens = np.linalg.norm(neigh_u.mean(axis=0) - x_move_u, axis=1) / mean_neigh_dist

        adj = gains - density_lambda * dens
        pos = adj > 0
        if not np.any(pos):
            rows.append((name, 0.0, x[j], 0.0, 0.0, 0.0)); continue

        adj_pos = adj[pos]; v_pos = v_vals[pos]; g_pos = gains[pos]
        k_use = max(1, min(topk_aggregate, adj_pos.size))
        top_idx = np.argpartition(-adj_pos, k_use-1)[:k_use]
        med_topk = float(np.median(adj_pos[top_idx]))
        best_i = int(top_idx[np.argmax(adj_pos[top_idx])])
        rows.append((name, med_topk, float(v_pos[best_i]), float(g_pos[best_i]),
                     med_topk, float(np.percentile(adj_pos[top_idx],75) - np.percentile(adj_pos[top_idx],25))))

    df = pd.DataFrame(rows, columns=["feature","flip_pfi","best_value","best_gain_unit","median_topk_gain","stability_iqr"])
    df["flip_pfi"] = np.clip(df["flip_pfi"], 0.0, None)
    # normalize
    s = df["flip_pfi"].sum()
    if s > 0: df["flip_pfi"] /= s
    return df.sort_values("flip_pfi", ascending=False, kind="stable").reset_index(drop=True)

# ---------- LIME (local, opposite-class importances) ----------
def lime_local_opposite(
    X_train: pd.DataFrame,
    x0: pd.Series,
    y_train: np.ndarray,
    model,
    *,
    num_samples: int = 2000,
    random_state: int = 123,
) -> np.ndarray:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception:
        return np.zeros(X_train.shape[1], dtype=float)

    feats = list(X_train.columns)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train[feats].values.astype(float))
    xs = scaler.transform(x0[feats].values.astype(float).reshape(1,-1))

    explainer = LimeTabularExplainer(
        training_data=Xs,
        training_labels=y_train.astype(int),
        feature_names=feats,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=random_state,
    )

    # figure out opposite class
    def _predict_proba_scaled(Z):
        Z_inv = scaler.inverse_transform(Z)
        return model.predict_proba(Z_inv)

    # Request BOTH labels so as_map() contains the opposite class too
    exp = explainer.explain_instance(
        xs[0],
        _predict_proba_scaled,
        num_features=len(feats),
        num_samples=num_samples,
        labels=[0, 1],        # <-- key fix
    )

    # predicted and opposite
    p = model.predict_proba(x0[feats].values.astype(float).reshape(1,-1))[0]
    pred = int(p[1] >= 0.5)
    target = 1 - pred

    mp = dict(exp.as_map().get(target, []))
    w = np.zeros(len(feats), dtype=float)
    for idx, val in mp.items():
        if 0 <= idx < len(feats):
            w[idx] = abs(float(val))

    # L1 normalize (safe)
    s = w.sum()
    return w / s if s > 0 else w


# ---------- SPSA randomized gradient (local, robust units) ----------
def spsa_local(
    X_train: pd.DataFrame,
    x0: pd.Series,
    model,
    *,
    repeats: int = 256,
    eps_units: float = 0.25,
    random_state: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    feats = list(X_train.columns)
    X = X_train[feats].values.astype(float)
    x = x0[feats].values.astype(float)

    # robust units
    med = np.median(X, axis=0)
    mad = np.maximum(1e-12, np.array([np.median(np.abs(X[:,j] - np.median(X[:,j]))) * 1.4826 for j in range(X.shape[1])]))
    def to_u(v): return (v - med) / mad
    def from_u(u): return u * mad + med

    xu = to_u(x)
    d = len(feats)

    # use DECISION MARGIN to avoid probability saturation
    def margin(Z):
        if hasattr(model, "decision_function"):
            f = model.decision_function(Z)
            f = f if f.ndim == 1 else f[:,0]
            return f
        p = model.predict_proba(Z)[:,1]
        # logit(p) clipped
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p/(1-p))

    # adapt epsilon: if |logit(p)| large, nudge bigger steps
    p0 = model.predict_proba(x.reshape(1,-1))[0,1]
    logit0 = np.log(np.clip(p0,1e-6,1-1e-6)/(1-np.clip(p0,1e-6,1-1e-6)))
    eps = eps_units * (1.0 + min(2.0, abs(logit0)/2.0))

    grads = []
    for _ in range(repeats):
        u = rng.choice([-1.0, 1.0], size=d).astype(float)
        x_plus  = from_u(xu + eps*u).reshape(1,-1)
        x_minus = from_u(xu - eps*u).reshape(1,-1)
        g = ((margin(x_plus) - margin(x_minus)) / (2.0*eps)) * u  # elementwise via u
        grads.append(g * 1.0)  # (scale)
    grads = np.vstack(grads)
    imp_raw = np.median(np.abs(grads), axis=0)

    s = imp_raw.sum()
    return imp_raw / s if s > 0 else imp_raw


# ---------- Global importances (built-in) ----------
def global_builtin_importance(
    model,
    feature_names: List[str],
    X_train: pd.DataFrame,
    y_train: Optional[np.ndarray] = None
) -> np.ndarray:
    imp = np.zeros(len(feature_names), dtype=float)
    m = model; cname = m.__class__.__name__
    try:
        if hasattr(m, "feature_importances_") and len(getattr(m, "feature_importances_")) == len(feature_names):
            imp = np.asarray(m.feature_importances_, dtype=float)
        elif cname == "XGBClassifier" and hasattr(m, "get_booster"):
            score = m.get_booster().get_score(importance_type="gain")
            for k, v in score.items():
                if k.startswith("f"):
                    j = int(k[1:])
                    if 0 <= j < len(feature_names): imp[j] = float(v)
        elif cname == "LGBMClassifier" and hasattr(m, "booster_"):
            b = m.booster_; vals = np.asarray(b.feature_importance(importance_type="gain"), dtype=float)
            names = b.feature_name(); mp = {n:i for i,n in enumerate(names)}
            for i,f in enumerate(feature_names):
                if f in mp: imp[i] = vals[mp[f]]
        elif cname == "CatBoostClassifier":
            try:
                from catboost import Pool
                pool = Pool(X_train.values, y_train, feature_names=feature_names)
                vals = np.asarray(m.get_feature_importance(data=pool, type="FeatureImportance"), dtype=float)
                if vals.size == len(feature_names): imp = vals
            except Exception:
                pass
        elif (cname in ("LinearSVC","SVC")) and getattr(m, "kernel", "linear") == "linear" and hasattr(m, "coef_"):
            w = np.abs(m.coef_.ravel().astype(float))
            std = np.std(X_train.values.astype(float), axis=0) + 1e-12
            imp = w * std
    except Exception:
        pass
    return _l1_normalize_nonneg(np.maximum(imp, 0.0))

# ---------- Master compare ----------
def compare_importances(
    X_train: pd.DataFrame,
    x0: pd.Series,
    y_train: np.ndarray,
    model,
    *,
    k_neighbors: int = 50,
    samples_per_feature: int = 200,
    topk_aggregate: int = 15,
    density_lambda: float = 0.05,
    lime_num_samples: int = 2000,
    spsa_repeats: int = 128,
    random_state: int = 123,
    include_global: bool = True,
) -> pd.DataFrame:
    feats = list(X_train.columns)
    prior = global_builtin_importance(model, feats, X_train, y_train) if include_global else None

    flip_df = flip_pfi_local(
        X_train, x0, model,
        k_neighbors=k_neighbors,
        samples_per_feature=samples_per_feature,
        topk_aggregate=topk_aggregate,
        density_lambda=density_lambda,
        # global_prior=prior,
        random_state=random_state,
    )
    flip_scores = np.zeros(len(feats), dtype=float)
    flip_scores[[feats.index(f) for f in flip_df["feature"].tolist()]] = flip_df["flip_pfi"].values
    flip_scores = _l1_normalize_nonneg(flip_scores)

    lime_scores = lime_local_opposite(
        X_train, x0, y_train, model,
        num_samples=lime_num_samples,
        random_state=random_state,
    )

    spsa_scores = spsa_local(
        X_train, x0, model,
        repeats=spsa_repeats,
        eps_units=0.25,
        random_state=random_state,
    )

    data = {
        "feature": feats,
        "flip_pfi": flip_scores,
        "lime_opposite": lime_scores,
        "spsa": spsa_scores,
    }
    if include_global:
        data["global"] = prior

    df = pd.DataFrame(data)
    for col in [c for c in df.columns if c != "feature"]:
        df[col + "_rank"] = df[col].rank(ascending=False, method="min").astype(int)
    return df.sort_values("flip_pfi", ascending=False, kind="stable").reset_index(drop=True)

# ---------- CLI main ----------
def main():
    ap = ArgumentParser(description="Compare local importances: Flip-PFI vs LIME vs SPSA (+ global).")
    ap.add_argument("--project", type=str, required=True, help="Project key (as in your read_dataset())")
    ap.add_argument("--model_type", type=str, required=True,
                    choices=["RandomForest", "SVM", "XGBoost", "LightGBM", "CatBoost"])
    ap.add_argument("--test_idx", type=int, default=None, help="Test index to explain; defaults to first true positive.")
    ap.add_argument("--k_neighbors", type=int, default=50)
    ap.add_argument("--samples_per_feature", type=int, default=200)
    ap.add_argument("--topk_aggregate", type=int, default=15)
    ap.add_argument("--density_lambda", type=float, default=0.05)
    ap.add_argument("--lime_num_samples", type=int, default=2000)
    ap.add_argument("--spsa_repeats", type=int, default=128)
    ap.add_argument("--no_global", action="store_true", help="Disable global built-in importance column")
    ap.add_argument("--outdir", type=str, default="./evaluations/importance_compare",
                    help="Directory to write CSV")
    args = ap.parse_args()

    # Load data & model
    ds = read_dataset()
    if args.project not in ds:
        raise ValueError(f"Project '{args.project}' not found in datasets.")
    train, test = ds[args.project]
    feat_cols = [c for c in train.columns if c != "target"]
    X_train = train[feat_cols].copy()
    y_train = train["target"].astype(int).values

    model = get_model(args.project, args.model_type)

    # Choose instance
    if args.test_idx is not None and int(args.test_idx) in test.index:
        x0 = test.loc[int(args.test_idx), feat_cols]
        chosen_idx = int(args.test_idx)
    else:
        # pick first true positive (y=1 & pred=1); fallback to first test row
        tp_df = get_true_positives(model, train, test)
        if not tp_df.empty:
            chosen_idx = int(tp_df.index.astype(int)[0])
            x0 = test.loc[chosen_idx, feat_cols]
        else:
            chosen_idx = int(test.index.astype(int)[0])
            x0 = test.loc[chosen_idx, feat_cols]

    print(f"[INFO] Project={args.project}, Model={args.model_type}, test_idx={chosen_idx}")

    # Compare
    table = compare_importances(
        X_train, x0, y_train, model,
        k_neighbors=args.k_neighbors,
        samples_per_feature=args.samples_per_feature,
        topk_aggregate=args.topk_aggregate,
        density_lambda=args.density_lambda,
        lime_num_samples=args.lime_num_samples,
        spsa_repeats=args.spsa_repeats,
        include_global=(not args.no_global),
    )

    # Show top-10
    print("\nTop-10 by Flip-PFI:")
    print(table.head(10).to_string(index=False))

    # Save
    outdir = Path(args.outdir) / args.project / args.model_type
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"importance_{chosen_idx}.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
