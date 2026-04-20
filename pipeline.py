"""
pipeline.py - Multi-Model Bearing Fault Diagnosis Training Pipeline (CWRU Dataset)

Trains 4 separate XGBoost models:
    1. DE + 12k  (MAIN - used for SHAP explainability)
    2. DE + 48k
    3. FE + 12k
    4. FE + 48k

Each model uses the correct sampling rate for feature extraction.
Data selection uses explicit file-ID mapping (same as original validated pipeline)
to ensure clean labeling and high accuracy without overfitting.
"""

import os
import glob
import math
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

from utils.feature_extraction import (
    preprocess_signal, segment_signal, extract_features,
    FEATURE_COLUMNS, WINDOW_SIZE,
)

warnings.filterwarnings("ignore")

CWRU_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CWRU-dataset")
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────
# FILE-ID TO LABEL MAPPING (validated, matches original pipeline)
# Each ID = one .mat file. The digit prefix of the filename is the ID.
# ──────────────────────────────────────────────────────────────────

# 12k Drive End files (DE_time signal)
ID_LABEL_DE_12K = {
    # Normal (4 load conditions)
    97: "Normal", 98: "Normal", 99: "Normal", 100: "Normal",
    # IR faults
    105: "IR_007", 106: "IR_007", 107: "IR_007", 108: "IR_007",
    169: "IR_014", 170: "IR_014", 171: "IR_014", 172: "IR_014",
    209: "IR_021", 210: "IR_021", 211: "IR_021", 212: "IR_021",
    3001: "IR_028", 3002: "IR_028", 3003: "IR_028", 3004: "IR_028",
    # OR faults (@6 o'clock position — standard)
    130: "OR_007", 131: "OR_007", 132: "OR_007", 133: "OR_007",
    197: "OR_014", 198: "OR_014", 199: "OR_014", 200: "OR_014",
    234: "OR_021", 235: "OR_021", 236: "OR_021", 237: "OR_021",
    # Ball faults
    118: "Ball_007", 119: "Ball_007", 120: "Ball_007", 121: "Ball_007",
    185: "Ball_014", 186: "Ball_014", 187: "Ball_014", 188: "Ball_014",
    222: "Ball_021", 223: "Ball_021", 224: "Ball_021", 225: "Ball_021",
    3005: "Ball_028", 3006: "Ball_028", 3007: "Ball_028", 3008: "Ball_028",
}

# 12k Fan End files (FE_time signal)
# Note: Fan End data has 9 fault classes (no Ball_028 / IR_028 / OR_028 in Fan End folder)
ID_LABEL_FE_12K = {
    97: "Normal", 98: "Normal", 99: "Normal", 100: "Normal",
    278: "IR_007", 279: "IR_007", 280: "IR_007", 281: "IR_007",
    274: "IR_014", 275: "IR_014", 276: "IR_014", 277: "IR_014",
    270: "IR_021", 271: "IR_021", 272: "IR_021", 273: "IR_021",
    294: "OR_007", 295: "OR_007", 296: "OR_007", 297: "OR_007",
    310: "OR_014", 309: "OR_014", 311: "OR_014", 312: "OR_014",
    315: "OR_021", 316: "OR_021", 317: "OR_021", 318: "OR_021",
    282: "Ball_007", 283: "Ball_007", 284: "Ball_007", 285: "Ball_007",
    286: "Ball_014", 287: "Ball_014", 288: "Ball_014", 289: "Ball_014",
    290: "Ball_021", 291: "Ball_021", 292: "Ball_021", 293: "Ball_021",
}

# ──────────────────────────────────────────────────────────────────
# Config: which folders to scan, which signal key, which ID map
# ──────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "DE_12k": {
        "folders": ["12k_Drive_End_Bearing_Fault_Data", "Normal"],
        "signal_key": "DE_time",
        "fs": 12000.0,
        "id_map": ID_LABEL_DE_12K,
    },
    "FE_12k": {
        "folders": ["12k_Fan_End_Bearing_Fault_Data", "Normal"],
        "signal_key": "FE_time",
        "fs": 12000.0,
        "id_map": ID_LABEL_FE_12K,
    },
}


# ──────────────────────────────────────────────────────────────────
# Extract file ID from filename (same logic as original code)
# ──────────────────────────────────────────────────────────────────

def _extract_id(path):
    """Extract numeric ID from filename. E.g. '105_0.mat' -> 105, '3001_0.mat' -> 3001."""
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = ""
    for ch in stem:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _fault_size_from_label(label):
    """Extract fault size from label. 'IR_007' -> 0.007, 'Normal' -> 0.0."""
    if label == "Normal":
        return 0.0
    parts = label.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1]) / 1000.0
    return 0.0


# ──────────────────────────────────────────────────────────────────
# Load signals for one configuration
# ──────────────────────────────────────────────────────────────────

def load_signals(config_name):
    cfg = DATASET_CONFIG[config_name]
    id_map = cfg["id_map"]
    records = []

    for folder in cfg["folders"]:
        folder_path = os.path.join(CWRU_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        mat_paths = glob.glob(os.path.join(folder_path, "**", "*.mat"), recursive=True)

        for path in mat_paths:
            file_id = _extract_id(path)
            if file_id is None or file_id not in id_map:
                continue
            label = id_map[file_id]

            try:
                data = loadmat(path)
            except Exception:
                continue

            sig_key = None
            for k in data.keys():
                if cfg["signal_key"] in k:
                    sig_key = k
                    break
            if sig_key is None:
                continue

            signal = np.asarray(data[sig_key]).squeeze().astype(np.float64)
            if signal.size < WINDOW_SIZE or not np.isfinite(signal).any():
                continue

            records.append({
                "signal": signal,
                "label": label,
                "fault_size": _fault_size_from_label(label),
                "filename": os.path.basename(path),
            })

    print(f"  [{config_name}] Loaded {len(records)} files, "
          f"{len(set(r['label'] for r in records))} classes")
    return records


# ──────────────────────────────────────────────────────────────────
# Build features
# ──────────────────────────────────────────────────────────────────

def build_features(records, fs):
    rows = []
    for rec in records:
        filtered = preprocess_signal(rec["signal"], fs=fs)
        windows = segment_signal(filtered)
        if len(windows) == 0:
            continue
        for idx, win in enumerate(windows):
            feats = extract_features(win, fs=fs)
            if feats is None:
                continue
            feats["label"] = rec["label"]
            feats["fault_size"] = rec["fault_size"]
            feats["filename"] = rec["filename"]
            feats["window_id"] = idx
            rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"  {len(df)} windows, {df['label'].nunique()} classes")
    return df


def balance_classes(df, label_col="label"):
    counts = df[label_col].value_counts()
    if counts.empty:
        return df
    min_count = counts.min()
    parts = [
        grp.sample(min_count, random_state=RANDOM_STATE) if len(grp) > min_count else grp
        for _, grp in df.groupby(label_col)
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
# Train model
# ──────────────────────────────────────────────────────────────────

def _manual_group_split(df, test_ratio=0.5):
    """Per-class file-level split guaranteeing every class appears in both train and test.

    For each class, sort its files deterministically, then split files into train/test.
    Ensures every class has >=1 file in each set (required for XGBoost training).
    """
    rng = np.random.RandomState(RANDOM_STATE)
    train_mask = np.zeros(len(df), dtype=bool)
    for label, grp in df.groupby("label"):
        files = sorted(grp["filename"].unique())
        n = len(files)
        # Shuffle deterministically
        idx = rng.permutation(n)
        files = [files[i] for i in idx]
        n_train = max(1, n - max(1, int(n * test_ratio)))  # at least 1 train, 1 test
        train_files = set(files[:n_train])
        train_mask |= (df["label"].values == label) & df["filename"].isin(train_files).values
    return train_mask


def train_model(df, config_name, models_dir):
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    class_names = sorted(np.unique(y))

    # Manual per-class file split — guarantees every class in both train and test
    train_mask = _manual_group_split(df, test_ratio=0.5)
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"  Split: {train_mask.sum()} train / {(~train_mask).sum()} test samples")
    print(f"  Classes in train: {len(np.unique(y_train))}, test: {len(np.unique(y_test))}")

    label2int = {lbl: i for i, lbl in enumerate(class_names)}
    int2label = {i: lbl for lbl, i in label2int.items()}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    y_train_int = np.array([label2int[l] for l in y_train])

    # ── XGBoost ──
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss",
    )
    xgb_clf.fit(X_train_s, y_train_int)

    xgb_pred_int = xgb_clf.predict(X_test_s)
    xgb_pred = np.array([int2label[i] for i in xgb_pred_int])

    acc = accuracy_score(y_test, xgb_pred)
    f1 = f1_score(y_test, xgb_pred, average="macro")
    print(f"\n  XGBoost Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_test, xgb_pred, digits=4))

    # ── Random Forest (for DE_12k only) ──
    rf_acc = None
    if config_name == "DE_12k":
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_test_s)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average="macro")
        print(f"  Random Forest Accuracy: {rf_acc:.4f} | Macro F1: {rf_f1:.4f}")
        print(classification_report(y_test, rf_pred, digits=4))

    # ── Save bundle ──
    bundle = {
        "model": xgb_clf,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": class_names,
        "label2int": label2int,
        "int2label": int2label,
        "config_name": config_name,
        "fs": DATASET_CONFIG[config_name]["fs"],
        "accuracy": acc,
        "f1_macro": f1,
    }
    out_path = os.path.join(models_dir, f"model_{config_name}.pkl")
    joblib.dump(bundle, out_path)
    print(f"  Saved -> {out_path}")
    return bundle


# ──────────────────────────────────────────────────────────────────
# SHAP plots (DE_12k only)
# ──────────────────────────────────────────────────────────────────

def generate_shap_plots(df, bundle, output_dir):
    import shap

    shap_dir = os.path.join(output_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_columns"]

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    X_df = pd.DataFrame(X_scaled, columns=feature_cols)

    print("\n[SHAP] Computing SHAP values (DE_12k)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_array = np.array(shap_values).transpose(1, 2, 0)
    elif shap_values.ndim == 3:
        shap_array = shap_values
    else:
        shap_array = shap_values[:, :, np.newaxis]

    print(f"[SHAP] Shape: {shap_array.shape}")

    class_imp = np.mean(np.abs(shap_array), axis=(0, 1))
    dc = int(np.argmax(class_imp))
    signed_2d = shap_array[:, :, dc]
    mean_abs_2d = np.mean(np.abs(shap_array), axis=2)

    base_value = explainer.expected_value
    base_val_dc = float(base_value[dc]) if isinstance(base_value, (list, np.ndarray)) else float(base_value)

    # Global plots
    for name, func_args in [
        ("summary_plot.png", lambda: shap.summary_plot(mean_abs_2d, X_df, show=False, plot_size=(12, 8))),
        ("bar_plot.png", lambda: shap.summary_plot(mean_abs_2d, X_df, plot_type="bar", show=False, plot_size=(12, 8))),
        ("beeswarm_plot.png", lambda: shap.summary_plot(signed_2d, X_df, plot_type="dot", show=False, max_display=16, plot_size=(12, 8))),
    ]:
        plt.figure(figsize=(12, 8))
        func_args()
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, name), dpi=150, bbox_inches="tight")
        plt.close("all")

    # Decision plot
    n = min(300, signed_2d.shape[0])
    plt.figure(figsize=(10, 8))
    shap.decision_plot(base_val_dc, signed_2d[:n], X_df.iloc[:n], show=False)
    plt.title("SHAP Decision Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "decision_plot.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Waterfall
    exp0 = shap.Explanation(values=signed_2d[0], base_values=base_val_dc,
                            data=X_df.iloc[0].values, feature_names=list(X_df.columns))
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(exp0, show=False, max_display=16)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "waterfall_plot.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Dependence
    mean_imp = np.mean(mean_abs_2d, axis=0)
    top_feat = feature_cols[int(np.argmax(mean_imp))]
    plt.figure()
    shap.dependence_plot(top_feat, mean_abs_2d, X_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "dependence_plot.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Per fault size
    for fs_val in sorted(df["fault_size"].unique()):
        if fs_val == 0.0:
            continue
        mask = df["fault_size"] == fs_val
        if mask.sum() < 5:
            continue
        idx = np.where(mask.values)[0]
        tag = f"{fs_val:.3f}"
        sv_sub = shap_array[idx]
        signed_sub = sv_sub[:, :, dc]
        abs_sub = np.mean(np.abs(sv_sub), axis=2)
        xdf_sub = X_df.iloc[idx].reset_index(drop=True)
        print(f"[SHAP] Fault size {tag}...")

        for pname, pfunc in [
            (f"summary_plot_{tag}.png", lambda: shap.summary_plot(abs_sub, xdf_sub, show=False, plot_size=(12, 8))),
            (f"bar_plot_{tag}.png", lambda: shap.summary_plot(abs_sub, xdf_sub, plot_type="bar", show=False, plot_size=(12, 8))),
            (f"beeswarm_plot_{tag}.png", lambda: shap.summary_plot(signed_sub, xdf_sub, plot_type="dot", show=False, max_display=16, plot_size=(12, 8))),
        ]:
            plt.figure(figsize=(12, 8))
            pfunc()
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, pname), dpi=150, bbox_inches="tight")
            plt.close("all")

        n_sub = min(300, signed_sub.shape[0])
        plt.figure(figsize=(10, 8))
        shap.decision_plot(base_val_dc, signed_sub[:n_sub], xdf_sub.iloc[:n_sub], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"decision_plot_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

        exp_s = shap.Explanation(values=signed_sub[0], base_values=base_val_dc,
                                 data=xdf_sub.iloc[0].values, feature_names=list(xdf_sub.columns))
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(exp_s, show=False, max_display=16)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"waterfall_plot_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

    # Dashboard artifacts
    np.save(os.path.join(shap_dir, "shap_values.npy"), shap_array)
    mean_imp = np.mean(mean_abs_2d, axis=0)
    ranking = sorted(zip(feature_cols, mean_imp.tolist()), key=lambda x: x[1], reverse=True)
    importance = [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in ranking]
    with open(os.path.join(shap_dir, "feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    print(f"[SHAP] All plots saved to {shap_dir}/")


# ──────────────────────────────────────────────────────────────────
# Visualization plots
# ──────────────────────────────────────────────────────────────────

def plot_raw_signals(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seen = {}
    for rec in records:
        if rec["label"] not in seen:
            seen[rec["label"]] = rec["signal"]
    labels = sorted(seen.keys())
    plt.figure(figsize=(14, 3 * len(labels)))
    for i, lbl in enumerate(labels):
        sig = seen[lbl]
        n = min(5000, len(sig))
        from utils.feature_extraction import LOWCUT, HIGHCUT
        t = np.arange(n) / 12000.0
        plt.subplot(len(labels), 1, i + 1)
        plt.plot(t, sig[:n], linewidth=0.5)
        plt.title(f"Raw DE signal - {lbl}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_signals_per_class.png"), dpi=150)
    plt.close()


def plot_feature_boxplots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n_cols = 4
    n_rows = math.ceil(len(FEATURE_COLUMNS) / n_cols)
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    labels_sorted = sorted(df["label"].unique())
    for idx, feat in enumerate(FEATURE_COLUMNS):
        plt.subplot(n_rows, n_cols, idx + 1)
        data = [df[df["label"] == lbl][feat].values for lbl in labels_sorted]
        plt.boxplot(data, labels=labels_sorted, showfliers=False)
        plt.title(feat)
        plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_boxplots_by_label.png"), dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    de_12k_df = None
    de_12k_records = None

    for config_name in ["DE_12k", "FE_12k"]:
        print(f"\n{'='*60}")
        print(f"Training: {config_name}")
        print(f"{'='*60}")

        records = load_signals(config_name)
        if len(records) == 0:
            print(f"  No data found. Skipping.")
            continue

        cfg = DATASET_CONFIG[config_name]
        df = build_features(records, fs=cfg["fs"])
        if df.empty or df["label"].nunique() < 2:
            print(f"  Insufficient data. Skipping.")
            continue

        df = balance_classes(df)
        print(f"  Balanced: {df.shape[0]} samples, {df['label'].nunique()} classes")
        print(f"  Classes: {sorted(df['label'].unique())}")

        train_model(df, config_name, models_dir)

        if config_name == "DE_12k":
            de_12k_df = df.copy()
            de_12k_records = records
            df.to_csv(os.path.join(output_dir, "cwru_features_DE_12k.csv"), index=False)

    # Plots + SHAP for DE_12k
    if de_12k_df is not None and de_12k_records is not None:
        plot_raw_signals(de_12k_records, output_dir)
        plot_feature_boxplots(de_12k_df, output_dir)
        de_12k_bundle = joblib.load(os.path.join(models_dir, "model_DE_12k.pkl"))
        generate_shap_plots(de_12k_df, de_12k_bundle, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"{'='*60}")
    for f in sorted(os.listdir(models_dir)):
        if f.endswith(".pkl"):
            b = joblib.load(os.path.join(models_dir, f))
            print(f"  {f}: {len(b['class_names'])} classes, "
                  f"acc={b['accuracy']:.4f}, fs={b['fs']:.0f}Hz")


if __name__ == "__main__":
    main()
