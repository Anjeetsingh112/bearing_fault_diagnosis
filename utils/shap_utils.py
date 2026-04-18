"""SHAP explanation utilities - works for ALL models in dashboard."""

import numpy as np
import shap
from utils.model_loader import load_bundle

# DE_12k has pre-generated plots in outputs/shap/ for reports.
# All models compute SHAP live in the dashboard.
PRIMARY_CONFIG = "DE_12k"


def get_shap_explanation(scaled_features, config_name):
    """Compute SHAP values for a single scaled sample using any model.

    Returns:
        shap_for_pred: (n_features,) signed SHAP for predicted class
        base_value: float
        feature_names: list
        all_shap: (n_features, n_classes)
    """
    bundle = load_bundle(config_name)
    if bundle is None:
        return None, None, None, None

    model = bundle["model"]
    feature_names = list(bundle["feature_columns"])

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(scaled_features)

    if isinstance(sv, list):
        all_shap = np.array([s[0] for s in sv]).T
    elif sv.ndim == 3:
        all_shap = sv[0]
    else:
        all_shap = sv[0][:, np.newaxis]

    pred_int = int(model.predict(scaled_features)[0])
    shap_for_pred = all_shap[:, pred_int]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base_val = float(base[pred_int])
    else:
        base_val = float(base)

    return shap_for_pred, base_val, feature_names, all_shap


def top_reasons(shap_values_1d, feature_names, n=3):
    """Return top-n human-readable reasons."""
    order = np.argsort(np.abs(shap_values_1d))[::-1][:n]
    reasons = []
    for i in order:
        v = shap_values_1d[i]
        arrow = "\u2191" if v > 0 else "\u2193"
        direction = "increases" if v > 0 else "decreases"
        reasons.append({
            "feature": feature_names[i],
            "shap_value": float(v),
            "text": f"{feature_names[i]} {arrow} {direction} fault probability",
        })
    return reasons
