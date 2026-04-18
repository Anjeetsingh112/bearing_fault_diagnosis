"""Model loading and prediction - multi-model aware."""

import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_CACHE = {}


def get_config_name(sensor, rate):
    """Convert user selection to config name.

    sensor: 'Drive End (DE)' or 'Fan End (FE)'
    rate: '12k' or '48k'
    """
    s = "DE" if "DE" in sensor or "Drive" in sensor else "FE"
    r = rate.replace("k", "k")
    return f"{s}_{r}"


def load_bundle(config_name):
    """Load and cache a model bundle."""
    if config_name in _CACHE:
        return _CACHE[config_name]

    path = os.path.join(MODELS_DIR, f"model_{config_name}.pkl")
    if not os.path.exists(path):
        return None

    bundle = joblib.load(path)
    _CACHE[config_name] = bundle
    return bundle


def list_available_models():
    """Return list of available config names."""
    available = []
    if not os.path.isdir(MODELS_DIR):
        return available
    for f in os.listdir(MODELS_DIR):
        if f.startswith("model_") and f.endswith(".pkl"):
            name = f.replace("model_", "").replace(".pkl", "")
            available.append(name)
    return sorted(available)


def predict(features_dict, config_name):
    """Predict using the specified model.

    Returns: (label, proba_array, class_names) or raises ValueError.
    """
    bundle = load_bundle(config_name)
    if bundle is None:
        raise ValueError(f"Model not found: model_{config_name}.pkl")

    x = np.array([[features_dict[c] for c in bundle["feature_columns"]]])
    x_scaled = bundle["scaler"].transform(x)
    pred_int = int(bundle["model"].predict(x_scaled)[0])
    proba = bundle["model"].predict_proba(x_scaled)[0]
    label = bundle["int2label"][pred_int]
    return label, proba, bundle["class_names"]


def get_scaled_features(features_dict, config_name):
    """Scale features using the model's scaler."""
    bundle = load_bundle(config_name)
    if bundle is None:
        raise ValueError(f"Model not found: model_{config_name}.pkl")
    x = np.array([[features_dict[c] for c in bundle["feature_columns"]]])
    return bundle["scaler"].transform(x)
