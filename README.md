# SHAP-Enhanced Digital Twin Bearing Fault Diagnosis System

A production-grade, explainable AI system for rolling bearing fault diagnosis using vibration signals from the CWRU (Case Western Reserve University) Bearing Dataset. The system combines multi-model XGBoost classification with SHAP (SHapley Additive exPlanations) interpretability analysis, following a Digital Twin architecture.

---

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bearing-diagnosis-2022ugme112.streamlit.app)

**Live Demo:** https://bearing-diagnosis-2022ugme112.streamlit.app


## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Dataset](#3-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [Models](#5-models)
6. [SHAP Explainability](#6-shap-explainability)
7. [Dashboard](#7-dashboard)
8. [Installation & Setup](#8-installation--setup)
9. [How to Run](#9-how-to-run)
10. [Project Structure](#10-project-structure)
11. [Technical Details](#11-technical-details)
12. [Results](#12-results)
13. [SHAP Analysis Results](#13-shap-analysis-results)
14. [Limitations](#14-limitations)
15. [Future Work](#15-future-work)
16. [References](#16-references)

---

## 1. System Overview

Rolling bearings are the most critical components in rotating machinery. Approximately 30-40% of all rotating machinery failures originate from bearing defects. Early and accurate diagnosis of bearing faults is essential for predictive maintenance and preventing catastrophic equipment failure.

This system provides:

- **Multi-model fault classification** across 2 sensor configurations (DE and FE, both at 12 kHz)
- **12-class fault diagnosis** covering Normal, Inner Race, Outer Race, and Ball faults at 4 severity levels (DE model)
- **SHAP-based explainability** revealing which signal features drive each prediction
- **Interactive Streamlit dashboard** for real-time diagnosis with visual explanations
- **Digital Twin architecture** connecting physical vibration signals to AI-driven health assessment

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| Fault Detection | Detects bearing faults from raw vibration signals |
| Fault Classification | Identifies fault type (IR, OR, Ball) and severity (0.007-0.028 inch) |
| Explainability | SHAP analysis explains why the model made each prediction |
| Multi-sensor Support | Drive End and Fan End accelerometer data |
| Sampling Rate | 12 kHz (standardized for consistency) |
| Dashboard | Interactive web UI for signal upload and diagnosis |

---

## 2. Architecture

The system follows a 4-layer Digital Twin architecture:

```
+------------------+     +-------------------+     +------------------+     +------------------+
|  Physical Layer  | --> |    Data Layer      | --> |   Model Layer    | --> |  Service Layer   |
|                  |     |                    |     |                  |     |                  |
| Vibration Signal |     | Bandpass Filter    |     | XGBoost (x2)    |     | Streamlit UI     |
| (.mat / .csv)    |     | Sliding Window     |     | Random Forest   |     | Signal Plots     |
| DE / FE sensor   |     | 16 Features        |     | SHAP Explainer  |     | SHAP Plots       |
| 12 kHz           |     | StandardScaler     |     | TreeExplainer   |     | Alert System     |
+------------------+     +-------------------+     +------------------+     +------------------+
```

### Data Flow

1. **Input**: Raw vibration signal (.mat or .csv file)
2. **Preprocessing**: Bandpass filter (20-5000 Hz) using Butterworth 4th-order filter
3. **Segmentation**: Sliding window (1024 samples, 50% overlap)
4. **Feature Extraction**: 16 features across time, frequency, and time-frequency domains
5. **Scaling**: StandardScaler (fit only on training data, no data leakage)
6. **Prediction**: XGBoost multi-class classifier
7. **Explanation**: SHAP TreeExplainer (both DE_12k and FE_12k models)
8. **Output**: Fault class, confidence, severity level, SHAP explanation

---

## 3. Dataset

### 3.1 CWRU Bearing Dataset

The Case Western Reserve University Bearing Data Center provides vibration data collected from a motor-driven mechanical system under controlled laboratory conditions.

**Experimental Setup:**
- Motor: 2 HP Reliance Electric motor
- Bearings: SKF deep groove ball bearings (6205-2RS JEM)
- Faults: Introduced using electro-discharge machining (EDM)
- Sensors: Accelerometers mounted at Drive End (DE) and Fan End (FE) bearing housings
- Sampling Rate: 12,000 Hz (standardized for this project)

### 3.2 Fault Types

| Fault Type | Abbreviation | Description |
|-----------|-------------|-------------|
| Normal | Normal | Healthy bearing, no defect |
| Inner Race | IR | Defect on the bearing inner raceway |
| Outer Race | OR | Defect on the bearing outer raceway (6 o'clock position) |
| Ball | Ball | Defect on the rolling element surface |

### 3.3 Fault Severities

| Fault Size (inches) | Fault Size (mm) | Severity Level |
|---------------------|-----------------|----------------|
| 0.007 | 0.178 | Incipient / Minor |
| 0.014 | 0.356 | Moderate |
| 0.021 | 0.533 | Significant |
| 0.028 | 0.711 | Severe (DE only) |

### 3.4 Class Distribution

**DE_12k Model - 12 Classes**

| Class | Fault Type | Size | Files | Windows (balanced) |
|-------|-----------|------|-------|--------------------|
| Normal | None | 0.0 | 4 | 940 |
| IR_007 | Inner Race | 0.007 | 4 | 940 |
| IR_014 | Inner Race | 0.014 | 4 | 940 |
| IR_021 | Inner Race | 0.021 | 4 | 940 |
| IR_028 | Inner Race | 0.028 | 4 | 940 |
| OR_007 | Outer Race | 0.007 | 4 | 940 |
| OR_014 | Outer Race | 0.014 | 4 | 940 |
| OR_021 | Outer Race | 0.021 | 4 | 940 |
| Ball_007 | Ball | 0.007 | 4 | 940 |
| Ball_014 | Ball | 0.014 | 4 | 940 |
| Ball_021 | Ball | 0.021 | 4 | 940 |
| Ball_028 | Ball | 0.028 | 4 | 940 |
| **Total** | | | **48** | **11,280** |

**FE_12k Model - 10 Classes**

The Fan End dataset does not include 0.028 inch fault sizes, so FE_12k has 10 classes:
Normal, IR_007, IR_014, IR_021, OR_007, OR_014, OR_021, Ball_007, Ball_014, Ball_021.

Each file corresponds to one motor load condition (0, 1, 2, or 3 HP).

### 3.5 Dataset Folder Structure

```
CWRU-dataset/
  Normal/
    97_Normal_0.mat          # Normal, 0 HP load
    98_Normal_1.mat          # Normal, 1 HP load
    99_Normal_2.mat          # Normal, 2 HP load
    100_Normal_3.mat         # Normal, 3 HP load
  12k_Drive_End_Bearing_Fault_Data/
    IR/007/                  # Inner Race, 0.007 inch
      105_0.mat ... 108_3.mat
    IR/014/
      169_0.mat ... 172_3.mat
    IR/021/
      209_0.mat ... 212_3.mat
    IR/028/
      3001_0.mat ... 3004_3.mat
    OR/007/@6/               # Outer Race, 0.007 inch, 6 o'clock position
      130@6_0.mat ... 133@6_3.mat
    OR/014/
      197@6_0.mat ... 200@6_3.mat
    OR/021/@6/
      234_0.mat ... 237_3.mat
    B/007/                   # Ball fault, 0.007 inch
      118_0.mat ... 121_3.mat
    B/014/
      185_0.mat ... 188_3.mat
    B/021/
      222_0.mat ... 225_3.mat
    B/028/
      3005_0.mat ... 3008_3.mat
  12k_Fan_End_Bearing_Fault_Data/
    (similar structure with FE_time signals, 10 classes - no 0.028 faults)
```

### 3.6 .mat File Contents

Each `.mat` file contains MATLAB arrays:

| Key Pattern | Description |
|------------|-------------|
| `X{ID}_DE_time` | Drive End accelerometer signal (time domain) |
| `X{ID}_FE_time` | Fan End accelerometer signal (time domain) |
| `X{ID}_BA_time` | Base accelerometer signal (not used) |
| `X{ID}RPM` | Motor RPM |

---

## 4. Feature Engineering

### 4.1 Preprocessing

**Bandpass Filter:**
- Type: Butterworth, 4th order
- Passband: 20 Hz to 5,000 Hz
- Purpose: Remove DC offset, low-frequency drift, and high-frequency noise
- Applied at 12 kHz sampling rate

**Sliding Window Segmentation:**
- Window size: 1024 samples
- Overlap: 50% (512-sample step)
- Each window = 85.3 ms of signal at 12 kHz

### 4.2 Feature Descriptions (16 Features)

#### Time-Domain Features (8)

| # | Feature | Formula | Physical Meaning |
|---|---------|---------|-----------------|
| 1 | **RMS** | sqrt(mean(x^2)) | Overall vibration energy / signal power |
| 2 | **Peak2Peak** | max(x) - min(x) | Maximum amplitude range of vibration |
| 3 | **Kurtosis** | E[(x-mu)^4] / sigma^4 - 3 | Impulsiveness of the signal; high kurtosis indicates sharp impacts |
| 4 | **Skewness** | E[(x-mu)^3] / sigma^3 | Asymmetry of the vibration distribution |
| 5 | **Crest Factor** | max(|x|) / RMS | Ratio of peak to effective value; detects sharp peaks |
| 6 | **Impulse Factor** | max(|x|) / mean(|x|) | Sensitivity to impulse-type faults |
| 7 | **Shape Factor** | RMS / mean(|x|) | Signal waveform shape indicator |
| 8 | **Clearance Factor** | max(|x|) * geo_mean(|x|) / mean(|x|)^2 | Combines peak and distribution information |

#### Frequency-Domain Features (4)

| # | Feature | Formula | Physical Meaning |
|---|---------|---------|-----------------|
| 9 | **Spectral Entropy** | -sum(P_k * log2(P_k)) | Complexity/flatness of the frequency spectrum; fault signals have more complex spectra |
| 10 | **Peak Frequency** | argmax(|FFT(x)|) | Dominant frequency component |
| 11 | **Frequency Centroid** | sum(f * |FFT|) / sum(|FFT|) | Center of mass of the spectrum |
| 12 | **Band Energy** | sum(|FFT|^2) for 2-4 kHz | Energy in the bearing fault characteristic frequency band |

#### Time-Frequency Features (4) - Discrete Wavelet Transform

| # | Feature | Wavelet | Physical Meaning |
|---|---------|---------|-----------------|
| 13 | **Wavelet_cA3** | db4, level 3 approx. | Low-frequency energy |
| 14 | **Wavelet_cD3** | db4, level 3 detail | Mid-low frequency energy |
| 15 | **Wavelet_cD2** | db4, level 2 detail | Mid-high frequency energy |
| 16 | **Wavelet_cD1** | db4, level 1 detail | High frequency energy |

The Daubechies-4 (db4) wavelet is chosen for its balance of time and frequency localization, well-suited for detecting transient impacts in bearing vibration.

### 4.3 Why These Features?

The 16 features are designed to capture bearing fault signatures across three complementary domains:

- **Time-domain**: Captures amplitude changes (RMS increases with fault severity), impulsiveness (Kurtosis spikes on impact faults), and waveform distortion
- **Frequency-domain**: Captures spectral complexity changes (Spectral Entropy increases when faults introduce new frequency components) and energy shifts to characteristic fault frequencies
- **Time-frequency**: Wavelet features capture transient, non-stationary fault impulses that are localized in both time and frequency

### 4.4 Class Balancing

After feature extraction, classes are balanced by downsampling to the minority class count (940 samples per class for DE_12k). This prevents the model from being biased toward classes with more data. Random sampling with a fixed seed (42) ensures reproducibility.

---

## 5. Models

### 5.1 Multi-Model Architecture

Two separate XGBoost models are trained, one for each sensor location:

| Model | Sensor | Rate | Signal Key | Classes | SHAP |
|-------|--------|------|-----------|---------|------|
| **model_DE_12k.pkl** | Drive End | 12 kHz | DE_time | 12 | Yes |
| **model_FE_12k.pkl** | Fan End | 12 kHz | FE_time | 10 | Yes |

**Why separate models?** The Drive End and Fan End sensors record different signals because they are located at different positions. DE sensor is closer to the faulty bearing (stronger signal). FE sensor picks up attenuated vibrations traveling through the shaft (weaker, noisier signal). A single model would perform worse on both; separate models exploit each sensor's specific characteristics.

### 5.2 XGBoost Configuration

```python
XGBClassifier(
    n_estimators=500,      # Number of boosting rounds
    max_depth=8,           # Maximum tree depth
    learning_rate=0.03,    # Step size shrinkage
    subsample=0.8,         # Row sampling per tree
    colsample_bytree=0.8,  # Feature sampling per tree
    min_child_weight=1,    # Minimum sum of instance weight in a child
    eval_metric="mlogloss" # Multi-class log loss
)
```

- `subsample=0.8` and `colsample_bytree=0.8` provide regularization against overfitting
- `max_depth=8` allows sufficient complexity for 12-class discrimination
- `learning_rate=0.03` with 500 trees provides gradual, stable learning

### 5.3 Random Forest (Baseline)

A Random Forest (100 trees) is also trained for the DE_12k configuration as a baseline comparison. It achieves 98.29% accuracy, essentially tied with XGBoost's 98.41%. XGBoost is preferred for SHAP compatibility.

### 5.4 Train/Test Split Strategy

**Stratified Group K-Fold cross-validation (n_splits=2)**

This project uses sklearn's `StratifiedGroupKFold` — the industry-standard approach that simultaneously addresses two concerns:

- **Stratified:** Keeps class proportions balanced between train and test folds (prevents majority-class bias)
- **Grouped:** All windows from the same `.mat` file go to the same fold (prevents data leakage from adjacent, highly-correlated windows)

**Implementation:**

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
train_idx, test_idx = next(sgkf.split(X, y, groups=filenames))
```

**Why 2 splits (50/50)?** With only 4 files per class, n_splits=2 puts 2 files in train and 2 in test per class — the maximum split that still gives meaningful train/test data.

**Seed retry logic:** With so few files per class, the algorithm can occasionally miss a class in one fold. The training loop tries multiple random seeds until it finds a split where **every class appears in both train and test**.

**Why not a simple random window split?** Adjacent windows in the same signal are highly correlated. A random split would allow windows from the same signal to appear in both train and test, inflating accuracy artificially. Stratified Group K-Fold prevents this.

### 5.5 Feature Scaling

StandardScaler is fit **only on training data** and applied to test data. This prevents information leakage from test set statistics into the training process.

### 5.6 Model Bundle Format

Each model is saved as a single `.pkl` file containing:

```python
{
    "model": XGBClassifier,       # Trained model
    "scaler": StandardScaler,     # Fitted scaler
    "feature_columns": list,      # 16 feature names
    "class_names": list,          # Sorted class labels
    "label2int": dict,            # Label -> integer mapping
    "int2label": dict,            # Integer -> label mapping
    "config_name": str,           # e.g., "DE_12k"
    "fs": float,                  # Sampling rate (Hz)
    "accuracy": float,            # Test accuracy
    "f1_macro": float,            # Test macro F1 score
}
```

---

## 6. SHAP Explainability

### 6.1 What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain individual predictions. Based on Shapley values from cooperative game theory, SHAP assigns each feature an importance value for a particular prediction:

```
f(x) = base_value + SHAP_1 + SHAP_2 + ... + SHAP_16
```

Where:
- `f(x)` is the model's output for sample x
- `base_value` is the average model output across the training set
- `SHAP_i` is the contribution of feature i to this specific prediction

A positive SHAP value means the feature pushes the prediction toward the positive class; negative means it pushes away.

### 6.2 TreeExplainer

For tree-based models like XGBoost, SHAP provides `TreeExplainer` which computes exact Shapley values in polynomial time (not exponential), making it practical for production use.

### 6.3 SHAP Availability

SHAP analysis is **available for both models (DE_12k and FE_12k)** in the dashboard. Live SHAP computation runs for whichever model the user selects.

Pre-generated SHAP report plots are only stored for DE_12k (primary model), since it has the most classes and highest accuracy.

### 6.4 SHAP Plots Generated

#### Global Plots (DE_12k, all data)

| Plot | File | Description |
|------|------|-------------|
| Summary Plot | `summary_plot.png` | Dot plot showing each feature's SHAP impact distribution |
| Bar Plot | `bar_plot.png` | Mean absolute SHAP value per feature (importance ranking) |
| Beeswarm Plot | `beeswarm_plot.png` | Signed SHAP values colored by feature value (red=high, blue=low) |
| Decision Plot | `decision_plot.png` | Shows how features cumulatively build the prediction from base value |
| Waterfall Plot | `waterfall_plot.png` | Single sample breakdown showing each feature's contribution |
| Dependence Plot | `dependence_plot.png` | Top feature's value vs. its SHAP value (non-linear relationships) |

#### Per-Fault-Size Plots

For each fault severity (0.007, 0.014, 0.021, 0.028), the same set of plots is generated:

- `summary_plot_0.007.png`, `bar_plot_0.007.png`, `beeswarm_plot_0.007.png`, `decision_plot_0.007.png`, `waterfall_plot_0.007.png`
- Same for `0.014`, `0.021`, `0.028`

This allows comparing how feature importance changes across fault severities.

#### Dashboard Artifacts

| File | Format | Purpose |
|------|--------|---------|
| `shap_values.npy` | NumPy array (11280 x 16 x 12) | Raw SHAP values for all samples |
| `feature_importance.json` | JSON | Ranked feature importance for dashboard display |

### 6.5 How to Read SHAP Plots

**Bar Plot**: Features ranked top-to-bottom by mean absolute SHAP value. Higher bar = more important feature.

**Beeswarm Plot**: Each dot is one sample. X-axis = SHAP value (impact on prediction). Color = feature value (red = high, blue = low). Shows both importance AND direction.

**Decision Plot**: Each line is one sample's prediction path. Lines start at the base value and accumulate feature contributions. Different colors correspond to different predicted classes.

**Waterfall Plot**: Single sample breakdown. Each row is a feature. Red bars push prediction up, blue bars push down. Shows exactly how the model reached its decision.

**Dependence Plot**: Scatter plot of one feature's value vs. its SHAP value. Reveals non-linear relationships and interaction effects.

---

## 7. Dashboard

### 7.1 Overview

The Streamlit dashboard provides a real-time interactive interface for bearing fault diagnosis. It implements the Service Layer of the Digital Twin architecture.

### 7.2 Dashboard Sections

#### Header
- System title and description
- Live status badge: STANDBY (gray), HEALTHY (green), WARNING (yellow), CRITICAL (red)

#### Sidebar (Control Panel)
- File upload (.mat or .csv)
- Sensor type selector: Drive End (DE) or Fan End (FE)
- Fixed at 12 kHz sampling rate
- Active model information (name, classes, accuracy)
- SHAP availability indicator
- Model Limitations panel
- How to Use guide

#### Diagnosis Result (Metric Cards)
- Fault Type (e.g., "Inner Race")
- Fault Class (e.g., "IR_014")
- Confidence percentage
- Fault Size (e.g., "0.014 in")
- Active Model (e.g., "DE_12k")

#### Alert System
Color-coded alert box based on severity:

| Confidence | Label | Color | Action |
|-----------|-------|-------|--------|
| Any | Normal | Green | No action needed |
| >= 90% | Fault | Red (Critical) | Immediate inspection recommended |
| 70-90% | Fault | Yellow (Warning) | Schedule maintenance |
| < 70% | Fault | Green (Low Risk) | Continue monitoring |

#### Tabs
1. **Signal View** - Raw + filtered signal (Plotly interactive)
2. **FFT Spectrum** - Frequency magnitude + peak marker
3. **Prediction** - Probability distribution across all classes
4. **SHAP Explanation** - Top reasons, bar chart, waterfall chart
5. **Feature Insights** - Table of 16 features with abnormal value flagging
6. **History** - Last 10 predictions with confidence trend

### 7.3 Input Validation

The dashboard validates:
- File format (.mat or .csv only)
- Signal key presence (DE_time / FE_time matching sensor selection)
- Signal length (minimum 1024 samples)
- Signal validity (at least some finite values)
- Feature extraction success (no NaN/Inf)
- Model availability

Clear error messages guide users to fix issues.

---

## 8. Installation & Setup

### 8.1 Prerequisites

- Python 3.9 or later
- pip package manager

### 8.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install streamlit plotly shap xgboost scikit-learn scipy PyWavelets joblib pandas numpy matplotlib
```

### 8.3 Dataset

Place the CWRU dataset in `CWRU-dataset/` directory alongside `pipeline.py`. The expected structure is shown in Section 3.5.

---

## 9. How to Run

### 9.1 Train Models

```bash
python pipeline.py
```

This will:
1. Load signals for DE_12k and FE_12k configurations
2. Extract 16 features per window at 12 kHz
3. Balance classes and split data using Stratified Group K-Fold (50/50)
4. Train XGBoost (+ Random Forest for DE_12k)
5. Save model bundles to `models/`
6. Generate SHAP plots for DE_12k to `outputs/shap/`
7. Save feature CSV to `outputs/`

Training takes approximately 3-5 minutes.

### 9.2 Launch Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### 9.3 Using the Dashboard

1. Select sensor type (Drive End or Fan End) in the sidebar
2. Upload a `.mat` file from the CWRU dataset
3. Wait for processing (progress bar shows pipeline stages)
4. View diagnosis result, alert, and explore tabs
5. SHAP explanation is available for both models

### 9.4 Example Inputs

| File | Select | Expected Result |
|------|--------|----------------|
| `CWRU-dataset/Normal/97_Normal_0.mat` | DE | Normal (healthy) |
| `CWRU-dataset/12k_Drive_End_.../IR/007/105_0.mat` | DE | IR_007 + SHAP |
| `CWRU-dataset/12k_Drive_End_.../OR/021/@6/234_0.mat` | DE | OR_021 + SHAP |
| `CWRU-dataset/12k_Drive_End_.../B/028/3005_0.mat` | DE | Ball_028 + SHAP |
| `CWRU-dataset/12k_Fan_End_.../IR/021/270_0.mat` | FE | IR_021 + SHAP |

---

## 10. Project Structure

```
project/
  pipeline.py                   # Multi-model training pipeline
  app.py                        # Streamlit dashboard
  README.md                     # This documentation
  DEPLOYMENT.md                 # Deployment guide
  requirements.txt              # Python dependencies
  .gitignore                    # Git exclusions
  .streamlit/
    config.toml                 # Dashboard theme config
  utils/
    __init__.py
    feature_extraction.py       # Signal processing + 16 features
    model_loader.py             # Multi-model loading, prediction, scaling
    shap_utils.py               # SHAP explanation (any model)
  models/
    model_DE_12k.pkl            # Primary model (12 classes)
    model_FE_12k.pkl            # Fan End model (10 classes)
  outputs/
    cwru_features_DE_12k.csv    # Extracted features dataset
    raw_signals_per_class.png   # Signal visualization
    feature_boxplots_by_label.png # Feature distributions
    shap/
      summary_plot.png          # Global SHAP plots (DE_12k only)
      bar_plot.png
      beeswarm_plot.png
      decision_plot.png
      waterfall_plot.png
      dependence_plot.png
      summary_plot_0.007.png    # Per-fault-size plots
      ... (4 sizes x 5 plot types)
      feature_importance.json   # Ranked importance
      shap_values.npy           # Raw SHAP values
  CWRU-dataset/                 # Raw CWRU bearing data (not in git)
```

---

## 11. Technical Details

### 11.1 Data Leakage Prevention

Three safeguards prevent data leakage:

1. **Stratified Group K-Fold split**: Windows from the same signal file never appear in both train and test. Groups (files) are kept together while class proportions are balanced across folds.
2. **Post-split scaling**: `StandardScaler` is fit only on training data, then applied to test data — test statistics never influence the scaler.
3. **Class balance**: Classes are downsampled to the minority count (940 windows per class for DE_12k) before splitting, preventing majority-class bias.

### 11.2 File ID Mapping

Each .mat file is identified by the numeric prefix of its filename (e.g., `105_0.mat` -> ID 105). An explicit ID-to-label mapping dictionary ensures correct labeling. This avoids the fragility of parsing labels from directory paths with special characters (e.g., `@6`).

### 11.3 Outer Race Position

For Outer Race faults, the CWRU dataset provides data at 3 positions (@3, @6, @12 o'clock). This system uses only the **@6 o'clock position** for consistency, as it is the most commonly used in literature and provides the clearest fault signatures (load zone).

### 11.4 Sampling Rate Standardization

Only 12 kHz data is used. This was chosen because:
- 12 kHz provides sufficient bandwidth for bearing fault frequencies
- Standardized sampling rate eliminates feature scale inconsistencies
- 1024-sample window at 12 kHz = 85 ms, enough to capture several fault impact events

### 11.5 SHAP Multi-Class Handling

XGBoost multi-class SHAP values have shape `(n_samples, n_features, n_classes)`. For visualization:
- **Bar plot / Summary plot**: Uses mean absolute SHAP across all classes
- **Beeswarm / Decision / Waterfall**: Uses SHAP values for the **dominant class** (class with highest mean |SHAP|), preserving sign information for meaningful visualization

---

## 12. Results

### 12.1 Model Performance

| Model | Classes | Accuracy | Macro F1 | Weighted F1 |
|-------|---------|----------|----------|-------------|
| **DE_12k (XGBoost)** | 12 | **98.41%** | **98.46%** | 98.40% |
| DE_12k (Random Forest) | 12 | 98.29% | 98.48% | 98.29% |
| **FE_12k (XGBoost)** | 10 | **75.82%** | **70.04%** | 70.67% |

### 12.2 DE_12k Per-Class Performance (XGBoost)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.9667 | 1.0000 | 0.9831 |
| IR_007 | 0.9986 | 1.0000 | 0.9993 |
| IR_014 | 0.9935 | 0.9851 | 0.9893 |
| IR_021 | 0.9985 | 0.9575 | 0.9776 |
| IR_028 | 1.0000 | 1.0000 | 1.0000 |
| OR_007 | 0.9957 | 0.9979 | 0.9968 |
| OR_014 | 0.9979 | 0.9958 | 0.9968 |
| OR_021 | 1.0000 | 1.0000 | 1.0000 |
| Ball_007 | 0.9355 | 0.9915 | 0.9627 |
| Ball_014 | 0.9341 | 0.9816 | 0.9572 |
| Ball_021 | 0.9819 | 0.9255 | 0.9529 |
| Ball_028 | 1.0000 | 1.0000 | 1.0000 |

### 12.3 Performance Analysis

**Strongest classes** (F1 >= 0.99): IR_007, IR_014, IR_028, OR_007, OR_014, OR_021, Ball_028 — all above 0.987 F1.

**Moderate classes** (F1 0.95-0.99): Normal, Ball_007, Ball_014, Ball_021, IR_021 — still excellent performance.

**No class below 0.95 F1** — DE_12k shows strong, consistent discrimination across all 12 fault types.

**Why DE_12k outperforms FE_12k**: The Drive End sensor is physically closer to the test bearing, capturing clearer fault signatures. The Fan End sensor receives attenuated vibrations traveling through the shaft, and the particular Stratified Group K-Fold split assigned some files in ways that challenge the model. FE_12k performance has higher variance across random seeds due to having only 4 files per class.

---

## 13. SHAP Analysis Results

### 13.1 Global Feature Importance (DE_12k)

Top features by mean absolute SHAP value:

| Rank | Feature | Domain |
|------|---------|--------|
| 1 | Wavelet_cD3 | Time-Frequency |
| 2 | FreqCentroid | Frequency |
| 3 | Wavelet_cA3 | Time-Frequency |
| 4 | Shape | Time |
| 5 | PeakFreq | Frequency |

### 13.2 Feature Importance by Domain

| Domain | Features in Top 5 |
|--------|-------------------|
| Time-Frequency (Wavelet) | Wavelet_cD3, Wavelet_cA3 |
| Frequency | FreqCentroid, PeakFreq |
| Time | Shape |

Wavelet features dominate, capturing the transient, non-stationary nature of bearing fault impulses.

### 13.3 Per-Fault-Size Insights

Comparing SHAP importance across fault severities reveals how diagnostic patterns change:

- **0.007 (incipient)**: Subtle features like Spectral Entropy and Shape Factor are relatively more important, as fault signals are weak
- **0.014-0.021 (moderate)**: Wavelet features and Frequency Centroid dominate as fault impacts become more energetic
- **0.028 (severe)**: Energy features (RMS, Wavelet energies) become more prominent as faults produce large-amplitude vibrations

---

## 14. Limitations

1. **Dataset scope**: Trained only on CWRU laboratory data. Real industrial bearings may have different fault characteristics, noise levels, and operating conditions.

2. **Artificial faults**: CWRU faults are introduced via EDM (electro-discharge machining), which produces different defect geometry than natural fatigue-induced faults.

3. **Single bearing type**: Only SKF 6205-2RS deep groove ball bearings. Other bearing types (roller, tapered, thrust) are not covered.

4. **Fixed operating conditions**: Data collected under constant speed and 4 discrete load levels. Variable speed or transient conditions are not addressed.

5. **Single sampling rate**: Only 12 kHz is supported. Input signals at other rates must be resampled before upload.

6. **FE_12k limitation**: Fan End model has 10 classes (no 0.028 fault size) because the CWRU Fan End dataset does not include 0.028-inch faults.

7. **No temporal tracking**: The system classifies individual signal windows independently. It does not track fault progression over time.

---

## 15. Future Work

### 15.1 System Extensions

- **Real-time streaming**: Replace file upload with WebSocket/MQTT sensor feed for continuous monitoring
- **FastAPI backend**: Wrap `utils/` functions in REST API for integration with existing industrial systems
- **IoT gateway integration**: Connect directly to industrial sensor hardware
- **Multi-bearing monitoring**: Extend dashboard to simultaneously monitor multiple bearing positions

### 15.2 Model Improvements

- **Transfer learning**: Pre-train on CWRU, fine-tune on target industrial data
- **Ensemble model**: Combine XGBoost and Random Forest predictions
- **Temporal models**: LSTM or Transformer-based models for fault progression tracking
- **Additional features**: Envelope analysis, spectral kurtosis, bearing characteristic frequencies (BPFI, BPFO, BSF, FTF)

### 15.3 Research Directions

- **Natural fault validation**: Test on bearings with naturally evolved faults
- **Cross-domain adaptation**: Apply to different bearing types and operating conditions
- **Fault severity estimation**: Regression-based fault size estimation instead of discrete classification
- **Remaining useful life (RUL)**: Predict time-to-failure from fault progression patterns

---

## 16. References

1. CWRU Bearing Data Center. Case Western Reserve University. https://engineering.case.edu/bearingdatacenter

2. Lundberg, S.M. and Lee, S.I., 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.

3. Wang, L. and Wu, M., 2025. Research on bearing fault diagnosis based on machine learning and SHAP interpretability analysis. Scientific Reports, 15, 41242.

4. Chen, T. and Guestrin, C., 2016. XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

5. Randall, R.B. and Antoni, J., 2011. Rolling element bearing diagnostics - A tutorial. Mechanical Systems and Signal Processing, 25(2), pp.485-520.

---

*Built as part of B.Tech Project (BTP) - Explainable Bearing Fault Diagnosis System*
