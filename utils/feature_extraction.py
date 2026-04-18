"""Feature extraction utilities - sampling-rate aware."""

import numpy as np
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy import stats

LOWCUT = 20.0
HIGHCUT = 5000.0
FILTER_ORDER = 4
WINDOW_SIZE = 1024
OVERLAP = 0.5

FEATURE_COLUMNS = [
    "RMS", "Peak2Peak", "Kurtosis", "Skewness",
    "Crest", "Impulse", "Shape", "Clearance",
    "SpectralEntropy", "PeakFreq", "FreqCentroid", "BandEnergy",
    "Wavelet_cA3", "Wavelet_cD3", "Wavelet_cD2", "Wavelet_cD1",
]

NORMAL_RANGES = {
    "RMS":             (0.01, 0.25),
    "Peak2Peak":       (0.05, 1.5),
    "Kurtosis":        (-1.0, 4.0),
    "Skewness":        (-1.0, 1.0),
    "Crest":           (2.0, 6.0),
    "Impulse":         (2.5, 8.0),
    "Shape":           (1.1, 1.6),
    "Clearance":       (1.0, 3.0),
    "SpectralEntropy": (4.0, 9.0),
    "PeakFreq":        (0, 3000),
    "FreqCentroid":    (500, 3500),
    "BandEnergy":      (0, 50),
    "Wavelet_cA3":     (0, 20),
    "Wavelet_cD3":     (0, 10),
    "Wavelet_cD2":     (0, 5),
    "Wavelet_cD1":     (0, 2),
}

_BP_CACHE = {}


def _bp_coeffs(fs):
    if fs not in _BP_CACHE:
        nyq = 0.5 * fs
        low = max(LOWCUT / nyq, 1e-5)
        high = min(HIGHCUT / nyq, 0.9999)
        _BP_CACHE[fs] = butter(FILTER_ORDER, [low, high], btype="band")
    return _BP_CACHE[fs]


def preprocess_signal(signal, fs=12000.0):
    """Bandpass filter using the correct sampling rate."""
    b, a = _bp_coeffs(fs)
    try:
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def segment_signal(signal, window_size=WINDOW_SIZE, overlap=OVERLAP):
    step = int(window_size * (1 - overlap))
    wins = [signal[i:i + window_size]
            for i in range(0, len(signal) - window_size + 1, step)]
    return np.array(wins) if wins else np.empty((0, window_size))


def extract_features(window, fs=12000.0):
    """Extract 16 features from one window using the correct sampling rate."""
    x = np.asarray(window, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return None

    x_abs = np.abs(x)
    mean_abs = np.mean(x_abs)
    rms = np.sqrt(np.mean(x ** 2))
    peak2peak = float(np.max(x) - np.min(x))
    kurt = float(stats.kurtosis(x, fisher=True, bias=False))
    skew = float(stats.skew(x, bias=False))
    max_abs = float(np.max(x_abs))
    crest = max_abs / rms if rms > 0 else np.nan
    impulse = max_abs / mean_abs if mean_abs > 0 else np.nan
    shape = rms / mean_abs if mean_abs > 0 else np.nan
    geo = np.exp(np.mean(np.log(x_abs + 1e-10)))
    denom = mean_abs ** 2
    clearance = (max_abs * geo / denom) if denom > 0 else np.nan

    N = len(x)
    yf = fft(x)
    xf = fftfreq(N, 1.0 / fs)[:N // 2]
    mag = np.abs(yf[:N // 2])
    sm = np.sum(mag) + 1e-10
    pm = mag / sm
    se = float(-np.sum(pm * np.log2(pm + 1e-10)))
    pf = float(xf[np.argmax(mag)])
    fc = float(np.sum(xf * mag) / sm)
    bm = (xf >= 2000) & (xf <= 4000)
    be = float(np.sum(mag[bm] ** 2))

    coeffs = pywt.wavedec(x, "db4", level=3)
    we = [float(np.sum(c ** 2)) for c in coeffs]

    feats = {
        "RMS": float(rms), "Peak2Peak": peak2peak,
        "Kurtosis": kurt, "Skewness": skew,
        "Crest": float(crest), "Impulse": float(impulse),
        "Shape": float(shape), "Clearance": float(clearance),
        "SpectralEntropy": se, "PeakFreq": pf,
        "FreqCentroid": fc, "BandEnergy": be,
        "Wavelet_cA3": we[0], "Wavelet_cD3": we[1],
        "Wavelet_cD2": we[2], "Wavelet_cD1": we[3],
    }
    vals = np.array(list(feats.values()), dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        return None
    return feats


def compute_fft(signal, fs=12000.0):
    """Return (frequencies, magnitudes) for plotting."""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1.0 / fs)[:N // 2]
    mag = np.abs(yf[:N // 2]) * 2.0 / N
    return xf, mag
