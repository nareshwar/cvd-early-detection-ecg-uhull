"""
Delineation utilities for ECG XAI.
- R-peak detection (simple, scipy-based)
- Beat segmentation
- Heuristic P/QRS/ST/T intervals
- Aggregation of attributions by intervals and leads
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.signal import find_peaks


def detect_r_peaks(signal: np.ndarray, fs: int, min_rr_ms: int = 250) -> np.ndarray:
    """
    Detect R-peaks on a single-lead signal using a simple peak finder.
    Parameters
    ----------
    signal : (T,) ECG array for one lead
    fs : sampling rate (Hz)
    min_rr_ms : minimum RR interval in milliseconds
    Returns
    -------
    r_indices : np.ndarray of indices where R-peaks were detected
    """
    # Basic preprocessing: z-score
    x = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    distance = int((min_rr_ms / 1000.0) * fs)
    # Height heuristic: 0.5 std
    peaks, _ = find_peaks(x, distance=distance, height=0.5)
    return peaks


def beats_from_r(signal: np.ndarray, r_idx: np.ndarray, fs: int, pre_ms: int = 250, post_ms: int = 400) -> List[Tuple[int, int]]:
    """
    Return (start, end) indices per beat window around each R-peak.
    """
    pre = int(pre_ms/1000*fs)
    post = int(post_ms/1000*fs)
    beats = []
    T = len(signal)
    for r in r_idx:
        s = max(0, r - pre)
        e = min(T, r + post)
        if e - s > int(0.2*fs):  # at least 200 ms
            beats.append((s, e))
    return beats


def delineate_heuristic(signal: np.ndarray, fs: int, r_idx: np.ndarray) -> List[Dict[str, Tuple[int, int]]]:
    """
    Very simple heuristic delineation per beat:
    Given an R-peak at index r, define windows:
      - P wave:   r - 200ms to r - 80ms
      - QRS:      r - 60ms to r + 60ms
      - ST seg:   r + 60ms to r + 120ms
      - T wave:   r + 120ms to r + 300ms
    Returns a list per beat of dicts mapping interval name -> (start, end).
    Note: Replace with a robust delineator (e.g., wavelet/ML) for production.
    """
    out = []
    for r in r_idx:
        p_start = r - int(0.200*fs)
        p_end   = r - int(0.080*fs)
        qrs_s   = r - int(0.060*fs)
        qrs_e   = r + int(0.060*fs)
        st_s    = r + int(0.060*fs)
        st_e    = r + int(0.120*fs)
        t_s     = r + int(0.120*fs)
        t_e     = r + int(0.300*fs)
        out.append({
            "P":   (max(0, p_start), max(0, p_end)),
            "QRS": (max(0, qrs_s), qrs_e),
            "ST":  (st_s, st_e),
            "T":   (t_s, t_e),
        })
    return out


def aggregate_attribution_by_intervals(attr: np.ndarray, intervals: List[Dict[str, Tuple[int, int]]]) -> Dict[str, float]:
    """
    Sum absolute attribution within each named interval across beats.
    attr: (T,) attribution for one lead
    intervals: list of dicts with interval -> (start, end)
    Returns dict: {"P": value, "QRS": value, "ST": value, "T": value}
    """
    agg = {"P":0.0, "QRS":0.0, "ST":0.0, "T":0.0}
    a = np.abs(attr)
    Tlen = len(a)
    for beat in intervals:
        for name, (s, e) in beat.items():
            s = max(0, min(Tlen, s))
            e = max(0, min(Tlen, e))
            if e > s:
                agg[name] += float(a[s:e].sum())
    total = sum(agg.values()) + 1e-8
    for k in agg:
        agg[k] = agg[k] / total  # normalize
    return agg


def lead_wave_table(attributions: np.ndarray, signals: np.ndarray, fs: int) -> Dict[str, Dict[str, float]]:
    """
    Compute per-lead, per-wave normalized attribution.
    attributions: (L, T) attribution per lead/time
    signals:      (L, T) ECG data
    Returns nested dict: {lead_i: {"P":..., "QRS":..., "ST":..., "T":...}, ...}
    """
    L, T = signals.shape
    out = {}
    for li in range(L):
        r_idx = detect_r_peaks(signals[li], fs=fs)
        intervals = delineate_heuristic(signals[li], fs=fs, r_idx=r_idx)
        agg = aggregate_attribution_by_intervals(attributions[li], intervals)
        out[f"lead_{li+1}"] = agg
    return out
