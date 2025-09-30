"""
Per-patient XAI report generator.
- Plots ECG with attribution overlays
- Summarizes per-lead/per-wave contributions
"""

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from .delineation import detect_r_peaks, delineate_heuristic, aggregate_attribution_by_intervals


def plot_ecg_with_attr(ecg: np.ndarray, attr: np.ndarray, fs: int, title: str = "", out_png: Optional[str] = None):
    """
    ecg: (L, T), attr: (L, T)
    """
    L, T = ecg.shape
    t = np.arange(T) / fs
    plt.figure(figsize=(12, 1.8*L))
    for li in range(L):
        offset = 2.5 * li
        plt.plot(t, ecg[li] + offset, linewidth=1.0)
        # normalized attribution as alpha
        a = np.abs(attr[li])
        a = a / (a.max() + 1e-8)
        plt.fill_between(t, offset, ecg[li] + offset, alpha=0.25, where=a>0, interpolate=True)
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.title(title or "ECG with attribution overlay")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        plt.close()


def summarize_lead_wave(ecg: np.ndarray, attr: np.ndarray, fs: int) -> Dict[str, Dict[str, float]]:
    L, T = ecg.shape
    table = {}
    for li in range(L):
        r_idx = detect_r_peaks(ecg[li], fs)
        intervals = delineate_heuristic(ecg[li], fs, r_idx)
        agg = aggregate_attribution_by_intervals(attr[li], intervals)
        table[f"lead_{li+1}"] = agg
    return table
