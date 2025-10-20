"""
timeseries_lime_xai.py
Fast (batched) LIME for time-series models + simple plotting utilities.

Works with models that take (batch, length, channels) = [N, T, F] or [N, F, T].
You pass a model via make_model_predict(model) to auto-handle axes.
"""

from typing import Callable, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Core utilities
# ============================================================

def _ensure_2d_ts(x: np.ndarray, feature_axis: int = -1) -> Tuple[np.ndarray, int, int]:
    """
    Ensure x is a 2D [T, F] time-series. If x is [T], becomes [T,1].
    If feature_axis=0, transpose from [F, T] to [T, F].
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    if feature_axis == 0:
        x = x.T
    T, F = x.shape
    return x.astype(np.float32), T, F


def _make_event_slices(T: int, n_events: int, mode: str = "fixed",
                       beats: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return [E,2] array of [start,end) indices along time.
    mode="fixed": equal-length windows (n_events).
    mode="beats": use provided beat boundary indices (len>=2).
    """
    if mode == "beats":
        if beats is None or len(beats) < 2:
            raise ValueError("mode='beats' needs beats array of boundaries (len>=2).")
        return np.stack([beats[:-1], beats[1:]], axis=1)

    n_events = max(1, int(n_events))
    edges = np.linspace(0, T, n_events + 1, dtype=int)
    slices = np.stack([edges[:-1], edges[1:]], axis=1)
    # ensure non-empty
    mask = slices[:, 1] > slices[:, 0]
    return slices[mask]


def _apply_masks(xTF: np.ndarray,
                 feature_mask: Optional[np.ndarray],
                 event_mask: Optional[np.ndarray],
                 event_slices: Optional[np.ndarray],
                 removal: str = "zero",
                 fill_values: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """
    Apply feature and/or event masks to xTF [T,F].
    removal: "zero" | "mean" | "noise".
    fill_values: {"mean": [F], "noise_std": [F]} (optional).
    """
    x = xTF.copy()
    T, F = x.shape
    if fill_values is None:
        fill_values = {}

    if feature_mask is not None:
        if feature_mask.shape[0] != F:
            raise ValueError("feature_mask length must equal F")
        off = np.where(feature_mask == 0)[0]
        if off.size > 0:
            if removal == "zero":
                x[:, off] = 0.0
            elif removal == "mean":
                m = fill_values.get("mean", np.nanmean(xTF, axis=0))
                x[:, off] = m[off]
            elif removal == "noise":
                std = fill_values.get("noise_std", np.nanstd(xTF, axis=0) + 1e-8)
                rnd = np.random.normal(loc=0.0, scale=std[off], size=(T, off.size))
                x[:, off] = rnd
            else:
                raise ValueError(f"Unknown removal='{removal}'")

    if event_mask is not None:
        if event_slices is None:
            raise ValueError("Provide event_slices if event_mask is used.")
        if event_mask.shape[0] != event_slices.shape[0]:
            raise ValueError("event_mask length must equal #event slices.")
        for e_idx, keep in enumerate(event_mask):
            if keep == 0:
                s, e = event_slices[e_idx]
                if removal == "zero":
                    x[s:e, :] = 0.0
                elif removal == "mean":
                    m = fill_values.get("mean", np.nanmean(xTF, axis=0))
                    x[s:e, :] = m[None, :]
                elif removal == "noise":
                    std = fill_values.get("noise_std", np.nanstd(xTF, axis=0) + 1e-8)
                    rnd = np.random.normal(loc=0.0, scale=std, size=(e - s, x.shape[1]))
                    x[s:e, :] = rnd
                else:
                    raise ValueError(f"Unknown removal='{removal}'")

    return x


def _kernel(dist: np.ndarray, width: float = 0.75) -> np.ndarray:
    """Exponential kernel for LIME weighting."""
    dist = np.asarray(dist, dtype=np.float32)
    return np.exp(-(dist ** 2) / (width ** 2))


def _cosine_distance_to_ones(mask_matrix: np.ndarray) -> np.ndarray:
    """
    Cosine distance between each mask vector and the all-ones vector.
    mask_matrix: [n_masks, D]
    returns [n_masks]
    """
    M = np.asarray(mask_matrix, dtype=np.float32)
    ones = np.ones((M.shape[1],), dtype=np.float32)
    num = (M @ ones)
    den = np.linalg.norm(M, axis=1) * np.linalg.norm(ones)
    cos = num / np.maximum(den, 1e-12)
    return 1.0 - cos


def _fit_weighted_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """
    Weighted ridge regression:
      beta = (X^T W X + l2 I)^(-1) X^T W y
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    W = np.diag(w.astype(np.float32))
    XtW = X.T @ W
    A = XtW @ X + l2 * np.eye(X.shape[1], dtype=np.float32)
    b = XtW @ y
    beta = np.linalg.pinv(A) @ b
    return beta


def _predict_in_batches(model_predict: Callable, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Predict on X [N,T,F] in mini-batches to speed up LIME.
    model_predict: callable([N,T,F]) -> [N,*]
    """
    outs = []
    N = len(X)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        outs.append(np.asarray(model_predict(X[s:e])))
    return np.concatenate(outs, axis=0)


# ============================================================
# Public API — LIME (batched)
# ============================================================

def lime_local(
    model_predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    n_feature_masks: int = 500,
    n_event_masks: int = 800,
    feature_axis: int = -1,
    event_mode: str = "fixed",
    n_events: int = 32,
    beats: Optional[np.ndarray] = None,
    removal: str = "zero",
    p_keep_feature: float = 0.5,
    p_keep_event: float = 0.5,
    kernel_width_feature: float = 0.75,
    kernel_width_event: float = 0.75,
    l2: float = 1e-3,
    class_index: Optional[int] = None,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Local LIME for a single sample x ([T,F] or [T]).
    Returns:
      feature_importance [F], event_importance [E], cell_importance [E,F],
      event_slices [E,2], pred (float), class_index (int or None)
    """
    xTF, T, F = _ensure_2d_ts(x, feature_axis=feature_axis)
    event_slices = _make_event_slices(T, n_events, mode=event_mode, beats=beats)

    # Original prediction
    pred0 = np.asarray(model_predict(xTF[None, ...])).squeeze()
    if pred0.ndim == 0:
        y0 = float(pred0)
    else:
        if class_index is None:
            class_index = int(np.argmax(pred0))
        y0 = float(pred0[class_index])

    fill_values = {
        "mean": np.nanmean(xTF, axis=0),
        "noise_std": np.nanstd(xTF, axis=0) + 1e-8,
    }

    # ---------- Feature LIME (batched) ----------
    Fm = int(n_feature_masks)
    feat_masks = (np.random.rand(Fm, F) < p_keep_feature).astype(np.float32)
    zero_rows = np.where(feat_masks.sum(axis=1) == 0)[0]
    if zero_rows.size > 0:
        feat_masks[zero_rows, np.random.randint(0, F, size=zero_rows.size)] = 1.0

    Xf = np.empty((Fm, T, F), dtype=np.float32)
    for k in range(Fm):
        Xf[k] = _apply_masks(xTF, feature_mask=feat_masks[k], event_mask=None,
                             event_slices=None, removal=removal, fill_values=fill_values)
    y_feat_all = _predict_in_batches(model_predict, Xf, batch_size=batch_size).squeeze()
    if y_feat_all.ndim == 1:
        y_feat = y_feat_all.astype(np.float32)
    else:
        y_feat = y_feat_all[:, class_index].astype(np.float32)

    d_feat = _cosine_distance_to_ones(feat_masks)
    w_feat = _kernel(d_feat, width=kernel_width_feature)
    beta_feat = _fit_weighted_linear(feat_masks, y_feat, w_feat, l2=l2)  # [F]

    # ---------- Event LIME (batched) ----------
    E = event_slices.shape[0]
    Em = int(n_event_masks)
    event_masks = (np.random.rand(Em, E) < p_keep_event).astype(np.float32)
    zero_rows_e = np.where(event_masks.sum(axis=1) == 0)[0]
    if zero_rows_e.size > 0:
        event_masks[zero_rows_e, np.random.randint(0, E, size=zero_rows_e.size)] = 1.0

    Xe = np.empty((Em, T, F), dtype=np.float32)
    for k in range(Em):
        Xe[k] = _apply_masks(xTF, feature_mask=None, event_mask=event_masks[k],
                             event_slices=event_slices, removal=removal, fill_values=fill_values)
    y_event_all = _predict_in_batches(model_predict, Xe, batch_size=batch_size).squeeze()
    if y_event_all.ndim == 1:
        y_event = y_event_all.astype(np.float32)
    else:
        y_event = y_event_all[:, class_index].astype(np.float32)

    d_event = _cosine_distance_to_ones(event_masks)
    w_event = _kernel(d_event, width=kernel_width_event)
    beta_event = _fit_weighted_linear(event_masks, y_event, w_event, l2=l2)  # [E]

    # ---------- Combine to cell map ----------
    f_imp = np.abs(beta_feat - beta_feat.mean()); f_imp = f_imp / (f_imp.sum() + 1e-12)
    e_imp = np.abs(beta_event - beta_event.mean()); e_imp = e_imp / (e_imp.sum() + 1e-12)
    cell = np.outer(e_imp, f_imp)  # [E,F]

    return {
        "feature_importance": beta_feat,
        "event_importance": beta_event,
        "cell_importance": cell,
        "event_slices": event_slices,
        "pred": y0,
        "class_index": class_index,
    }


def lime_global(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_samples: int = 64,
    batch_size: int = 256,
    **lime_local_kwargs
) -> Dict[str, np.ndarray]:
    """
    Global LIME via aggregation of local LIME over a subset of X.
    X: [N,T,F] (or [N,T]) — use consistent T,F across samples.
    Returns global feature/event/cell importances.
    """
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, :, None]
    N, T, F = X.shape

    n_samples = min(n_samples, N)
    idx = np.random.choice(N, size=n_samples, replace=False)

    feat_list, event_list = [], []
    cell_sum = None

    for i, n in enumerate(idx):
        res = lime_local(model_predict, X[n], batch_size=batch_size, **lime_local_kwargs)
        feat_list.append(res["feature_importance"])
        event_list.append(res["event_importance"])

        f_imp = np.abs(res["feature_importance"] - np.mean(res["feature_importance"]))
        e_imp = np.abs(res["event_importance"] - np.mean(res["event_importance"]))
        f_imp = f_imp / (f_imp.sum() + 1e-12)
        e_imp = e_imp / (e_imp.sum() + 1e-12)

        cell = np.outer(e_imp, f_imp)
        if cell_sum is None:
            cell_sum = np.zeros_like(cell)
        cell_sum += cell

    feat_global = np.vstack(feat_list).mean(axis=0)
    event_global = np.vstack(event_list).mean(axis=0)
    cell_global = cell_sum / n_samples

    return {
        "feature_importance_global": feat_global,
        "event_importance_global": event_global,
        "cell_importance_global": cell_global,
    }


# ============================================================
# Plotting utilities (matplotlib, single-plot figures, no colors set)
# ============================================================

# --- Plotly plotting utilities for timeseries LIME ---
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_feature_importance_plotly(feat_imp, feature_names=None, title="Feature importance (LIME)"):
    feat_imp = np.asarray(feat_imp, dtype=np.float32)
    F = feat_imp.shape[0]
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(F)]
    fig = go.Figure(data=[go.Bar(x=feature_names, y=feat_imp)])
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title="Importance", bargap=0.2)
    fig.show()
    return None

def plot_event_importance_plotly(event_imp, event_slices, sr=None, title="Event importance (LIME)"):
    event_imp = np.asarray(event_imp, dtype=np.float32)
    centers = np.array([0.5*(s+e) for s, e in event_slices], dtype=np.float32)
    if sr is not None:
        centers = centers / float(sr)
        xtitle = "Time (s)"
    else:
        xtitle = "Time (samples)"
    fig = go.Figure(data=[go.Scatter(x=centers, y=event_imp, mode="lines+markers")])
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title="Importance")
    fig.show()
    return None

def plot_cell_heatmap_plotly(cell_imp, feature_names=None, title="Cell importance (Event × Feature)"):
    cell_imp = np.asarray(cell_imp, dtype=np.float32)
    E, F = cell_imp.shape
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(F)]
    y_labels = [f"E{i}" for i in range(E)]
    fig = go.Figure(data=go.Heatmap(
        z=cell_imp,
        x=feature_names,
        y=y_labels,
        colorscale="Viridis",
        colorbar_title="Importance",
        zsmooth=False
    ))
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title="Event index")
    fig.show()
    return None

def plot_signal_with_event_overlay_plotly(x, event_imp, event_slices, sr=None, title="Signal + event importance"):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:  # [T, F] -> take first feature
        x = x[:, 0]
    T = x.shape[0]
    step = np.zeros(T, dtype=np.float32)
    for k, (s, e) in enumerate(event_slices):
        step[s:e] = event_imp[k]
    t = np.arange(T, dtype=np.float32)
    if sr is not None:
        t = t / float(sr)
        xtitle = "Time (s)"
    else:
        xtitle = "Time (samples)"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="Signal"), secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=step, mode="lines", name="Event importance (step)"), secondary_y=True)
    fig.update_layout(title=title)
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text="Amplitude", secondary_y=False)
    fig.update_yaxes(title_text="Importance", secondary_y=True)
    fig.show()
    return None


# ============================================================
# Convenience wrapper for Keras/PyTorch models
# ============================================================

def make_model_predict(model) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a callable that accepts [N,T,F] and:
      * uses model.input_shape (if available) to decide whether to transpose to [N,F,T]
      * calls model(x, training=False) if possible (faster than model.predict)
      * otherwise falls back to model.predict(x)
    """
    def _predict(batch_ntf: np.ndarray):
        x = np.asarray(batch_ntf, dtype=np.float32)
        shp = getattr(model, "input_shape", None)
        # Some models return list/tuple for multiple inputs; take the first if needed.
        if isinstance(shp, (list, tuple)) and shp and isinstance(shp[0], (list, tuple)):
            shp = shp[0]
        # Decide axes
        if shp is not None and len(shp) >= 3:
            L_exp, C_exp = shp[1], shp[2]   # Conv1D: (batch, length, channels)
            L_in,  C_in  = x.shape[1], x.shape[2]
            if (L_in, C_in) == (L_exp, C_exp):
                xin = x
            elif (L_in, C_in) == (C_exp, L_exp):
                xin = np.transpose(x, (0, 2, 1))
            else:
                xin = x  # try as-is first
        else:
            xin = x

        # Try faster call path
        try:
            y = model(xin, training=False)
        except Exception:
            y = model.predict(xin)
        return np.asarray(y)
    return _predict
